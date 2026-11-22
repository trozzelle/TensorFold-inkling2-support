//! Model-free dependent FP32 operators with untracked resources and one externally completed command buffer.
const std = @import("std");
const mtl = @import("metal");
pub const counters = @import("core/gpu_counters.zig");
const source = @import("kernel_sources").counter_probe;
pub const element_count: u32 = 4096;
const classes = [_]counters.Class{ .dense, .activation, .copy };

pub const Report = struct {
    spans: [3]counters.Span,
    aggregates: [counters.class_count]counters.Aggregate,
    calibration: counters.Calibration,
    in_flight_rejected: bool,
};

pub const Consumer = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    pipeline: mtl.Pipeline,
    a: mtl.Buffer,
    b: mtl.Buffer,
    indirect: mtl.Buffer,
    gate: mtl.SharedEvent,
    next_gate: u64 = 1,

    pub fn init(device: mtl.Device) !Consumer {
        const queue = try device.queue();
        errdefer queue.deinit();
        const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const pipeline = try mtl.Pipeline.init(device, library, "tf_counter_affine", false);
        errdefer pipeline.deinit();
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const a = try device.buffer(element_count * @sizeOf(f32), options);
        errdefer a.deinit();
        const b = try device.buffer(element_count * @sizeOf(f32), options);
        errdefer b.deinit();
        const indirect = try device.buffer(3 * @sizeOf(u32), options);
        errdefer indirect.deinit();
        @memcpy(indirect.slice(u32, 3), &[_]u32{ element_count / 256, 1, 1 });
        const gate = try device.sharedEvent();
        return .{
            .device = .{ .id = mtl.objc.retain(device.id) },
            .queue = queue,
            .pipeline = pipeline,
            .a = a,
            .b = b,
            .indirect = indirect,
            .gate = gate,
        };
    }

    pub fn deinit(self: Consumer) void {
        self.gate.deinit();
        self.indirect.deinit();
        self.b.deinit();
        self.a.deinit();
        self.pipeline.deinit();
        self.queue.deinit();
        self.device.deinit();
    }

    /// The only CPU wait is the consumer's original final completion wait, never a span wait.
    pub fn execute(self: *Consumer, recorder: *counters.Recorder) !void {
        for (self.a.slice(f32, element_count), 0..) |*value, i| value.* = initial(i);
        @memset(self.b.slice(f32, element_count), std.math.nan(f32));
        const cb = self.queue.commandBuffer();
        const gate_value = self.next_gate;
        self.next_gate += 1;
        cb.waitFor(self.gate, gate_value);
        try recorder.beginCommand(cb);
        var encoder = cb.compute(.serial);
        var submitted = false;
        errdefer {
            if (submitted) {
                self.gate.set(gate_value);
                cb.wait();
                recorder.discardAfterCompletion() catch {};
            } else recorder.cancelEncoding(&encoder) catch {};
        }
        for (classes, 0..) |kind, i| {
            try recorder.beginSpan(&encoder, kind);
            self.encodeOperator(encoder, i);
            try recorder.endSpan(&encoder);
        }
        try recorder.finishEncoding(&encoder);
        cb.commit();
        submitted = true;
        if (recorder.enabled()) {
            try expectFailure(error.CounterCommandNotCompleted, recorder.readAfterCompletion());
            try expectFailure(error.CounterRecordingInFlight, recorder.beginCommand(cb));
            try expectFailure(error.CounterCommandNotCompleted, recorder.discardAfterCompletion());
            try expectFailure(error.CounterCommandNotCompleted, recorder.deinit());
        }
        self.gate.set(gate_value);
        cb.wait();
        if (cb.failure() != null) return error.CounterConsumerGpuFailure;
        _ = try recorder.readAfterCompletion();
        for (self.b.slice(f32, element_count), 0..) |value, i| {
            const expected = initial((i + 3) % element_count) + 1;
            if (@as(u32, @bitCast(value)) != @as(u32, @bitCast(expected))) return error.CounterConsumerOutputMismatch;
        }
    }

    /// Operator `i` of the dependent chain, `a` to `b` and back, on `encoder`.
    fn encodeOperator(self: *const Consumer, encoder: mtl.ComputeEncoder, i: usize) void {
        const scales = [_]f32{ 2, 1, 0.5 };
        const biases = [_]f32{ 1, -3, 2 };
        const input = if (i % 2 == 0) self.a else self.b;
        const output = if (i % 2 == 0) self.b else self.a;
        encoder.setPipeline(self.pipeline);
        encoder.setBuffer(input, 0, 0);
        encoder.setBuffer(output, 0, 1);
        encoder.setValue(element_count, 2);
        encoder.setValue(scales[i], 3);
        encoder.setValue(biases[i], 4);
        encoder.useResource(input, mtl.ResourceUsage.read);
        encoder.useResource(output, mtl.ResourceUsage.write);
        encoder.useResource(self.indirect, mtl.ResourceUsage.read);
        if (i == 1) {
            encoder.dispatchIndirect(self.indirect, 0, mtl.Size.of(256, 1, 1));
        } else encoder.dispatchThreads(mtl.Size.of(element_count, 1, 1), mtl.Size.of(256, 1, 1));
        encoder.barrier();
    }

    /// `execute` with the operators spread over `buffers` command buffers, each committed as soon as it is encoded and held at
    /// the start by an event until the one before has run (the probe's buffers are untracked, so the queue does not order them):
    /// the recording carries on across them and the only CPU wait is the last one's.
    pub fn executeChained(self: *Consumer, recorder: *counters.Recorder, buffers: usize) !void {
        std.debug.assert(buffers >= 1 and buffers <= classes.len);
        for (self.a.slice(f32, element_count), 0..) |*value, i| value.* = initial(i);
        @memset(self.b.slice(f32, element_count), std.math.nan(f32));
        const first_gate = self.next_gate;
        self.next_gate += buffers;
        var all: [classes.len]mtl.CommandBuffer = undefined;
        var made: usize = 1;
        var cb = self.queue.commandBuffer();
        all[0] = cb;
        cb.waitFor(self.gate, first_gate);
        try recorder.beginCommand(cb);
        var encoder = cb.compute(.serial);
        for (classes, 0..) |kind, i| {
            try recorder.beginSpan(&encoder, kind);
            self.encodeOperator(encoder, i);
            try recorder.endSpan(&encoder);
            if (made < buffers and i + 1 == made * classes.len / buffers) {
                try recorder.endCommand(&encoder);
                cb.signal(self.gate, first_gate + made);
                cb.commit();
                cb = self.queue.commandBuffer();
                all[made] = cb;
                cb.waitFor(self.gate, first_gate + made);
                made += 1;
                try recorder.continueCommand(cb);
                encoder = cb.compute(.serial);
            }
        }
        try recorder.finishEncoding(&encoder);
        cb.commit();
        self.gate.set(first_gate);
        cb.wait();
        for (all[0..made]) |one| if (one.failure() != null) return error.CounterConsumerGpuFailure;
        _ = try recorder.readAfterCompletion();
        for (self.b.slice(f32, element_count), 0..) |value, i| {
            const expected = initial((i + 3) % element_count) + 1;
            if (@as(u32, @bitCast(value)) != @as(u32, @bitCast(expected))) return error.CounterConsumerOutputMismatch;
        }
    }

    pub fn probe(self: *Consumer, allocator: std.mem.Allocator) !Report {
        var plain = try counters.Recorder.init(allocator, self.device, .{});
        defer plain.deinit() catch unreachable;
        try self.execute(&plain);
        var expected_bits: [element_count]u32 = undefined;
        for (self.b.slice(f32, element_count), &expected_bits) |value, *bits| bits.* = @bitCast(value);
        var instrumented = try counters.Recorder.init(allocator, self.device, .{ .enabled = true, .max_spans = 3 });
        defer instrumented.deinit() catch unreachable;
        try self.execute(&instrumented);
        for (self.b.slice(f32, element_count), expected_bits) |value, bits| {
            if (@as(u32, @bitCast(value)) != bits) return error.CounterInstrumentationChangedBits;
        }
        const spans = try instrumented.readAfterCompletion();
        if (spans.len != 3) return error.CounterConsumerSpanCountMismatch;
        var report: Report = .{
            .spans = undefined,
            .aggregates = instrumented.aggregates(),
            .calibration = instrumented.calibration orelse return error.MissingClockCalibration,
            .in_flight_rejected = true,
        };
        @memcpy(&report.spans, spans);
        var seconds: f64 = 0;
        for (report.spans, classes) |span, kind| {
            if (span.kind != kind) return error.CounterConsumerClassMismatch;
            seconds += span.gpu_seconds;
        }
        if (seconds <= 0 or !std.math.isFinite(seconds)) return error.InvalidCounterSample;
        return report;
    }
};

fn initial(index: usize) f32 {
    const value: i32 = @as(i32, @intCast(index % 127)) - 63;
    return @floatFromInt(value);
}

fn expectFailure(expected: anyerror, result: anytype) !void {
    if (result) |_| {
        return error.CounterExpectedFailureMissing;
    } else |err| if (err != expected) return err;
}

fn supportedDevice() !mtl.Device {
    const device = mtl.Device.init() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (mtl.counters.query(device).mode() == .unsupported) {
        device.deinit();
        return error.SkipZigTest;
    }
    return device;
}

test "sampled dependent direct and indirect operators preserve FP32 bits and repeat without stale samples" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    const first = try consumer.probe(std.testing.allocator);
    const second = try consumer.probe(std.testing.allocator);
    for (first.spans, second.spans, classes) |a, b, kind| {
        try std.testing.expectEqual(kind, a.kind);
        try std.testing.expectEqual(kind, b.kind);
        try std.testing.expect(b.begin_timestamp >= a.end_timestamp);
        try std.testing.expectEqual(@as(u64, 1), second.aggregates[@backingInt(kind)].calls);
    }
    try std.testing.expect(first.in_flight_rejected and second.in_flight_rejected);
}

test "bounded recorder rejects nesting, overflow, open spans and premature reuse then cancels and reuses" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 1 });
    defer recorder.deinit() catch unreachable;
    const cb = queue.commandBuffer();
    try recorder.beginCommand(cb);
    var encoder = cb.compute(.serial);
    errdefer recorder.cancelEncoding(&encoder) catch {};
    try std.testing.expectError(error.CounterEncoderStillOpen, recorder.deinit());
    try std.testing.expectError(error.CounterEncoderStillOpen, recorder.discardAfterCompletion());
    try std.testing.expectError(error.CounterRecordingInFlight, recorder.beginCommand(cb));
    try std.testing.expectError(error.NoActiveCounterSpan, recorder.endSpan(&encoder));
    try recorder.beginSpan(&encoder, .dense);
    try std.testing.expectError(error.NestedCounterSpan, recorder.beginSpan(&encoder, .copy));
    try std.testing.expectError(error.UnclosedCounterSpan, recorder.finishEncoding(&encoder));
    try recorder.endSpan(&encoder);
    try std.testing.expectError(error.CounterSpanCapacityExceeded, recorder.beginSpan(&encoder, .copy));
    try recorder.finishEncoding(&encoder);
    try std.testing.expectError(error.CounterCommandNotCompleted, recorder.readAfterCompletion());
    try recorder.cancelEncoding(&encoder);
    const next = queue.commandBuffer();
    try recorder.beginCommand(next);
    encoder = next.compute(.serial);
    try recorder.finishEncoding(&encoder);
    next.commit();
    next.wait();
    if (next.failure() != null) return error.CounterConsumerGpuFailure;
    try std.testing.expectEqual(@as(usize, 0), (try recorder.readAfterCompletion()).len);
    for (recorder.aggregates()) |aggregate| try std.testing.expectEqual(@as(u64, 0), aggregate.calls);
    try recorder.discardAfterCompletion();
}

test "completed recorder reuses bounded samples and read results do not double count" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 3 });
    defer recorder.deinit() catch unreachable;
    try consumer.execute(&recorder);
    const first = try recorder.readAfterCompletion();
    const prior_end = first[2].end_timestamp;
    _ = try recorder.readAfterCompletion();
    for (classes) |kind| try std.testing.expectEqual(@as(u64, 1), recorder.aggregates()[@backingInt(kind)].calls);
    try consumer.execute(&recorder);
    const second = try recorder.readAfterCompletion();
    try std.testing.expectEqual(@as(usize, 3), second.len);
    try std.testing.expect(second[0].begin_timestamp >= prior_end);
    for (classes) |kind| try std.testing.expectEqual(@as(u64, 1), recorder.aggregates()[@backingInt(kind)].calls);
}

test "a recording carries on across command buffers committed one after the other and reads every span once they finish" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = classes.len });
    defer recorder.deinit() catch unreachable;
    for (1..classes.len + 1) |buffers| {
        try consumer.executeChained(&recorder, buffers);
        const spans = try recorder.readAfterCompletion();
        try std.testing.expectEqual(classes.len, spans.len);
        for (spans, classes) |span, kind| {
            try std.testing.expectEqual(kind, span.kind);
            try std.testing.expect(span.gpu_seconds > 0 and std.math.isFinite(span.gpu_seconds));
            try std.testing.expectEqual(@as(u64, 1), recorder.aggregates()[@backingInt(kind)].calls);
        }
        for (spans[1..], spans[0 .. spans.len - 1]) |next, prev| try std.testing.expect(next.begin_timestamp >= prev.end_timestamp);
    }
}

test "a recorder that records nothing still ends a command buffer and carries on in the next, with the bits as they were" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    var plain = try counters.Recorder.init(std.testing.allocator, device, .{});
    defer plain.deinit() catch unreachable;
    for (1..classes.len + 1) |buffers| {
        try consumer.executeChained(&plain, buffers);
        try std.testing.expectEqual(@as(usize, 0), (try plain.readAfterCompletion()).len);
    }
}

test "a recording leaves a command buffer only between spans, once, for an uncommitted buffer" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    const queue = consumer.queue;
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 2 });
    defer recorder.deinit() catch unreachable;
    const first = queue.commandBuffer();
    // nothing is recording yet
    try std.testing.expectError(error.CounterNotRecording, recorder.continueCommand(first));
    try recorder.beginCommand(first);
    var encoder = first.compute(.serial);
    errdefer recorder.cancelEncoding(&encoder) catch {};
    // the buffer is still the recording's: it cannot be given another before it ends, and not inside a span
    try std.testing.expectError(error.CounterNotRecording, recorder.continueCommand(first));
    try recorder.beginSpan(&encoder, .dense);
    consumer.encodeOperator(encoder, 0);
    try std.testing.expectError(error.UnclosedCounterSpan, recorder.endCommand(&encoder));
    try recorder.endSpan(&encoder);
    try recorder.endCommand(&encoder);
    // between buffers: no encoder to end or finish, and no way to start a span
    try std.testing.expectError(error.CounterNotRecording, recorder.endCommand(&encoder));
    try std.testing.expectError(error.CounterNotRecording, recorder.finishEncoding(&encoder));
    try std.testing.expectError(error.CounterNotRecording, recorder.beginSpan(&encoder, .copy));
    // a buffer that was committed cannot carry the recording on
    const spent = queue.commandBuffer();
    var spent_encoder = spent.compute(.serial);
    spent_encoder.end();
    spent.commit();
    spent.wait();
    try std.testing.expectError(error.CounterCommandAlreadySubmitted, recorder.continueCommand(spent));
    // an uncommitted one can, once
    const next = queue.commandBuffer();
    try recorder.continueCommand(next);
    try std.testing.expectError(error.CounterNotRecording, recorder.continueCommand(next));
    encoder = next.compute(.serial);
    try recorder.beginSpan(&encoder, .copy);
    consumer.encodeOperator(encoder, 2);
    try recorder.endSpan(&encoder);
    try recorder.finishEncoding(&encoder);
    first.commit();
    next.commit();
    next.wait();
    if (first.failure() != null or next.failure() != null) return error.CounterConsumerGpuFailure;
    const spans = try recorder.readAfterCompletion();
    try std.testing.expectEqual(@as(usize, 2), spans.len);
    try std.testing.expectEqual(counters.Class.dense, spans[0].kind);
    try std.testing.expectEqual(counters.Class.copy, spans[1].kind);
    try recorder.discardAfterCompletion();
}
