//! The weights and scratch in one residency set for the keepalive, when the 70% limit holds them.
const std = @import("std");
const mtl = @import("metal");
const Resident = mtl.resident.Resident;

pub fn hold(h: anytype) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    const m = h.model;
    for (m.checkpoint.shards.items) |sh| try list.append(h.gpa, sh.buffer);
    try list.appendSlice(h.gpa, m.weights.tables);
    try mtl.resident.fields.append(h.gpa, &list, &h.metal.scratch);
    h.resident = try Resident.hold(h.gpa, h.io, m.device, m.queue, list.items, "inkling2");
    if (h.resident) |*r| h.metal.setResident(r.follow());
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = m.queue };
}

pub fn drop(h: anytype) void {
    h.metal.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.queue };
}
