//! What the server reads from the environment, read the same way by tf-inkling-run's modes: the prompt chunk's rows
//! (INKLING_PREFILL_ROWS) and the draft head's settings (INKLING_DRAFT_DEPTH, INKLING_DRAFT_CONFIDENCE, INKLING_DRAFT_BITS).
const std = @import("std");
const tf = @import("tensorfold");
const ik = tf.inkling2;

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// Rows of a prompt chunk at most, as `native/inkling2.zig` reads them: INKLING_PREFILL_ROWS, a multiple of 64 from 64 to 8192
/// (null when unset: the load takes 4,096, or 512 on the per-operation path).
pub fn prefillRows() !?u32 {
    return ik.state.parsePrefillRows(env("INKLING_PREFILL_ROWS")) catch error.PrefillRowsIsAMultipleOf64From64To8192;
}

/// How the draft head runs: how a round drafts (INKLING_DRAFT_DEPTH and INKLING_DRAFT_CONFIDENCE) and the width of its layers'
/// weights (INKLING_DRAFT_BITS).
pub const Draft = struct { drafting: ik.dspark.Drafting, bits: ik.dspark_weights.Bits };

pub fn draft() !Draft {
    const drafting = ik.dspark.Drafting.parse(env("INKLING_DRAFT_DEPTH"), env("INKLING_DRAFT_CONFIDENCE")) catch |err| return switch (err) {
        error.BadDraftDepth => error.DraftDepthIsOneToSevenOrAuto,
        error.BadDraftConfidence => error.DraftConfidenceIsAboveZeroAndBelowOneOrOff,
    };
    const bits = ik.dspark_weights.Bits.parse(env("INKLING_DRAFT_BITS")) catch return error.DraftBitsIsEightOrSixteen;
    return .{ .drafting = drafting, .bits = bits };
}
