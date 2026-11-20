//! Native Metal Inkling 2 Small with EXL3 routed experts (kortexa-inkling2-exl3-v1 packs), text only.
pub const config = @import("config.zig");
pub const plan = @import("plan.zig");
pub const expert_pack = @import("expert_pack.zig");
pub const weights = @import("weights.zig");
pub const state = @import("state.zig");
pub const model = @import("model.zig");
pub const Model = model.Model;
pub const forward = @import("forward.zig");
pub const budget = @import("budget.zig");
pub const residency = @import("residency.zig");
pub const backend = @import("backend.zig");

test {
    _ = config;
    _ = plan;
    _ = expert_pack;
    _ = state;
    _ = budget;
}
