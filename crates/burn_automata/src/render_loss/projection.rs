use super::{EPS, RenderViewPreset};
use crate::render_loss::math::{cross3, dot3, normalize3};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ProjectedPoint {
    pub(super) x: f32,
    pub(super) y: f32,
    pub(super) depth: f32,
}

pub(super) fn project_positions(
    view: RenderViewPreset,
    positions: &[[f32; 4]],
    world_scale: f32,
) -> Vec<ProjectedPoint> {
    let (right, up, forward) = view_basis(view);
    positions
        .iter()
        .map(|position| {
            let depth =
                (0.5 + 0.5 * dot3(*position, forward) / world_scale.max(EPS)).clamp(0.0, 1.0);
            ProjectedPoint {
                x: dot3(*position, right),
                y: dot3(*position, up),
                depth,
            }
        })
        .collect()
}

pub(super) fn view_basis(view: RenderViewPreset) -> ([f32; 3], [f32; 3], [f32; 3]) {
    match view {
        RenderViewPreset::Xy => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        RenderViewPreset::Xz => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        RenderViewPreset::Yz => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        RenderViewPreset::Iso => {
            let right = normalize3([1.0, -1.0, 0.0]);
            let up = normalize3([1.0, 1.0, 2.0]);
            let forward = normalize3(cross3(right, up));
            (right, up, forward)
        }
    }
}
