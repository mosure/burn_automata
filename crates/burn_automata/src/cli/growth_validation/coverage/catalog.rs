use super::*;

pub(crate) fn growth_3d_catalog_sanity_report(
    target: MeshTargetArg,
    render_loss: &MultiViewRenderLossReport,
) -> Growth3dCatalogSanityReport {
    let (max_total_loss, min_density_psnr_db, min_color_psnr_db, min_depth_psnr_db) = match target {
        MeshTargetArg::Torus => (0.90, 0.95, 16.0, 14.8),
        MeshTargetArg::Teapot => (0.85, 0.95, 18.0, 18.0),
        MeshTargetArg::Sphere
        | MeshTargetArg::Ellipsoid
        | MeshTargetArg::Cube
        | MeshTargetArg::Cylinder
        | MeshTargetArg::Cone
        | MeshTargetArg::Capsule
        | MeshTargetArg::Pyramid
        | MeshTargetArg::Bicone
        | MeshTargetArg::Dumbbell
        | MeshTargetArg::Cross => (0.90, 0.95, 16.0, 14.8),
    };
    let passed = render_loss.total_loss <= max_total_loss
        && render_loss.density_psnr_db >= min_density_psnr_db
        && render_loss.color_psnr_db >= min_color_psnr_db
        && render_loss.depth_psnr_db >= min_depth_psnr_db;
    Growth3dCatalogSanityReport {
        passed,
        max_total_loss,
        min_density_psnr_db,
        min_color_psnr_db,
        min_depth_psnr_db,
        total_loss: render_loss.total_loss,
        density_psnr_db: render_loss.density_psnr_db,
        color_psnr_db: render_loss.color_psnr_db,
        depth_psnr_db: render_loss.depth_psnr_db,
    }
}
