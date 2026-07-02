use super::prelude::*;

pub(crate) fn assert_compact_capture(metrics: CaptureMetrics) {
    let occupancy = metrics.lit_pixels as f32 / (metrics.width * metrics.height) as f32;
    assert!(
        occupancy > 0.002,
        "automata render is too sparse or blank: occupancy={occupancy:.6}, metrics={metrics:?}"
    );
    assert!(
        occupancy < 0.45,
        "automata render covers too much of the frame: occupancy={occupancy:.6}, metrics={metrics:?}"
    );
    assert!(
        metrics.bbox_width() < metrics.width * 9 / 10
            && metrics.bbox_height() < metrics.height * 9 / 10,
        "automata render bbox is too large: {:?}",
        metrics
    );
}

#[derive(Resource, Default)]
pub(crate) struct RenderCapture {
    pub(crate) captured: bool,
    pub(crate) metrics: Option<CaptureMetrics>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CaptureMetrics {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) lit_pixels: usize,
    pub(crate) max_delta: u8,
    pub(crate) min_x: u32,
    pub(crate) max_x: u32,
    pub(crate) min_y: u32,
    pub(crate) max_y: u32,
}

impl CaptureMetrics {
    pub(crate) fn bbox_width(&self) -> u32 {
        if self.lit_pixels == 0 {
            0
        } else {
            self.max_x - self.min_x + 1
        }
    }

    pub(crate) fn bbox_height(&self) -> u32 {
        if self.lit_pixels == 0 {
            0
        } else {
            self.max_y - self.min_y + 1
        }
    }
}

pub(crate) fn capture_metrics(image: &Image) -> Option<CaptureMetrics> {
    let data = image.data.as_ref()?;
    let width = image.width();
    let height = image.height();
    let background = data.get(0..3)?;
    let mut metrics = CaptureMetrics {
        width,
        height,
        lit_pixels: 0,
        max_delta: 0,
        min_x: width,
        max_x: 0,
        min_y: height,
        max_y: 0,
    };
    for (pixel_index, rgba) in data.chunks_exact(4).enumerate() {
        let delta = rgba[0]
            .abs_diff(background[0])
            .max(rgba[1].abs_diff(background[1]))
            .max(rgba[2].abs_diff(background[2]));
        metrics.max_delta = metrics.max_delta.max(delta);
        let lit = delta > 8;
        if lit {
            let x = pixel_index as u32 % width;
            let y = pixel_index as u32 / width;
            metrics.lit_pixels += 1;
            metrics.min_x = metrics.min_x.min(x);
            metrics.max_x = metrics.max_x.max(x);
            metrics.min_y = metrics.min_y.min(y);
            metrics.max_y = metrics.max_y.max(y);
        }
    }
    Some(metrics)
}
