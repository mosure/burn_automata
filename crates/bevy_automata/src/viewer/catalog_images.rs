use super::catalog::ModelCatalogKey;
use bevy::{
    asset::RenderAssetUsages,
    image::{CompressedImageFormats, ImageSampler, ImageType},
    prelude::Image,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

mod generic;
mod raster;
mod teapot;
mod torus;

use generic::*;
use raster::fill_thumbnail_background;
use teapot::*;
use torus::*;

pub(super) fn catalog_thumbnail_png(key: ModelCatalogKey) -> &'static [u8] {
    match key {
        ModelCatalogKey::Lizard => {
            include_bytes!("../../../../assets/catalog_thumbnails/lizard.png")
        }
        ModelCatalogKey::Butterfly => {
            include_bytes!("../../../../assets/catalog_thumbnails/butterfly.png")
        }
        ModelCatalogKey::Rose => include_bytes!("../../../../assets/catalog_thumbnails/rose.png"),
        ModelCatalogKey::Turtle => {
            include_bytes!("../../../../assets/catalog_thumbnails/turtle.png")
        }
        ModelCatalogKey::Mushroom => {
            include_bytes!("../../../../assets/catalog_thumbnails/mushroom.png")
        }
        ModelCatalogKey::TropicalFish => {
            include_bytes!("../../../../assets/catalog_thumbnails/tropical_fish.png")
        }
        ModelCatalogKey::Sun => {
            include_bytes!("../../../../assets/catalog_thumbnails/sun_with_face.png")
        }
        ModelCatalogKey::Ghost => include_bytes!("../../../../assets/catalog_thumbnails/ghost.png"),
        ModelCatalogKey::Frog => {
            include_bytes!("../../../../assets/catalog_thumbnails/frog_face.png")
        }
        ModelCatalogKey::Apple => {
            include_bytes!("../../../../assets/catalog_thumbnails/red_apple.png")
        }
        ModelCatalogKey::Polka => {
            include_bytes!("../../../../assets/catalog_thumbnails/polka_dotted_0121.png")
        }
        ModelCatalogKey::Bubbly => {
            include_bytes!("../../../../assets/catalog_thumbnails/bubbly_0101.png")
        }
        ModelCatalogKey::Clouds => {
            include_bytes!("../../../../assets/catalog_thumbnails/clouds.png")
        }
        ModelCatalogKey::Galaxy => {
            include_bytes!("../../../../assets/catalog_thumbnails/galaxy.png")
        }
        ModelCatalogKey::Hearts => {
            include_bytes!("../../../../assets/catalog_thumbnails/hearts.png")
        }
        ModelCatalogKey::Rings => include_bytes!("../../../../assets/catalog_thumbnails/rings.png"),
        ModelCatalogKey::Stars => include_bytes!("../../../../assets/catalog_thumbnails/stars.png"),
        ModelCatalogKey::Grid => {
            include_bytes!("../../../../assets/catalog_thumbnails/grid_0040.png")
        }
        ModelCatalogKey::Banded => {
            include_bytes!("../../../../assets/catalog_thumbnails/banded_0037.png")
        }
        ModelCatalogKey::Tree => include_bytes!("../../../../assets/catalog_thumbnails/tree.png"),
        ModelCatalogKey::Snow => include_bytes!("../../../../assets/catalog_thumbnails/snow.png"),
        ModelCatalogKey::Digit0 => {
            include_bytes!("../../../../assets/catalog_thumbnails/digit_0.png")
        }
        ModelCatalogKey::LetterA => {
            include_bytes!("../../../../assets/catalog_thumbnails/letter_a.png")
        }
        ModelCatalogKey::Growing2d => {
            include_bytes!("../../../../assets/catalog_thumbnails/growing_2d.png")
        }
        ModelCatalogKey::Texture2d => {
            include_bytes!("../../../../assets/catalog_thumbnails/texture_2d.png")
        }
        ModelCatalogKey::Growing3dGs => {
            include_bytes!("../../../../assets/catalog_thumbnails/growing_3d_gs.png")
        }
        ModelCatalogKey::UvTorusMorphogen3d => {
            include_bytes!("../../../../assets/catalog_thumbnails/uv_torus_morphogen_3d.png")
        }
        ModelCatalogKey::TeapotMorphogen3d => {
            include_bytes!("../../../../assets/catalog_thumbnails/teapot_morphogen_3d.png")
        }
        ModelCatalogKey::PointMnist => {
            include_bytes!("../../../../assets/catalog_thumbnails/point_mnist.png")
        }
    }
}

pub(super) fn catalog_thumbnail_image(key: ModelCatalogKey) -> Image {
    Image::from_buffer(
        catalog_thumbnail_png(key),
        ImageType::Extension("png"),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::linear(),
        RenderAssetUsages::default(),
    )
    .unwrap_or_else(|_| procedural_catalog_thumbnail_image(key))
}

pub(super) fn catalog_preview_image(key: ModelCatalogKey, seconds: f32) -> Image {
    match key {
        ModelCatalogKey::UvTorusMorphogen3d => procedural_uv_torus_preview_image(seconds),
        ModelCatalogKey::TeapotMorphogen3d => procedural_teapot_preview_image(seconds),
        _ => catalog_thumbnail_image(key),
    }
}

fn procedural_catalog_thumbnail_image(key: ModelCatalogKey) -> Image {
    const WIDTH: u32 = 96;
    const HEIGHT: u32 = 72;
    let mut data = vec![0; (WIDTH * HEIGHT * 4) as usize];
    fill_thumbnail_background(&mut data, WIDTH, HEIGHT);
    match key {
        ModelCatalogKey::Lizard => draw_lizard_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Polka => draw_polka_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Butterfly
        | ModelCatalogKey::Rose
        | ModelCatalogKey::Turtle
        | ModelCatalogKey::Mushroom
        | ModelCatalogKey::TropicalFish
        | ModelCatalogKey::Sun
        | ModelCatalogKey::Ghost
        | ModelCatalogKey::Frog
        | ModelCatalogKey::Apple => draw_growing_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Bubbly
        | ModelCatalogKey::Clouds
        | ModelCatalogKey::Galaxy
        | ModelCatalogKey::Hearts
        | ModelCatalogKey::Rings
        | ModelCatalogKey::Stars
        | ModelCatalogKey::Grid
        | ModelCatalogKey::Banded
        | ModelCatalogKey::Tree
        | ModelCatalogKey::Snow
        | ModelCatalogKey::Digit0
        | ModelCatalogKey::LetterA => draw_texture_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Growing2d => draw_growing_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Texture2d => draw_texture_2d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::Growing3dGs => draw_growing_3d_thumbnail(&mut data, WIDTH, HEIGHT),
        ModelCatalogKey::UvTorusMorphogen3d => {
            draw_uv_torus_morphogen_thumbnail(&mut data, WIDTH, HEIGHT)
        }
        ModelCatalogKey::TeapotMorphogen3d => {
            draw_teapot_morphogen_thumbnail(&mut data, WIDTH, HEIGHT)
        }
        ModelCatalogKey::PointMnist => draw_point_mnist_thumbnail(&mut data, WIDTH, HEIGHT),
    }
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn procedural_uv_torus_preview_image(seconds: f32) -> Image {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 232;
    let mut data = vec![0; (WIDTH * HEIGHT * 4) as usize];
    fill_thumbnail_background(&mut data, WIDTH, HEIGHT);
    draw_uv_torus_target_preview(&mut data, WIDTH, HEIGHT, seconds * 0.45, 0.72);
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}

fn procedural_teapot_preview_image(seconds: f32) -> Image {
    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 232;
    let mut data = vec![0; (WIDTH * HEIGHT * 4) as usize];
    fill_thumbnail_background(&mut data, WIDTH, HEIGHT);
    draw_teapot_target_preview(&mut data, WIDTH, HEIGHT, seconds * 0.35);
    Image::new(
        Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    )
}
