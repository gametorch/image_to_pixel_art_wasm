use wasm_bindgen::prelude::*;
use image::{self, DynamicImage, GenericImageView, ImageFormat, imageops::FilterType};
use palette::{cast::from_component_slice, Lab, Srgb, IntoColor};
use kmeans_colors::get_kmeans;

/// Convert an input image to low‐color pixel art.
///
/// Steps performed:
/// 1. Run k-means (`k = n_colors`) in Lab color space to build a palette.
/// 2. Re-color every pixel with its closest palette entry.
/// 3. Down-scale, keeping the aspect ratio, so that the longest side equals `scale` (nearest-neighbour).
/// 4. Up-scale back to the original size, or to an optional `output_size` (also nearest-neighbour).
///
/// The returned `Vec<u8>` contains a PNG-encoded image that can be turned into a
/// `Blob`, `ImageBitmap`, etc. in JavaScript.
#[wasm_bindgen]
pub fn pixelate(
    input: Vec<u8>,
    n_colors: usize,
    scale: u32,
    output_size: Option<u32>,
) -> Result<Vec<u8>, JsValue> {
    // ----------------------
    // 1. Decode the image
    // ----------------------
    let img = image::load_from_memory(&input)
        .map_err(|e| JsValue::from_str(&format!("Unable to decode image: {e}")))?;
    let (orig_w, orig_h) = img.dimensions();

    // ----------------------
    // 2. Build the color palette (k-means)
    // ----------------------
    // Extract RGB pixels as bytes
    let rgb8 = img.to_rgb8();
    let raw = rgb8.into_raw(); // length = w * h * 3

    // Cast the &[u8] slice to a slice of sRGB pixels
    let srgb_slice = from_component_slice::<Srgb<u8>>(&raw);

    // Convert to Lab for perceptual k-means
    let lab_pixels: Vec<Lab> = srgb_slice
        .iter()
        .map(|c| c.into_linear().into_color())
        .collect();

    // Run k-means (20 iterations max, small convergence threshold)
    let kmeans = get_kmeans(n_colors, 20, 1e-4, false, &lab_pixels, 0);

    // Convert centroids back to sRGB (8-bit)
    let centroids: Vec<Srgb<u8>> = kmeans
        .centroids
        .iter()
        .map(|&lab| {
            let rgb_f32: Srgb<f32> = Srgb::from_linear(lab.into_color());
            rgb_f32.into_format::<u8>()
        })
        .collect();

    // Re-color every pixel using the centroid index map
    let mut quantized_raw: Vec<u8> = Vec::with_capacity(raw.len());
    for &idx in &kmeans.indices {
        let c = centroids[idx as usize];
        quantized_raw.extend_from_slice(&[c.red, c.green, c.blue]);
    }

    let quantized_img = DynamicImage::ImageRgb8(
        image::ImageBuffer::from_raw(orig_w, orig_h, quantized_raw)
            .ok_or_else(|| JsValue::from_str("Failed to rebuild image buffer"))?,
    );

    // ----------------------
    // 3. Down-scale
    // ----------------------
    let max_side = orig_w.max(orig_h) as f32;
    let ratio_down = scale as f32 / max_side;
    let down_w = ((orig_w as f32) * ratio_down).round().max(1.0) as u32;
    let down_h = ((orig_h as f32) * ratio_down).round().max(1.0) as u32;

    let downscaled = image::imageops::resize(&quantized_img, down_w, down_h, FilterType::Nearest);

    // ----------------------
    // 4. Up-scale
    // ----------------------
    let (final_w, final_h) = if let Some(size) = output_size {
        let ratio_up = size as f32 / max_side;
        (
            ((orig_w as f32) * ratio_up).round().max(1.0) as u32,
            ((orig_h as f32) * ratio_up).round().max(1.0) as u32,
        )
    } else {
        (orig_w, orig_h)
    };

    let upscaled = image::imageops::resize(&downscaled, final_w, final_h, FilterType::Nearest);

    // ----------------------
    // 5. PNG encode and return
    // ----------------------
    let mut cursor = std::io::Cursor::new(Vec::new());
    upscaled
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| JsValue::from_str(&format!("PNG encode error: {e}")))?;

    return Ok(cursor.into_inner());
}
