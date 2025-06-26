use wasm_bindgen::prelude::*;
use image::{self, DynamicImage, GenericImageView, ImageFormat, imageops::FilterType};
use palette::{Lab, Srgb, IntoColor};
use kmeans_colors::get_kmeans;
use js_sys::{Uint8Array, Array, Object, Reflect};
use image::RgbaImage;
#[cfg(not(target_arch = "wasm32"))]
use anyhow::{Result, anyhow, bail};

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
    palette: Option<Array>,
) -> Result<Object, JsValue> {
    // ----------------------
    // 1. Decode the image
    // ----------------------
    let img = image::load_from_memory(&input)
        .map_err(|e| JsValue::from_str(&format!("Unable to decode image: {e}")))?;
    let (orig_w, orig_h) = img.dimensions();

    // Work with RGBA to preserve alpha channel
    let rgba8 = img.to_rgba8();
    let raw = rgba8.into_raw(); // length = w * h * 4

    // Collect Lab pixels from non-transparent areas (alpha > 0) if we need to run k-means
    let mut lab_pixels: Vec<Lab> = Vec::new();
    let mut pixel_to_lab_idx: Vec<Option<usize>> = Vec::with_capacity((raw.len() / 4) as usize);

    if palette.is_none() {
        for chunk in raw.chunks(4) {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];
            let a = chunk[3];
            if a == 0 {
                pixel_to_lab_idx.push(None);
            } else {
                let srgb = Srgb::<u8>::new(r, g, b);
                lab_pixels.push(srgb.into_linear().into_color());
                pixel_to_lab_idx.push(Some(lab_pixels.len() - 1));
            }
        }
    } else {
        // we still need mapping vector size
        pixel_to_lab_idx.resize(raw.len() / 4, None);
    }

    // Determine centroids (Vec<Srgb<u8>>)
    let (centroids, kmeans_indices_opt): (Vec<Srgb<u8>>, Option<Vec<usize>>) = if let Some(js_palette) = palette.clone() {
        let mut tmp = Vec::new();
        for val in js_palette.iter() {
            let s = val.as_string().ok_or_else(|| JsValue::from_str("Palette values must be strings"))?;
            let hex = s.trim_start_matches('#');
            if hex.len() != 6 {
                return Err(JsValue::from_str("Hex color must be 6 characters"));
            }
            let r = u8::from_str_radix(&hex[0..2], 16).map_err(|_| JsValue::from_str("Invalid hex"))?;
            let g = u8::from_str_radix(&hex[2..4], 16).map_err(|_| JsValue::from_str("Invalid hex"))?;
            let b = u8::from_str_radix(&hex[4..6], 16).map_err(|_| JsValue::from_str("Invalid hex"))?;
            tmp.push(Srgb::new(r, g, b));
        }
        (tmp, None)
    } else {
        let kmeans = get_kmeans(n_colors, 20, 1e-4, false, &lab_pixels, 0);
        let centroids_vec: Vec<Srgb<u8>> = kmeans
            .centroids
            .iter()
            .map(|&lab| {
                let rgb_f32: Srgb<f32> = Srgb::from_linear(lab.into_color());
                rgb_f32.into_format::<u8>()
            })
            .collect();
        let indices_vec: Vec<usize> = kmeans.indices.iter().map(|x| *x as usize).collect();
        (centroids_vec, Some(indices_vec))
    };

    // Map each non-transparent pixel to nearest centroid if palette provided, or use kmeans indices
    let mut quantized_raw: Vec<u8> = Vec::with_capacity(raw.len());

    for (i, chunk) in raw.chunks(4).enumerate() {
        let a = chunk[3];
        if a == 0 {
            // keep fully transparent pixel as is
            quantized_raw.extend_from_slice(chunk);
        } else {
            let centroid_color = if let Some(_js_palette) = &palette {
                // need to compute nearest centroid in rgb space quickly (Euclidean)
                let r = chunk[0] as i32;
                let g = chunk[1] as i32;
                let b = chunk[2] as i32;
                let mut best_idx = 0;
                let mut best_dist = i32::MAX;
                for (idx, c) in centroids.iter().enumerate() {
                    let dr = r - c.red as i32;
                    let dg = g - c.green as i32;
                    let db = b - c.blue as i32;
                    let dist = dr*dr + dg*dg + db*db;
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = idx;
                    }
                }
                centroids[best_idx]
            } else {
                // kmeans path
                let idx_in_lab = pixel_to_lab_idx[i].unwrap();
                let cluster_idx = kmeans_indices_opt.as_ref().unwrap()[idx_in_lab];
                centroids[cluster_idx]
            };

            quantized_raw.extend_from_slice(&[centroid_color.red, centroid_color.green, centroid_color.blue, a]);
        }
    }

    let quantized_img = DynamicImage::ImageRgba8(
        RgbaImage::from_raw(orig_w, orig_h, quantized_raw)
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
    let mut buf = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut buf);
        upscaled
            .write_to(&mut cursor, ImageFormat::Png)
            .map_err(|e| JsValue::from_str(&format!("PNG encode error: {e}")))?;
    }

    let encoded = buf;

    // Build palette hex strings
    let palette_hex: Vec<String> = centroids
        .iter()
        .map(|c| format!("{:02X}{:02X}{:02X}", c.red, c.green, c.blue))
        .collect();

    // Convert to JS types
    let img_js = Uint8Array::from(encoded.as_slice());
    let palette_js = Array::new();
    for hex in palette_hex {
        palette_js.push(&JsValue::from_str(&hex));
    }

    let result = Object::new();
    Reflect::set(&result, &JsValue::from_str("image"), &img_js)?;
    Reflect::set(&result, &JsValue::from_str("palette"), &palette_js)?;

    Ok(result)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn pixelate_bytes(
    input: &[u8],
    n_colors: usize,
    scale: u32,
    output_size: Option<u32>,
    palette_hex: Option<&[String]>,
) -> Result<(Vec<u8>, Vec<String>)> {
    // Decode
    let img = image::load_from_memory(input)?;
    let (orig_w, orig_h) = img.dimensions();
    let rgba8 = img.to_rgba8();
    let raw = rgba8.into_raw();

    // Build lab pixels if needed
    let mut lab_pixels: Vec<Lab> = Vec::new();
    let mut pixel_to_lab_idx: Vec<Option<usize>> = Vec::with_capacity(raw.len() / 4);
    let use_kmeans = palette_hex.is_none();
    if use_kmeans {
        for chunk in raw.chunks(4) {
            let a = chunk[3];
            if a == 0 {
                pixel_to_lab_idx.push(None);
            } else {
                let srgb = Srgb::<u8>::new(chunk[0], chunk[1], chunk[2]);
                lab_pixels.push(srgb.into_linear().into_color());
                pixel_to_lab_idx.push(Some(lab_pixels.len() - 1));
            }
        }
    } else {
        pixel_to_lab_idx.resize(raw.len() / 4, None);
    }

    // centroids
    let (centroids, kmeans_indices_opt): (Vec<Srgb<u8>>, Option<Vec<usize>>) = if let Some(palette_list) = palette_hex {
        let mut tmp = Vec::new();
        for s in palette_list {
            let hex = s.trim_start_matches('#');
            if hex.len() != 6 { bail!("Invalid hex length"); }
            let r = u8::from_str_radix(&hex[0..2], 16).map_err(|e| anyhow!(e))?;
            let g = u8::from_str_radix(&hex[2..4], 16).map_err(|e| anyhow!(e))?;
            let b = u8::from_str_radix(&hex[4..6], 16).map_err(|e| anyhow!(e))?;
            tmp.push(Srgb::new(r, g, b));
        }
        (tmp, None)
    } else {
        let kmeans = get_kmeans(n_colors, 20, 1e-4, false, &lab_pixels, 0);
        let centroids_vec: Vec<Srgb<u8>> = kmeans.centroids.iter().map(|&lab| {
            let rgb_f32: Srgb<f32> = Srgb::from_linear(lab.into_color());
            rgb_f32.into_format::<u8>()
        }).collect();
        let indices_vec: Vec<usize> = kmeans.indices.iter().map(|x| *x as usize).collect();
        (centroids_vec, Some(indices_vec))
    };

    // recolor
    let mut quantized_raw = Vec::with_capacity(raw.len());
    for (i, chunk) in raw.chunks(4).enumerate() {
        let a = chunk[3];
        if a == 0 {
            quantized_raw.extend_from_slice(chunk);
        } else {
            let centroid = if palette_hex.is_some() {
                // nearest RGB
                let r = chunk[0] as i32;
                let g = chunk[1] as i32;
                let b = chunk[2] as i32;
                let mut best = 0;
                let mut best_dist = i32::MAX;
                for (idx, c) in centroids.iter().enumerate() {
                    let dr = r - c.red as i32;
                    let dg = g - c.green as i32;
                    let db = b - c.blue as i32;
                    let d = dr*dr+dg*dg+db*db;
                    if d < best_dist { best_dist = d; best = idx; }
                }
                centroids[best]
            } else {
                let idx_in_lab = pixel_to_lab_idx[i].unwrap();
                let cluster = kmeans_indices_opt.as_ref().unwrap()[idx_in_lab];
                centroids[cluster]
            };
            quantized_raw.extend_from_slice(&[centroid.red, centroid.green, centroid.blue, a]);
        }
    }

    let quantized_img = DynamicImage::ImageRgba8(RgbaImage::from_raw(orig_w, orig_h, quantized_raw).ok_or_else(|| anyhow!("rebuild image buffer"))?);

    // scale
    let max_side = orig_w.max(orig_h) as f32;
    let ratio_down = scale as f32 / max_side;
    let down_w = ((orig_w as f32)*ratio_down).round().max(1.0) as u32;
    let down_h = ((orig_h as f32)*ratio_down).round().max(1.0) as u32;
    let downscaled = image::imageops::resize(&quantized_img, down_w, down_h, FilterType::Nearest);

    let (final_w, final_h) = if let Some(sz) = output_size { let r= sz as f32 / max_side; (((orig_w as f32)*r).round() as u32, ((orig_h as f32)*r).round() as u32) } else { (orig_w, orig_h) };
    let upscaled = image::imageops::resize(&downscaled, final_w, final_h, FilterType::Nearest);

    let mut buf = Vec::new();
    {
        let mut cursor = std::io::Cursor::new(&mut buf);
        upscaled.write_to(&mut cursor, ImageFormat::Png).map_err(|e| anyhow!(e))?;
    }

    let palette_hex_out: Vec<String> = centroids.iter().map(|c| format!("{:02X}{:02X}{:02X}", c.red, c.green, c.blue)).collect();

    Ok((buf, palette_hex_out))
}
