use wasm_bindgen::prelude::*;
use image::{self, DynamicImage, GenericImageView, ImageFormat, imageops::FilterType};
use palette::{Lab, Srgb, IntoColor};
use kmeans_colors::get_kmeans;
use js_sys::{Uint8Array, Array, Object, Reflect};
use image::RgbaImage;
#[cfg(not(target_arch = "wasm32"))]
use anyhow::{Result, anyhow, bail};

const LUMINANCE_WEIGHT: f32 = 0.0;

// ------------------------------------------------------------
// Luminance‐based downscaling helpers
// ------------------------------------------------------------

/// Strategy used when selecting a representative pixel from a sampling block
/// based on its luminance.
#[derive(Clone, Copy, Debug)]
enum LuminanceStrategy {
    /// Choose the pixel with the **lowest** luminance.
    Darkest,
    /// Choose the pixel with the **highest** luminance.
    Lightest,
    /// Choose the pixel whose luminance is **farthest** from mid-luminance (≈0.5).
    /// In other words, pick the pixel that is closest to *either* black **or** white.
    MostExtreme,
    /// Choose the pixel whose luminance is **closest** to mid-luminance (≈0.5).
    LeastExtreme,
}

/// Utility that decides whether the `candidate` metric is "better" than the
/// current `best` metric for the given `strategy`.
#[inline(always)]
fn is_better(strategy: LuminanceStrategy, candidate: f32, best: f32) -> bool {
    match strategy {
        LuminanceStrategy::Darkest | LuminanceStrategy::LeastExtreme => candidate < best,
        LuminanceStrategy::Lightest | LuminanceStrategy::MostExtreme => candidate > best,
    }
}

// ------------------------------------------------------------
// Generic luminance-aware downscaling (supports multiple strategies)
// ------------------------------------------------------------

/// Down-scale an image by selecting a representative pixel from every sampling
/// block according to the provided `strategy`.
///
/// The implementation mirrors the original two-pass SIMD-friendly approach
/// (vertical then horizontal) while allowing different selection criteria.
fn downscale_with_luminance(
    img: &DynamicImage,
    out_w: u32,
    out_h: u32,
    strategy: LuminanceStrategy,
) -> DynamicImage {
    let (in_w, in_h) = img.dimensions();

    // Fast path – no scaling required.
    if out_w == in_w && out_h == in_h {
        return img.clone();
    }

    // Raw RGBA bytes.
    let raw = img.to_rgba8().into_raw();

    // --------------------------------------------------------
    // First pass: vertical reduction (m×1)
    // --------------------------------------------------------
    let mut vertical: Vec<[u8; 4]> = vec![[0u8; 4]; (in_w * out_h) as usize];
    let scale_y = in_h as f32 / out_h as f32;

    let mid_lum = 127.5_f32; // Approximate mid-luminance in 0-255 space.

    for x in 0..in_w {
        for y_out in 0..out_h {
            let y_start = (y_out as f32 * scale_y).floor() as u32;
            let y_end = (((y_out as f32 + 1.0) * scale_y).ceil() as u32).min(in_h);

            let mut best_metric = match strategy {
                LuminanceStrategy::Darkest | LuminanceStrategy::LeastExtreme => f32::INFINITY,
                LuminanceStrategy::Lightest | LuminanceStrategy::MostExtreme => f32::NEG_INFINITY,
            };
            let mut chosen = [0u8; 4];

            for y in y_start..y_end {
                let idx = ((y * in_w + x) * 4) as usize;
                let r = raw[idx] as f32;
                let g = raw[idx + 1] as f32;
                let b = raw[idx + 2] as f32;
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                let metric = match strategy {
                    LuminanceStrategy::Darkest | LuminanceStrategy::Lightest => lum,
                    LuminanceStrategy::MostExtreme | LuminanceStrategy::LeastExtreme => (lum - mid_lum).abs(),
                };

                if is_better(strategy, metric, best_metric) {
                    best_metric = metric;
                    chosen = [raw[idx], raw[idx + 1], raw[idx + 2], raw[idx + 3]];
                }
            }

            vertical[(y_out * in_w + x) as usize] = chosen;
        }
    }

    // --------------------------------------------------------
    // Second pass: horizontal reduction (1×m)
    // --------------------------------------------------------
    let mut out_buf: Vec<u8> = vec![0u8; (out_w * out_h * 4) as usize];
    let scale_x = in_w as f32 / out_w as f32;

    for y_out in 0..out_h {
        for x_out in 0..out_w {
            let x_start = (x_out as f32 * scale_x).floor() as u32;
            let x_end = (((x_out as f32 + 1.0) * scale_x).ceil() as u32).min(in_w);

            let mut best_metric = match strategy {
                LuminanceStrategy::Darkest | LuminanceStrategy::LeastExtreme => f32::INFINITY,
                LuminanceStrategy::Lightest | LuminanceStrategy::MostExtreme => f32::NEG_INFINITY,
            };
            let mut chosen = [0u8; 4];

            for x in x_start..x_end {
                let pix = vertical[(y_out * in_w + x) as usize];
                let r = pix[0] as f32;
                let g = pix[1] as f32;
                let b = pix[2] as f32;
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                let metric = match strategy {
                    LuminanceStrategy::Darkest | LuminanceStrategy::Lightest => lum,
                    LuminanceStrategy::MostExtreme | LuminanceStrategy::LeastExtreme => (lum - mid_lum).abs(),
                };

                if is_better(strategy, metric, best_metric) {
                    best_metric = metric;
                    chosen = pix;
                }
            }

            let idx = ((y_out * out_w + x_out) * 4) as usize;
            out_buf[idx..idx + 4].copy_from_slice(&chosen);
        }
    }

    DynamicImage::ImageRgba8(
        RgbaImage::from_raw(out_w, out_h, out_buf)
            .expect("Failed to build output image in downscale_with_luminance"),
    )
}

// Darkest-pixel downscaling (min-luminance)
// ------------------------------------------------------------
// NOTE: The old `downscale_darkest` function has been replaced by the
// generic `downscale_with_luminance`. For backwards compatibility (internal
// to this crate) we could keep a thin wrapper, but we opt to migrate the call
// sites instead.
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

    // let downscaled = image::imageops::resize(&quantized_img, down_w, down_h, FilterType::Nearest);
    // let downscaled = downscale_with_luminance(&quantized_img, down_w, down_h, LuminanceStrategy::Darkest);
    let downscaled = downscale_with_histogram(&quantized_img, down_w, down_h, n_colors, LUMINANCE_WEIGHT);

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

// ------------------------------------------------------------
// Histogram-based downscaling (mode color with darkest-pixel tie-break)
// ------------------------------------------------------------
fn downscale_with_histogram(
    img: &DynamicImage,
    out_w: u32,
    out_h: u32,
    n_colors: usize,
    weight_c: f32,
) -> DynamicImage {
    let (in_w, in_h) = img.dimensions();

    // Fast path – no scaling required.
    if out_w == in_w && out_h == in_h {
        return img.clone();
    }

    // Raw RGBA bytes.
    let raw = img.to_rgba8().into_raw();

    // --------------------------------------------------------
    // First pass: vertical reduction (m×1)
    // --------------------------------------------------------
    let mut vertical: Vec<[u8; 4]> = vec![[0u8; 4]; (in_w * out_h) as usize];
    let scale_y = in_h as f32 / out_h as f32;

    // A small, stack-allocated histogram ‑ we never use more than `n_colors ≤ 64`.
    const MAX_COLORS: usize = 64;

    for x in 0..in_w {
        for y_out in 0..out_h {
            let y_start = (y_out as f32 * scale_y).floor() as u32;
            let y_end = (((y_out as f32 + 1.0) * scale_y).ceil() as u32).min(in_h);

            let mut colors: [[u8; 4]; MAX_COLORS] = [[0; 4]; MAX_COLORS];
            let mut counts: [u32; MAX_COLORS] = [0; MAX_COLORS];
            let mut used = 0usize;

            for y in y_start..y_end {
                let idx = ((y * in_w + x) * 4) as usize;
                let pix = [raw[idx], raw[idx + 1], raw[idx + 2], raw[idx + 3]];

                // Linear search – fine for ≤64 entries.
                let mut found = None;
                for i in 0..used {
                    if colors[i] == pix {
                        found = Some(i);
                        break;
                    }
                }

                match found {
                    Some(i) => counts[i] += 1,
                    None => {
                        if used < n_colors && used < MAX_COLORS {
                            colors[used] = pix;
                            counts[used] = 1;
                            used += 1;
                        }
                    }
                }
            }

            // Choose the colour with the highest weighted frequency; if equal, prefer the darkest.
            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_lum = f32::INFINITY; // smaller = darker

            for i in 0..used {
                let lum = 0.2126 * colors[i][0] as f32
                    + 0.7152 * colors[i][1] as f32
                    + 0.0722 * colors[i][2] as f32;
                let norm_lum = lum / 255.0; // 0-1
                let score = (counts[i] as f32) * (1.0 + weight_c * (1.0 - norm_lum));

                if score > best_score {
                    best_score = score;
                    best_idx = i;
                    best_lum = lum;
                } else if (score - best_score).abs() < 1e-6 {
                    // tie – pick the darker colour
                    if lum < best_lum {
                        best_idx = i;
                        best_lum = lum;
                    }
                }
            }

            vertical[(y_out * in_w + x) as usize] = colors[best_idx];
        }
    }

    // --------------------------------------------------------
    // Second pass: horizontal reduction (1×m)
    // --------------------------------------------------------
    let mut out_buf: Vec<u8> = vec![0u8; (out_w * out_h * 4) as usize];
    let scale_x = in_w as f32 / out_w as f32;

    for y_out in 0..out_h {
        for x_out in 0..out_w {
            let x_start = (x_out as f32 * scale_x).floor() as u32;
            let x_end = (((x_out as f32 + 1.0) * scale_x).ceil() as u32).min(in_w);

            let mut colors: [[u8; 4]; MAX_COLORS] = [[0; 4]; MAX_COLORS];
            let mut counts: [u32; MAX_COLORS] = [0; MAX_COLORS];
            let mut used = 0usize;

            for x in x_start..x_end {
                let pix = vertical[(y_out * in_w + x) as usize];

                let mut found = None;
                for i in 0..used {
                    if colors[i] == pix {
                        found = Some(i);
                        break;
                    }
                }

                match found {
                    Some(i) => counts[i] += 1,
                    None => {
                        if used < n_colors && used < MAX_COLORS {
                            colors[used] = pix;
                            counts[used] = 1;
                            used += 1;
                        }
                    }
                }
            }

            let mut best_idx = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_lum = f32::INFINITY;

            for i in 0..used {
                let lum = 0.2126 * colors[i][0] as f32
                    + 0.7152 * colors[i][1] as f32
                    + 0.0722 * colors[i][2] as f32;
                let norm_lum = lum / 255.0;
                let score = (counts[i] as f32) * (1.0 + weight_c * (1.0 - norm_lum));

                if score > best_score {
                    best_score = score;
                    best_idx = i;
                    best_lum = lum;
                } else if (score - best_score).abs() < 1e-6 {
                    if lum < best_lum {
                        best_idx = i;
                        best_lum = lum;
                    }
                }
            }

            let idx = ((y_out * out_w + x_out) * 4) as usize;
            out_buf[idx..idx + 4].copy_from_slice(&colors[best_idx]);
        }
    }

    DynamicImage::ImageRgba8(
        RgbaImage::from_raw(out_w, out_h, out_buf)
            .expect("Failed to build output image in downscale_with_histogram"),
    )
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
    // let downscaled = image::imageops::resize(&quantized_img, down_w, down_h, FilterType::Nearest);
    // let downscaled = downscale_with_luminance(&quantized_img, down_w, down_h, LuminanceStrategy::Darkest);
    let downscaled = downscale_with_histogram(&quantized_img, down_w, down_h, n_colors, LUMINANCE_WEIGHT);

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

#[cfg(not(target_arch = "wasm32"))]
pub fn extract_palette_bytes(
    input: &[u8],
    n_colors: usize,
    downscale: Option<u32>,
) -> Result<Vec<String>> {
    // Decode the image
    let img = image::load_from_memory(input)?;

    // Optionally downscale for faster k-means
    let working_img: DynamicImage = if let Some(scale) = downscale {
        let (orig_w, orig_h) = img.dimensions();
        let max_side = orig_w.max(orig_h) as f32;
        let ratio = scale as f32 / max_side;
        let w = ((orig_w as f32) * ratio).round().max(1.0) as u32;
        let h = ((orig_h as f32) * ratio).round().max(1.0) as u32;
        image::DynamicImage::ImageRgba8(image::imageops::resize(&img, w, h, FilterType::Nearest))
    } else {
        img
    };

    let raw = working_img.to_rgba8().into_raw();

    // Collect Lab pixels from opaque areas
    let mut lab_pixels: Vec<Lab> = Vec::new();
    for chunk in raw.chunks(4) {
        if chunk[3] == 0 { continue; }
        let srgb = Srgb::<u8>::new(chunk[0], chunk[1], chunk[2]);
        lab_pixels.push(srgb.into_linear().into_color());
    }

    // Run k-means
    let kmeans = get_kmeans(n_colors, 20, 1e-4, false, &lab_pixels, 0);

    // Convert centroids to HEX strings
    let palette_hex: Vec<String> = kmeans.centroids.iter().map(|&lab| {
        let rgb_f32: Srgb<f32> = Srgb::from_linear(lab.into_color());
        let c: Srgb<u8> = rgb_f32.into_format::<u8>();
        format!("{:02X}{:02X}{:02X}", c.red, c.green, c.blue)
    }).collect();

    Ok(palette_hex)
}
