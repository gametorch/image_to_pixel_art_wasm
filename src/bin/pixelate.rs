use clap::Parser;
use std::fs;
use std::path::{PathBuf};
use image_to_pixel_art_wasm::{pixelate_bytes, extract_palette_bytes};
use anyhow::{Context, Result, bail};
use image::{self, GenericImageView};

/// Pixel-artify images using the Rust WASM library (native wrapper).
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// One or more input image paths
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Number of colors for k-means when no custom palette is provided
    #[arg(short = 'k', long, default_value_t = 16)]
    n_colors: usize,

    /// Scaling factor for resulting "pixel" size. (0.0, ∞). Applied per-image.
    #[arg(long, default_value_t = 1.0)]
    relative_scale: f32,

    /// Optional upscale size. If omitted, original dimensions are used.
    #[arg(short, long)]
    output_size: Option<u32>,

    /// Comma-separated list of hex colors to use as palette (skip k-means)
    #[arg(short = 'c', long)]
    palette: Option<String>,

    /// Extract k-means palette from this reference image, then apply to inputs
    #[arg(long, value_name = "FILE")]
    fix_palette: Option<PathBuf>,

    /// Skip downscale before palette extraction
    #[arg(long, default_value_t = false)]
    no_downscale: bool,

    /// Output directory
    #[arg(short = 'd', long)]
    out_dir: Option<PathBuf>,

    /// Output filename prefix (ignored when --out-dir supplied)
    #[arg(short = 'p', long, default_value = "pixelated_")]
    prefix: String,

    /// Machine-readable output (JSON)
    #[arg(long)]
    porcelain: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate relative_scale range
    if args.relative_scale <= 0.0 {
        bail!("--relative-scale must be within (0.0, ∞)");
    }

    if args.palette.is_some() && args.fix_palette.is_some() {
        bail!("--palette and --fix-palette cannot be used together");
    }

    let palette_vec: Option<Vec<String>> = if let Some(ref ref_img) = args.fix_palette {
        let bytes = fs::read(ref_img).with_context(|| format!("Failed to read reference image {}", ref_img.display()))?;

        // Determine downscale size for reference palette extraction
        let downscale_opt = if args.no_downscale {
            None
        } else {
            let rel = args.relative_scale;
            let img = image::load_from_memory(&bytes)
                .with_context(|| format!("Decoding reference image {} failed", ref_img.display()))?;
            let (w, h) = img.dimensions();
            let longest = w.max(h) as f32;
            Some(((longest * (2.0 / (longest * rel * 0.5).sqrt())).round().max(1.0)) as u32)
        };

        Some(extract_palette_bytes(&bytes, args.n_colors, downscale_opt).context("Extracting palette with k-means failed")?)
    } else if let Some(ref s) = args.palette {
        Some(s.split(',').map(|x| x.trim().trim_start_matches('#').to_uppercase()).collect())
    } else {
        None
    };

    for input in &args.inputs {
        let bytes = fs::read(input)?;

        // Determine scale in pixels based on --relative-scale
        let rel = args.relative_scale;
        let img = image::load_from_memory(&bytes)
            .with_context(|| format!("Decoding image {} for relative-scale computation failed", input.display()))?;
        let (w, h) = img.dimensions();
        let longest = w.max(h) as f32;
        let scale_px: u32 = ((longest * (2.0 / (longest * rel * 0.5).sqrt())).round().max(1.0)) as u32;

        let (png, pal) = pixelate_bytes(
            &bytes,
            args.n_colors,
            scale_px,
            args.output_size,
            palette_vec.as_deref(),
        ).context("pixelate processing failed")?;

        let out_path = if let Some(dir) = &args.out_dir {
            let stem = input.file_stem().unwrap_or_default().to_string_lossy();
            dir.join(format!("{stem}.png"))
        } else {
            let stem = input.file_name().unwrap().to_string_lossy();
            PathBuf::from(format!("{}{}", args.prefix, stem))
        };

        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out_path, png)?;

        if args.porcelain {
            use serde_json::json;

            let abs_path = out_path.canonicalize().unwrap_or(out_path.clone());

            let args_json = json!({
                "inputs": args.inputs.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
                "n_colors": args.n_colors,
                "scale": scale_px,
                "relative_scale": args.relative_scale,
                "output_size": args.output_size,
                "palette": palette_vec,
                "fix_palette": args.fix_palette.as_ref().map(|p| p.to_string_lossy()),
                "no_downscale": args.no_downscale,
                "out_dir": args.out_dir.as_ref().map(|p| p.to_string_lossy()),
                "prefix": args.prefix,
            });

            let output_json = json!({
                "is_directory": args.out_dir.is_some(),
                "path": abs_path.to_string_lossy(),
                "palette": pal,
            });

            let report = json!({
                "input_args": args_json,
                "output": output_json,
            });

            println!("{}", report.to_string());
        } else {
            println!("Saved → {}", out_path.display());
        }
    }

    Ok(())
} 