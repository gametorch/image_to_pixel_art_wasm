use clap::Parser;
use std::fs;
use std::path::{PathBuf};
use image_to_pixel_art_wasm::pixelate_bytes; // we need to implement
use anyhow::Context;
use anyhow::Result;

/// Pixel-artify images using the Rust WASM library (native wrapper).
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// One or more input image paths
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Number of colors for k-means when no custom palette is provided
    #[arg(short = 'k', long, default_value_t = 8)]
    n_colors: usize,

    /// Target down-sample size (longest side)
    #[arg(short, long, default_value_t = 64)]
    scale: u32,

    /// Optional upscale size. If omitted, original dimensions are used.
    #[arg(short, long)]
    output_size: Option<u32>,

    /// Comma-separated list of hex colors to use as palette (skip k-means)
    #[arg(short = 'c', long)]
    palette: Option<String>,

    /// Output directory
    #[arg(short = 'd', long)]
    out_dir: Option<PathBuf>,

    /// Output filename prefix (ignored when --out-dir supplied)
    #[arg(short = 'p', long, default_value = "pixelated_")]
    prefix: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let palette_vec: Option<Vec<String>> = args.palette.as_ref().map(|s| {
        s.split(',').map(|x| x.trim().trim_start_matches('#').to_uppercase()).collect()
    });

    for input in &args.inputs {
        let bytes = fs::read(input)?;
        let (png, _pal) = pixelate_bytes(
            &bytes,
            args.n_colors,
            args.scale,
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
        println!("Saved → {}", out_path.display());
    }

    Ok(())
} 