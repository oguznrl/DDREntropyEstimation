#!/usr/bin/env python3
"""
Statistical Feature Extraction for DDR-RAM Entropy Region Selection
Paper Reference: "Multi-Metric Statistical Filtering for Entropy-Driven DDR-RAM Region Selection"
Section III (Preliminaries) & Section IV-B (Single-Stage Filtering, Eq. 10)

This script scans raw grayscale PNG window images, computes 7 complementary statistical
randomness metrics for each window, and exports the results to a CSV file (e.g.,
entropy_features_by_window.csv) directly consumed by ddr_entropy_gen_method.py.
"""

import os
import re
import sys
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool, cpu_count
from numpy.lib.stride_tricks import sliding_window_view

def calc_binary_bias(bits: np.ndarray) -> float:
    """Equation (7): Binary Bias B = |n_1 / n_b - 0.5|"""
    nb = len(bits)
    if nb == 0:
        return 0.5
    n1 = np.count_nonzero(bits)
    val = abs((n1 / nb) - 0.5)
    return float(min(0.5, max(0.0, val)))

def calc_binary_shannon(bits: np.ndarray) -> float:
    """Equation (1): Binary Shannon Entropy H(X) = - p0*log2(p0) - p1*log2(p1)"""
    nb = len(bits)
    if nb == 0:
        return 0.0
    n1 = np.count_nonzero(bits)
    p1 = n1 / nb
    p0 = 1.0 - p1
    if p0 <= 0.0 or p1 <= 0.0:
        return 0.0
    val = -(p0 * math.log2(p0) + p1 * math.log2(p1))
    return float(min(1.0, max(0.0, val)))

def calc_blockwise_entropy(bits: np.ndarray, num_blocks: int = 16) -> float:
    """Equation (2): Block-wise Entropy H_G(X) = (1/N) * sum(H(X^i)) across N blocks"""
    blocks = np.array_split(bits, num_blocks)
    val = np.mean([calc_binary_shannon(b) for b in blocks])
    return float(min(1.0, max(0.0, val)))

def calc_grouped_std(pixels: np.ndarray, grid: int = 4) -> float:
    """Equation (6): Grouped Standard Deviation sigma_G(X) across grid x grid blocks"""
    h, w = pixels.shape
    bh, bw = h // grid, w // grid
    if bh == 0 or bw == 0:
        return float(np.std(pixels))
    stds = [
        np.std(pixels[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw])
        for i in range(grid)
        for j in range(grid)
    ]
    return float(np.mean(stds))

def calc_spectral_entropy(pixels: np.ndarray) -> float:
    """Equation (3)-(4): Spectral Entropy SE = - sum(p * log2(p)) from PSD"""
    f = np.fft.fft2(pixels)
    fshift = np.fft.fftshift(f)
    psd = np.abs(fshift) ** 2
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    p = (psd / psd_sum).flatten()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def calc_permutation_entropy(pixels: np.ndarray, d: int = 3, tau: int = 1) -> float:
    """
    Equation (8)-(9): Normalized Permutation Entropy PE = - sum(p * log2(p)) / log2(D!)
    Vectorized sliding-window pattern encoding.
    """
    x = pixels.ravel()
    n = len(x)
    min_len = (d - 1) * tau + 1
    if n < min_len:
        return 1.0
    
    w = sliding_window_view(x, d)
    if tau > 1:
        w = w[::tau]
        
    orders = np.argsort(w, axis=1)
    weights = np.array([d ** i for i in range(d)][::-1])
    codes = np.dot(orders, weights)
    
    counts = np.bincount(codes)
    p = counts[counts > 0] / len(codes)
    pe = -np.sum(p * np.log2(p))
    val = pe / math.log2(math.factorial(d))
    return float(min(1.0, max(0.0, val)))

def process_single_image(args):
    img_path, filename = args
    match = re.search(r'\d+', filename)
    window_idx = int(match.group()) if match else None
    
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L")
            pixels = np.array(gray, dtype=np.uint8)
            bits = np.unpackbits(pixels)
            
            bias_val = calc_binary_bias(bits)
            shannon_val = calc_binary_shannon(bits)
            block_shannon_val = calc_blockwise_entropy(bits, num_blocks=16)
            img_std = float(np.std(pixels))
            img_block_std = calc_grouped_std(pixels, grid=4)
            spec_ent = calc_spectral_entropy(pixels)
            perm_ent = calc_permutation_entropy(pixels, d=3, tau=1)
            
            return {
                "window_index": window_idx,
                "filename": filename,
                "bias": bias_val,
                "shannon_entropy": shannon_val,
                "blockwise_entropy": block_shannon_val,
                "image_std": img_std,
                "image_blockwise_std": img_block_std,
                "image_spectral_entropy": spec_ent,
                "image_permutation_entropy": perm_ent
            }
    except Exception as e:
        print(f"Error processing {filename}: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Extract 7 statistical randomness metrics from DDR-RAM window images."
    )
    parser.add_argument(
        "-i", "--image-dir",
        required=True,
        help="Directory containing grayscale PNG images."
    )
    parser.add_argument(
        "-o", "--output-csv",
        default="entropy_features_by_window.csv",
        help="Output CSV path (default: entropy_features_by_window.csv)."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: all CPU cores)."
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        print(f"Error: Directory not found: {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    all_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(".png")])
    if not all_files:
        print(f"Error: No PNG files found in {args.image_dir}", file=sys.stderr)
        sys.exit(1)

    print("=========================================================")
    print("DDR-RAM MULTI-METRIC FEATURE EXTRACTION")
    print("=========================================================")
    print(f"Image Directory : {args.image_dir}")
    print(f"Total Images    : {len(all_files)}")
    print(f"Worker Processes: {args.workers}")
    print(f"Output File     : {args.output_csv}")
    print("---------------------------------------------------------")

    tasks = [(os.path.join(args.image_dir, f), f) for f in all_files]

    with Pool(args.workers) as pool:
        records = pool.map(process_single_image, tasks)

    records = [r for r in records if r is not None]
    if not records:
        print("Error: No images were successfully processed.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    if "window_index" in df.columns and df["window_index"].notnull().all():
        df = df.sort_values("window_index")

    df.to_csv(args.output_csv, index=False)
    print("---------------------------------------------------------")
    print(f"Successfully processed {len(df)} images.")
    print(f"Feature table written to: {args.output_csv}")
    print("You can now pass this CSV to ddr_entropy_gen_method.py using the -f flag.")
    print("=========================================================")

if __name__ == "__main__":
    main()
