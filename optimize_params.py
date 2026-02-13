# -*- coding: utf-8 -*-
"""
Hyperparameter Optimiser for Retinal Vessel Segmentation
=========================================================

Performs a grid search over the key segmentation hyperparameters to find
the combination that maximises F1-score (harmonic mean of precision and
recall) on a configurable subset of images.

Usage::

    python optimize_params.py                 # optimise main dataset
    python optimize_params.py --dataset add   # optimise supplementary dataset
    python optimize_params.py --sample 20     # use 20 images instead of 10
    python optimize_params.py --metric iou    # optimise for IoU instead of F1

The script prints a ranked table of all tested configurations and writes
the optimal CONFIG dict to the console for direct copy-paste into
imageSegment.py.

Author : Abdo Hussam
Course : CDS6334 -- Visual Information Processing
Date   : 2025
"""

import os
import sys
import time
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

# ── Make sure imageSegment is importable ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imageSegment as seg


# ── Evaluation helpers ──────────────────────────────────────────────────────

def _compute_metrics(pred: np.ndarray, gt: np.ndarray):
    """Compute precision, recall, F1, IoU, accuracy from binary masks."""
    pred_f = pred.astype(np.float64).flatten()
    gt_f = gt.astype(np.float64).flatten()

    tp = np.sum(pred_f * gt_f)
    fp = np.sum(pred_f * (1.0 - gt_f))
    fn = np.sum((1.0 - pred_f) * gt_f)
    tn = np.sum((1.0 - pred_f) * (1.0 - gt_f))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


def _load_gt(gt_path: str) -> np.ndarray:
    """Load and binarise a ground-truth image."""
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"Cannot read: {gt_path}")
    return (gt > 127).astype(np.float64)


# ── Grid definition ────────────────────────────────────────────────────────

# Only the most impactful parameters are searched; others are fixed.
PARAM_GRID = {
    "bg_kernel_size":    [21, 25, 29],
    "clahe_clip_limit":  [1.5, 2.0, 2.5, 3.0],
    "threshold_offset":  [6, 7, 8, 9, 10, 11, 12],
    "min_vessel_area":   [30, 50, 80, 100, 120, 150],
}

# Fixed parameters (not searched)
FIXED_PARAMS = {
    "clahe_grid_size":  (8, 8),
    "median_ksize":     3,
    "mean_filter_size": 18,
    "morph_kernel_size": 3,
    "mask_kernel_size":  7,
    "gaussian_ksize":   (3, 3),
    "gaussian_sigma":   0.5,
    "binary_threshold": 128,
}


def _build_configs():
    """Generate all combinations from the parameter grid."""
    keys = list(PARAM_GRID.keys())
    values = [PARAM_GRID[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(FIXED_PARAMS)
        for k, v in zip(keys, combo):
            cfg[k] = v
        configs.append(cfg)
    return configs


# ── Main optimiser loop ────────────────────────────────────────────────────

def optimise(dataset: str, sample_size: int, metric: str):
    """Run grid search and print results."""

    # Resolve paths
    if dataset == "add":
        input_dir = "add_dataset/test"
        gt_dir = "add_dataset/groundtruth"
        ds_label = "ADD DATASET"
    else:
        input_dir = "dataset/test"
        gt_dir = "dataset/groundtruth"
        ds_label = "MAIN DATASET"

    if not os.path.isdir(input_dir):
        print(f"  ERROR: directory not found: {input_dir}")
        sys.exit(1)

    # Collect image list
    all_files = sorted([f for f in os.listdir(input_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg',
                                               '.bmp', '.tif', '.tiff'))])
    if not all_files:
        print(f"  ERROR: no images in {input_dir}")
        sys.exit(1)

    # Subsample for speed
    step = max(1, len(all_files) // sample_size)
    sample_files = all_files[::step][:sample_size]

    print(f"\n{'='*64}")
    print(f"  PARAMETER OPTIMISER  |  {ds_label}")
    print(f"  Images: {len(sample_files)} / {len(all_files)}  |  Metric: {metric.upper()}")
    print(f"{'='*64}\n")

    # Pre-load images and ground truths
    images, gts = [], []
    for fname in sample_files:
        img = cv2.imread(os.path.join(input_dir, fname))
        stem = Path(fname).stem
        # Try common GT naming patterns
        gt = None
        for ext in ['.png', '.jpg', '.tif', '.bmp', '.gif']:
            gt_path = os.path.join(gt_dir, stem + ext)
            if os.path.isfile(gt_path):
                gt = _load_gt(gt_path)
                break
        if img is None or gt is None:
            print(f"  SKIP {fname} (missing image or ground truth)")
            continue
        images.append(img)
        gts.append(gt)

    if not images:
        print("  ERROR: no valid image/GT pairs found")
        sys.exit(1)

    print(f"  Loaded {len(images)} image/GT pairs")

    # Generate all configs
    configs = _build_configs()
    total = len(configs)
    print(f"  Testing {total} parameter combinations...\n")

    results = []
    t0 = time.time()

    for idx, cfg in enumerate(configs):
        # Inject config directly
        seg._active_config = cfg

        # Evaluate on all sample images
        metrics_sum = {"precision": 0, "recall": 0, "f1": 0,
                       "iou": 0, "accuracy": 0}
        for img, gt in zip(images, gts):
            pred = seg.segmentImage(img).astype(np.float64)
            m = _compute_metrics(pred, gt)
            for k in metrics_sum:
                metrics_sum[k] += m[k]

        n = len(images)
        avg = {k: v / n for k, v in metrics_sum.items()}
        avg["config"] = {k: cfg[k] for k in PARAM_GRID}  # Store only searched params
        results.append(avg)

        # Progress
        if (idx + 1) % 25 == 0 or idx == 0 or idx == total - 1:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1:4d}/{total}]  best {metric}={max(r[metric] for r in results):.4f}"
                  f"  ({rate:.1f} cfg/s  ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Grid search complete in {elapsed:.1f}s")

    # Sort by target metric
    results.sort(key=lambda r: r[metric], reverse=True)

    # Print top 15
    print(f"\n  {'='*78}")
    print(f"  TOP 15 CONFIGURATIONS (sorted by {metric.upper()})")
    print(f"  {'='*78}")
    print(f"  {'Rank':>4s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}"
          f"  {'IoU':>6s}  {'Acc':>6s}  Config")
    print(f"  {'-'*78}")

    for i, r in enumerate(results[:15]):
        cfg_str = ", ".join(f"{k}={v}" for k, v in r["config"].items())
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:4d}  {r['precision']:6.4f}  {r['recall']:6.4f}"
              f"  {r['f1']:6.4f}  {r['iou']:6.4f}  {r['accuracy']:6.4f}"
              f"  {cfg_str}{marker}")

    # Print the best config for copy-paste
    best = results[0]
    print(f"\n  {'='*78}")
    print(f"  OPTIMAL CONFIGURATION")
    print(f"  {'='*78}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  F1:        {best['f1']:.4f}")
    print(f"  IoU:       {best['iou']:.4f}")
    print(f"  Accuracy:  {best['accuracy']:.4f}")

    # Build full config dict for copy-paste
    full_cfg = dict(FIXED_PARAMS)
    full_cfg.update(best["config"])
    print(f"\n  Copy-paste for imageSegment.py:")
    print(f"  {{")
    for k, v in full_cfg.items():
        print(f'      "{k}": {repr(v)},')
    print(f"  }}")

    return best


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid-search optimiser for vessel segmentation parameters")
    parser.add_argument("--dataset", choices=["main", "add"],
                        default="main",
                        help="Which dataset to optimise on (default: main)")
    parser.add_argument("--sample", type=int, default=10,
                        help="Number of images to sample (default: 10)")
    parser.add_argument("--metric", choices=["f1", "iou", "precision",
                                             "recall", "accuracy"],
                        default="f1",
                        help="Metric to maximise (default: f1)")
    args = parser.parse_args()
    optimise(args.dataset, args.sample, args.metric)
