#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py â€” End-to-End Retinal Vessel Segmentation Pipeline
===================================================================

Orchestrates the complete workflow:
    1. Batch-process all fundus images through the segmentation pipeline
    2. Evaluate results against ground-truth annotations
    3. Generate comprehensive performance visualizations
    4. Export a detailed metrics report (CSV + console summary)

Usage
-----
    # Run on main dataset (80 images):
    python run_pipeline.py --dataset main

    # Run on supplementary dataset (20 images):
    python run_pipeline.py --dataset add

    # Run on both datasets:
    python run_pipeline.py --dataset all

    # Skip re-segmentation (evaluate existing outputs only):
    python run_pipeline.py --dataset main --evaluate-only

Author : Abdo Hussam
Course : CDS6334 â€“ Visual Information Processing
"""

import argparse
import os
import sys
import time
import csv

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for figure generation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

import imageSegment as seg

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Constants
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DATASET_CONFIGS = {
    "main": {
        "input_dir": "dataset/test",
        "output_dir": "dataset/output",
        "gt_dir": "dataset/groundtruth",
        "results_dir": "results/main_dataset",
        "preset": "dataset",
        "label": "Main Dataset (80 images)",
    },
    "add": {
        "input_dir": "add_dataset/test",
        "output_dir": "add_dataset/output",
        "gt_dir": "add_dataset/groundtruth",
        "results_dir": "results/add_dataset",
        "preset": "add_dataset",
        "label": "Supplementary Dataset (20 images)",
    },
}

EPS = 1e-8

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Utility helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def list_images(directory: str) -> list:
    """Return sorted list of image filenames in *directory*."""
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
    return sorted(
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in valid_ext
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 1 â€” Batch Segmentation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def batch_segment(input_dir: str, output_dir: str) -> float:
    """Run segmentation on every image in *input_dir*, save masks to *output_dir*.

    Returns the total wall-clock time in seconds.
    """
    ensure_dir(output_dir)
    files = list_images(input_dir)
    print(f"\n{'='*60}")
    print(f"  SEGMENTATION  |  {len(files)} images  |  {input_dir}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    for i, fname in enumerate(files, 1):
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            print(f"  [SKIP] Could not read {fname}")
            continue

        mask = seg.segmentImage(img).astype(np.float32)
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{stem}.png")
        plt.imsave(out_path, 255 * mask, cmap=cm.gray)
        print(f"  [{i:3d}/{len(files)}] {fname}  â†’  {stem}.png")

    elapsed = time.perf_counter() - t0
    print(f"\n  âœ“ Segmentation complete in {elapsed:.1f}s "
          f"({elapsed/len(files):.2f}s per image)")
    return elapsed


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 2 â€” Evaluation Metrics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute segmentation quality metrics for a single image.

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary masks âˆˆ {0, 1} of the same shape.

    Returns
    -------
    dict with keys: precision, recall, f1, iou, error, dice, specificity, accuracy
    """
    pred = pred.astype(np.float64).flatten()
    gt = gt.astype(np.float64).flatten()

    tp = np.sum(gt * pred)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    tn = len(gt) - tp - fp - fn

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)
    error = 1.0 - f1
    dice = f1  # Dice coefficient == F1 for binary masks
    specificity = tn / (tn + fp + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "iou": round(iou, 6),
        "error": round(error, 6),
        "dice": round(dice, 6),
        "specificity": round(specificity, 6),
        "accuracy": round(accuracy, 6),
    }


def evaluate_dataset(output_dir: str, gt_dir: str) -> list:
    """Evaluate all output masks against ground-truth.

    Returns a list of dicts, one per image, with an ``'image'`` key added.
    """
    files = list_images(output_dir)
    results = []

    print(f"\n{'='*60}")
    print(f"  EVALUATION  |  {len(files)} images")
    print(f"{'='*60}")
    print(f"  {'Image':<12} {'Prec':>8} {'Recall':>8} {'F1':>8} "
          f"{'IoU':>8} {'Acc':>8} {'Error':>8}")
    print(f"  {'-'*66}")

    for fname in files:
        stem = os.path.splitext(fname)[0]

        # Load prediction
        pred_path = os.path.join(output_dir, fname)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        pred = np.round(pred.astype(np.float32) / max(pred.max(), 1))

        # Load ground-truth
        gt_path = os.path.join(gt_dir, f"{stem}.png")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        gt = np.round(gt.astype(np.float32) / max(gt.max(), 1))

        metrics = compute_metrics(pred, gt)
        metrics["image"] = fname
        results.append(metrics)

        print(f"  {fname:<12} {metrics['precision']:8.4f} {metrics['recall']:8.4f} "
              f"{metrics['f1']:8.4f} {metrics['iou']:8.4f} "
              f"{metrics['accuracy']:8.4f} {metrics['error']:8.4f}")

    if results:
        avg = {k: np.mean([r[k] for r in results])
               for k in results[0] if k != "image"}
        print(f"  {'-'*66}")
        print(f"  {'AVERAGE':<12} {avg['precision']:8.4f} {avg['recall']:8.4f} "
              f"{avg['f1']:8.4f} {avg['iou']:8.4f} "
              f"{avg['accuracy']:8.4f} {avg['error']:8.4f}")

    return results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 3 â€” Visualization Generation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Professional color palette
COLORS = {
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B",
    "danger": "#EF4444",
    "bg": "#F8FAFC",
    "text": "#1E293B",
    "grid": "#E2E8F0",
}


def _style_axis(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent professional styling to a matplotlib axis."""
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["text"], pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=COLORS["text"])
    ax.set_ylabel(ylabel, fontsize=10, color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def generate_per_image_bar_chart(results: list, save_path: str) -> None:
    """Bar chart showing F1 and IoU for every image."""
    names = [r["image"].replace(".png", "") for r in results]
    f1_vals = [r["f1"] for r in results]
    iou_vals = [r["iou"] for r in results]

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.4), 5))
    fig.patch.set_facecolor("white")
    x = np.arange(len(names))
    w = 0.35

    ax.bar(x - w / 2, f1_vals, w, label="F1-Score", color=COLORS["primary"], alpha=0.85)
    ax.bar(x + w / 2, iou_vals, w, label="IoU", color=COLORS["secondary"], alpha=0.85)

    # Mean reference lines
    ax.axhline(np.mean(f1_vals), ls="--", color=COLORS["primary"], alpha=0.5,
               label=f"Mean F1 = {np.mean(f1_vals):.3f}")
    ax.axhline(np.mean(iou_vals), ls="--", color=COLORS["secondary"], alpha=0.5,
               label=f"Mean IoU = {np.mean(iou_vals):.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    _style_axis(ax, "Per-Image Segmentation Performance",
                "Image ID", "Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] Per-image metrics  â†’  {save_path}")


def generate_metrics_summary(results: list, save_path: str,
                             dataset_label: str) -> None:
    """Radar + box-plot summary of all metrics."""
    metrics_keys = ["precision", "recall", "f1", "iou", "accuracy", "specificity"]
    labels = ["Precision", "Recall", "F1-Score", "IoU", "Accuracy", "Specificity"]

    means = [np.mean([r[k] for r in results]) for k in metrics_keys]
    stds = [np.std([r[k] for r in results]) for k in metrics_keys]

    fig = plt.figure(figsize=(14, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Segmentation Performance Summary â€” {dataset_label}",
                 fontsize=14, fontweight="bold", color=COLORS["text"], y=1.02)

    gs = GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35)

    # â”€â”€ Left: Radar chart â”€â”€
    ax_radar = fig.add_subplot(gs[0], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = means + [means[0]]
    angles += [angles[0]]

    ax_radar.fill(angles, values, alpha=0.15, color=COLORS["primary"])
    ax_radar.plot(angles, values, "o-", color=COLORS["primary"], linewidth=2,
                  markersize=6)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels, fontsize=9, color=COLORS["text"])
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Mean Metrics (Radar)", fontsize=11, fontweight="bold",
                       color=COLORS["text"], pad=20)

    # â”€â”€ Right: Box plot â”€â”€
    ax_box = fig.add_subplot(gs[1])
    data = [[r[k] for r in results] for k in metrics_keys]
    bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color=COLORS["danger"], linewidth=2))

    box_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  "#8B5CF6", "#06B6D4", "#EC4899"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    _style_axis(ax_box, "Metric Distribution (Box Plot)", "", "Score")
    ax_box.set_ylim(0, 1.05)

    # Add mean Â± std annotations
    for i, (m, s) in enumerate(zip(means, stds)):
        ax_box.text(i + 1, 1.02, f"{m:.3f}Â±{s:.3f}", ha="center",
                    fontsize=7, color=COLORS["text"])

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] Metrics summary    â†’  {save_path}")


def generate_comparison_grid(input_dir: str, output_dir: str, gt_dir: str,
                             results: list, save_path: str,
                             dataset_label: str) -> None:
    """Side-by-side comparison of best, median, and worst performing images."""
    if len(results) < 3:
        return

    sorted_by_f1 = sorted(results, key=lambda r: r["f1"])
    picks = {
        "Worst (F1)": sorted_by_f1[0],
        "Median (F1)": sorted_by_f1[len(sorted_by_f1) // 2],
        "Best (F1)": sorted_by_f1[-1],
    }

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Segmentation Comparison â€” {dataset_label}",
                 fontsize=14, fontweight="bold", color=COLORS["text"])

    for row, (label, r) in enumerate(picks.items()):
        stem = os.path.splitext(r["image"])[0]

        # Original
        orig = cv2.imread(os.path.join(input_dir, r["image"]))
        if orig is None:
            orig = cv2.imread(os.path.join(input_dir, f"{stem}.png"))
        if orig is not None:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(orig if orig is not None else np.zeros((10, 10)))
        axes[row, 0].set_title("Original" if row == 0 else "", fontsize=10)
        axes[row, 0].set_ylabel(f"{label}\n(F1={r['f1']:.3f}, IoU={r['iou']:.3f})",
                                fontsize=9, fontweight="bold")

        # Prediction
        pred = cv2.imread(os.path.join(output_dir, f"{stem}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if pred is not None:
            axes[row, 1].imshow(pred, cmap="gray")
        axes[row, 1].set_title("Prediction" if row == 0 else "", fontsize=10)

        # Ground truth
        gt = cv2.imread(os.path.join(gt_dir, f"{stem}.png"),
                        cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            axes[row, 2].imshow(gt, cmap="gray")
        axes[row, 2].set_title("Ground Truth" if row == 0 else "", fontsize=10)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] Comparison grid    â†’  {save_path}")


def generate_error_heatmap(results: list, save_path: str,
                           dataset_label: str) -> None:
    """Heatmap of all metrics across images for quick visual inspection."""
    metrics_keys = ["precision", "recall", "f1", "iou", "accuracy", "error"]
    names = [r["image"].replace(".png", "") for r in results]
    data = np.array([[r[k] for k in metrics_keys] for r in results])

    fig, ax = plt.subplots(figsize=(8, max(5, len(names) * 0.25)))
    fig.patch.set_facecolor("white")

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics_keys)))
    ax.set_xticklabels(["Precision", "Recall", "F1", "IoU", "Accuracy", "Error"],
                       fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title(f"Per-Image Metrics Heatmap â€” {dataset_label}",
                 fontsize=12, fontweight="bold", color=COLORS["text"], pad=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] Error heatmap      â†’  {save_path}")


def generate_pixel_distribution(input_dir: str, save_path: str,
                                dataset_label: str) -> None:
    """Histogram of green-channel pixel intensities across all images."""
    files = list_images(input_dir)
    all_green = []

    for fname in files[:20]:  # Sample first 20 for speed
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is not None:
            all_green.append(img[:, :, 1].flatten())

    if not all_green:
        return

    combined = np.concatenate(all_green)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.hist(combined, bins=128, color=COLORS["primary"], alpha=0.7,
            edgecolor="white", linewidth=0.3)
    _style_axis(ax, f"Green Channel Intensity Distribution â€” {dataset_label}",
                "Pixel Intensity", "Frequency")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] Pixel distribution â†’  {save_path}")


def export_csv(results: list, save_path: str) -> None:
    """Export detailed metrics to CSV."""
    if not results:
        return
    keys = ["image", "precision", "recall", "f1", "iou", "dice",
            "accuracy", "specificity", "error"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

        # Write averages
        avg = {k: round(np.mean([r[k] for r in results]), 6)
               for k in keys if k != "image"}
        avg["image"] = "AVERAGE"
        writer.writerow(avg)

    print(f"  [CSV] Metrics exported   â†’  {save_path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Stage 4 â€” LinkedIn-Ready Summary Figure
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_showcase_figure(all_results: dict, save_path: str) -> None:
    """Create a single compelling summary figure for LinkedIn/portfolio.

    Combines key results from all evaluated datasets into one polished visual.
    """
    n_datasets = len(all_results)
    if n_datasets == 0:
        return

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")

    # Title bar
    fig.text(0.5, 0.97, "Retinal Vessel Segmentation â€” Performance Overview",
             ha="center", fontsize=16, fontweight="bold", color=COLORS["text"])
    fig.text(0.5, 0.935,
             "Classical Image Processing Pipeline  |  CLAHE + Adaptive Thresholding + Morphological Refinement",
             ha="center", fontsize=10, color="#64748B")

    gs = GridSpec(1, 2, wspace=0.3, left=0.08, right=0.95, top=0.88, bottom=0.12)

    # â”€â”€ Left panel: grouped bar comparison â”€â”€
    ax1 = fig.add_subplot(gs[0])
    metrics_display = ["precision", "recall", "f1", "iou"]
    labels_display = ["Precision", "Recall", "F1-Score", "IoU"]
    x = np.arange(len(labels_display))
    width = 0.3
    palette = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

    for idx, (ds_name, results) in enumerate(all_results.items()):
        means = [np.mean([r[k] for r in results]) for k in metrics_display]
        offset = (idx - (n_datasets - 1) / 2) * width
        bars = ax1.bar(x + offset, means, width, label=ds_name,
                       color=palette[idx % len(palette)], alpha=0.85)
        for bar, val in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", fontsize=8, color=COLORS["text"])

    _style_axis(ax1, "Average Metrics by Dataset", "", "Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_display, fontsize=10)
    ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=9)

    # â”€â”€ Right panel: technique highlights â”€â”€
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    techniques = [
        ("Green Channel Extraction", "Maximizes vesselâ€“background contrast"),
        ("CLAHE Enhancement", "Normalizes illumination across the fundus"),
        ("Adaptive Thresholding", "Local mean-based vessel binarization"),
        ("Morphological Ops", "Close/open to refine vessel boundaries"),
        ("ROI Masking", "Restricts analysis to retinal disc region"),
        ("Gaussian Smoothing", "Softens jagged boundary artifacts"),
    ]

    ax2.text(0.5, 0.95, "Pipeline Techniques", ha="center", fontsize=12,
             fontweight="bold", color=COLORS["text"], transform=ax2.transAxes)

    icons = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]"]
    for i, (tech, desc) in enumerate(techniques):
        y = 0.82 - i * 0.14
        ax2.text(0.05, y, icons[i], fontsize=14, transform=ax2.transAxes,
                 va="center")
        ax2.text(0.12, y + 0.02, tech, fontsize=10, fontweight="bold",
                 color=COLORS["text"], transform=ax2.transAxes, va="center")
        ax2.text(0.12, y - 0.04, desc, fontsize=8, color="#64748B",
                 transform=ax2.transAxes, va="center")

    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  â˜… Showcase figure        â†’  {save_path}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main Pipeline Orchestrator
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_dataset_pipeline(name: str, cfg: dict, evaluate_only: bool = False) -> list:
    """Run the full pipeline for one dataset configuration."""
    print(f"\n{'#'*60}")
    print(f"  {cfg['label'].upper()}")
    print(f"{'#'*60}")

    # Set the appropriate segmentation config
    seg.set_config(cfg["preset"])

    # Ensure results directory exists
    fig_dir = os.path.join(cfg["results_dir"], "figures")
    ensure_dir(fig_dir)

    # Stage 1: Segmentation
    if not evaluate_only:
        batch_segment(cfg["input_dir"], cfg["output_dir"])
    else:
        print("\n  [SKIP] Segmentation â€” using existing outputs")

    # Stage 2: Evaluation
    results = evaluate_dataset(cfg["output_dir"], cfg["gt_dir"])
    if not results:
        print("  [WARN] No results to evaluate.")
        return []

    # Stage 3: Visualization
    print(f"\n  GENERATING FIGURES  â†’  {fig_dir}")
    generate_per_image_bar_chart(
        results, os.path.join(fig_dir, "per_image_metrics.png"))
    generate_metrics_summary(
        results, os.path.join(fig_dir, "metrics_summary.png"), cfg["label"])
    generate_comparison_grid(
        cfg["input_dir"], cfg["output_dir"], cfg["gt_dir"],
        results, os.path.join(fig_dir, "comparison_grid.png"), cfg["label"])
    generate_error_heatmap(
        results, os.path.join(fig_dir, "metrics_heatmap.png"), cfg["label"])
    generate_pixel_distribution(
        cfg["input_dir"], os.path.join(fig_dir, "pixel_distribution.png"),
        cfg["label"])

    # Export CSV
    export_csv(results, os.path.join(cfg["results_dir"], "detailed_metrics.csv"))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retinal Vessel Segmentation â€” Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["main", "add", "all"], default="all",
        help="Which dataset to process (default: all)")
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Skip segmentation; evaluate existing outputs only")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RETINAL VESSEL SEGMENTATION PIPELINE")
    print("  CDS6334 â€” Visual Information Processing")
    print("=" * 60)

    datasets_to_run = (
        ["main", "add"] if args.dataset == "all"
        else [args.dataset]
    )

    all_results = {}
    for ds_name in datasets_to_run:
        cfg = DATASET_CONFIGS[ds_name]
        if not os.path.isdir(cfg["input_dir"]):
            print(f"\n  [SKIP] {cfg['label']} â€” input directory not found")
            continue
        results = run_dataset_pipeline(ds_name, cfg, args.evaluate_only)
        if results:
            all_results[cfg["label"]] = results

    # Generate combined showcase figure
    if all_results:
        ensure_dir("results")
        generate_showcase_figure(all_results, "results/showcase_summary.png")

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

