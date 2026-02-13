# -*- coding: utf-8 -*-
"""
visualization.py — Publication-Quality Figures for Retinal Vessel Segmentation
===============================================================================

Generates professional visualizations for reporting and portfolio presentation.
All figures follow a consistent colour palette and typographic style suitable
for academic papers, technical reports, and LinkedIn showcases.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Design System
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "blue":   "#2563EB",
    "green":  "#10B981",
    "amber":  "#F59E0B",
    "red":    "#EF4444",
    "purple": "#8B5CF6",
    "cyan":   "#06B6D4",
    "pink":   "#EC4899",
    "bg":     "#F8FAFC",
    "text":   "#1E293B",
    "muted":  "#64748B",
    "grid":   "#E2E8F0",
}

def _apply_style(ax, title="", xlabel="", ylabel=""):
    """Consistent styling for all axes."""
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=PALETTE["text"], pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["text"])
    ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["text"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    ax.grid(True, alpha=0.25, color=PALETTE["grid"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Overlay Visualization
# ──────────────────────────────────────────────────────────────────────────────

def overlay_vessels_on_fundus(fundus_bgr: np.ndarray,
                              vessel_mask: np.ndarray,
                              color: tuple = (0, 255, 0),
                              alpha: float = 0.5) -> np.ndarray:
    """Overlay predicted vessels on the original fundus image.

    Parameters
    ----------
    fundus_bgr : np.ndarray
        Original fundus image (BGR, H×W×3).
    vessel_mask : np.ndarray
        Binary vessel mask (H×W), values {0, 1} or {0, 255}.
    color : tuple
        BGR colour for the vessel overlay.
    alpha : float
        Blending factor (0 = no overlay, 1 = full colour).

    Returns
    -------
    np.ndarray
        Blended image in BGR format.
    """
    mask = vessel_mask.copy()
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    overlay = fundus_bgr.copy()
    vessel_pixels = mask > 128
    overlay[vessel_pixels] = (
        (1 - alpha) * fundus_bgr[vessel_pixels]
        + alpha * np.array(color, dtype=np.float64)
    ).astype(np.uint8)
    return overlay


def create_error_overlay(pred: np.ndarray, gt: np.ndarray,
                         fundus_bgr: np.ndarray) -> np.ndarray:
    """Create a colour-coded error visualization.

    - Green  = True positive  (correct vessel detection)
    - Red    = False positive  (detected but not a vessel)
    - Blue   = False negative  (missed vessel)

    Parameters
    ----------
    pred, gt : np.ndarray
        Binary masks (H×W), values ∈ {0, 1}.
    fundus_bgr : np.ndarray
        Original fundus image for background context.

    Returns
    -------
    np.ndarray
        Colour-coded error map in BGR format.
    """
    pred_b = (pred > 0.5).astype(np.uint8)
    gt_b = (gt > 0.5).astype(np.uint8)

    tp = (pred_b & gt_b).astype(bool)
    fp = (pred_b & ~gt_b).astype(bool)
    fn = (~pred_b & gt_b).astype(bool)

    # Start with dimmed fundus
    overlay = (fundus_bgr.astype(np.float64) * 0.3).astype(np.uint8)
    overlay[tp] = [0, 200, 0]     # Green = TP
    overlay[fp] = [0, 0, 220]     # Red (BGR) = FP
    overlay[fn] = [220, 0, 0]     # Blue (BGR) = FN

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Stage Visualization
# ──────────────────────────────────────────────────────────────────────────────

def visualize_pipeline_stages(stages: Dict[str, np.ndarray],
                              save_path: str,
                              title: str = "Segmentation Pipeline Stages") -> None:
    """Show each intermediate stage of the segmentation pipeline.

    Parameters
    ----------
    stages : dict
        Ordered mapping of {stage_name: image_ndarray}.
    save_path : str
        Output file path.
    title : str
        Figure super-title.
    """
    n = len(stages)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=14, fontweight="bold",
                 color=PALETTE["text"], y=1.02)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, (name, img) in zip(axes_flat, stages.items()):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=10, fontweight="bold", color=PALETTE["text"])
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Image Comparison Strip
# ──────────────────────────────────────────────────────────────────────────────

def create_comparison_strip(images: List[Dict],
                            save_path: str,
                            title: str = "Original → Prediction → Ground Truth") -> None:
    """Create a vertical strip comparing multiple images.

    Parameters
    ----------
    images : list of dict
        Each dict should have keys: 'original', 'prediction', 'ground_truth', 'label'.
        Values are np.ndarray images.
    save_path : str
        Output file path.
    """
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=14, fontweight="bold",
                 color=PALETTE["text"])

    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Original", "Prediction", "Ground Truth"]

    for row, item in enumerate(images):
        for col, key in enumerate(["original", "prediction", "ground_truth"]):
            img = item.get(key)
            if img is not None:
                if img.ndim == 3:
                    axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    axes[row, col].imshow(img, cmap="gray")

            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11,
                                         fontweight="bold")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        axes[row, 0].set_ylabel(item.get("label", ""), fontsize=9,
                                fontweight="bold", rotation=0,
                                labelpad=60, va="center")

    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
