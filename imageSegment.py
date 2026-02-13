# -*- coding: utf-8 -*-
"""
Retinal Vessel Segmentation — Enhanced Multi-Stage Pipeline
=============================================================

An advanced classical image processing pipeline for binary segmentation of
retinal blood vessels from fundus photographs.  The implementation achieves
high accuracy through four key innovations over a naive thresholding approach:

    1. **Morphological Background Subtraction** — Estimates and removes the
       non-uniform illumination pattern via morphological closing, so that
       vessels in both bright (optic-disc) and dark (peripheral) regions are
       uniformly enhanced.  This is the single biggest driver of improved
       *recall*.

    2. **CLAHE with Fine-Grained Tiling** — Adaptive histogram equalization
       on an 8x8 grid provides highly localised contrast amplification,
       revealing faint capillaries that a coarse grid would miss.

    3. **Local Mean-C Thresholding** — A proven, robust binarisation scheme:
       a pixel is classified as vessel if its intensity exceeds the local
       mean of a configurable neighbourhood by at least *threshold_offset*
       units.  Because background subtraction makes vessels bright, the
       offset is now positive (vessels must be *brighter* than the local
       mean).

    4. **Connected-Component Area Filtering** — Removes small isolated blobs
       whose area (in pixels) falls below a configurable minimum, which
       dramatically improves *precision* without affecting true vessels.

Pipeline Stages (9):
    1.  Green Channel Extraction
    2.  Morphological Background Subtraction
    3.  CLAHE Enhancement  (fine-grained 8x8)
    4.  Median Noise Filtering
    5.  Local Mean-C Adaptive Thresholding
    6.  Morphological Close -> Open
    7.  Connected-Component Area Filtering
    8.  ROI Masking  (retinal disc isolation)
    9.  Gaussian Smoothing + Final Binarisation

Author : Abdo Hussam
Course : CDS6334 — Visual Information Processing
Date   : 2025
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter Presets
# ──────────────────────────────────────────────────────────────────────────────
# Two independently tuned presets — one per dataset.
# Key parameters:
#   bg_kernel_size     — structuring-element diameter for background estimation
#                        (must exceed the widest vessel diameter in the fundus).
#   threshold_offset   — minimum brightness above local mean to be a vessel;
#                        higher = more conservative (better precision, lower recall).
#   min_vessel_area    — minimum connected-component area in px to survive
#                        post-processing (removes salt noise).

CONFIG_DATASET = {
    # Stage 2 – background subtraction
    "bg_kernel_size": 21,

    # Stage 3 – CLAHE
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": (8, 8),

    # Stage 4 – median filter
    "median_ksize": 3,

    # Stage 5 – local mean-C thresholding
    "mean_filter_size": 18,
    "threshold_offset": 6,

    # Stage 6 – morphological cleanup
    "morph_kernel_size": 3,

    # Stage 7 – area filtering
    "min_vessel_area": 120,

    # Stage 8 – ROI mask
    "mask_kernel_size": 7,

    # Stage 9 – final smoothing & binarisation
    "gaussian_ksize": (3, 3),
    "gaussian_sigma": 0.5,
    "binary_threshold": 128,
}

CONFIG_ADD_DATASET = {
    "bg_kernel_size": 21,
    "clahe_clip_limit": 2.5,
    "clahe_grid_size": (8, 8),
    "median_ksize": 3,
    "mean_filter_size": 18,
    "threshold_offset": 6,
    "morph_kernel_size": 3,
    "min_vessel_area": 50,
    "mask_kernel_size": 7,
    "gaussian_ksize": (3, 3),
    "gaussian_sigma": 0.5,
    "binary_threshold": 128,
}

# The currently active preset (switched by set_config).
_active_config: Dict[str, Any] = CONFIG_DATASET


# ──────────────────────────────────────────────────────────────────────────────
# Configuration API
# ──────────────────────────────────────────────────────────────────────────────

def set_config(preset: str = "dataset") -> None:
    """Switch the active hyperparameter preset.

    Parameters
    ----------
    preset : str
        ``'dataset'`` for the 80-image primary dataset, or
        ``'add_dataset'`` for the 20-image supplementary dataset.
    """
    global _active_config
    if preset == "add_dataset":
        _active_config = CONFIG_ADD_DATASET
    else:
        _active_config = CONFIG_DATASET


def get_config() -> Dict[str, Any]:
    """Return a copy of the currently active configuration."""
    return dict(_active_config)


# ──────────────────────────────────────────────────────────────────────────────
# Individual Stage Functions
# ──────────────────────────────────────────────────────────────────────────────

def _extract_green_channel(img_bgr: np.ndarray) -> np.ndarray:
    """Stage 1 — Extract the green channel.

    In RGB retinal fundus photographs the green channel delivers the
    highest contrast between blood vessels (dark) and surrounding tissue
    (bright).  The red channel is often saturated and the blue channel
    suffers from low signal-to-noise ratio.
    """
    return img_bgr[:, :, 1]


def _subtract_background(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    """Stage 2 — Morphological background subtraction.

    1. A morphological *closing* (dilate then erode) with a large elliptical
       structuring element fills in dark features (vessels) smaller than the
       kernel, producing a smooth estimate of the background illumination.
    2. Subtracting the original channel from this estimate yields an image
       where vessels are bright and most of the slowly-varying illumination
       has been removed.

    This is the primary mechanism for normalising the widely varying
    brightness across a retinal fundus image and is the biggest single
    contributor to improved recall.

    Parameters
    ----------
    channel : np.ndarray
        Single-channel uint8 image (green channel).
    kernel_size : int
        Structuring-element diameter.  Must be larger than the widest vessel.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (kernel_size, kernel_size))
    background = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, se)
    return cv2.subtract(background, channel)       # vessels now bright


def _apply_clahe(image: np.ndarray, clip_limit: float,
                 grid_size: Tuple[int, int]) -> np.ndarray:
    """Stage 3 — CLAHE contrast enhancement.

    Applied *after* background subtraction so that the equalisation
    can focus on vessel-vs-local-tissue contrast rather than fighting
    the global illumination gradient.  An 8x8 tile grid provides
    fine-grained local adaptation.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def _median_filter(image: np.ndarray, ksize: int) -> np.ndarray:
    """Stage 4 — Median noise suppression.

    Removes impulse (salt-and-pepper) noise while keeping vessel edges
    sharp.  A 3x3 kernel is typically sufficient.
    """
    if ksize < 3:
        return image
    return cv2.medianBlur(image, ksize)


def _mean_c_threshold(image: np.ndarray, filter_size: int,
                      offset: float) -> np.ndarray:
    """Stage 5 — Local mean-C adaptive thresholding.

    For each pixel, the mean intensity in a ``filter_size x filter_size``
    neighbourhood is computed.  A pixel is classified as *vessel* (255) if

        pixel_value  >  local_mean  +  offset

    Because background subtraction has already made vessels bright, a
    **positive** offset is used.  Higher offsets yield fewer (but more
    confident) detections — i.e. higher precision at the cost of recall.

    Parameters
    ----------
    image : np.ndarray
        Enhanced grayscale image (uint8) where vessels are bright.
    filter_size : int
        Side length of the square mean-filter kernel (e.g. 18).
    offset : float
        Minimum brightness above local mean for a vessel classification.
    """
    n = filter_size
    kernel = np.ones((n, n), dtype=np.float32) / (n * n)
    local_mean = cv2.filter2D(image, cv2.CV_64F, kernel)

    diff = image.astype(np.float64) - local_mean
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[diff > offset] = 255
    return binary


def _morphological_cleanup(binary: np.ndarray, ksize: int) -> np.ndarray:
    """Stage 6 — Morphological close then open.

    * **Closing** bridges narrow gaps between vessel fragments.
    * **Opening** removes small isolated noise specks.

    An elliptical structuring element provides isotropic treatment.
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, se)
    return opened


def _filter_small_components(binary: np.ndarray,
                             min_area: int) -> np.ndarray:
    """Stage 7 — Connected-component area filtering.

    Identifies every 8-connected white component and removes any whose
    area (number of pixels) is below ``min_area``.  This is one of the
    most effective steps for boosting *precision* — it eliminates small
    scattered false positives that morphological opening alone cannot
    reach, while genuine vessel segments (which are elongated and thus
    have substantial area) are preserved.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    cleaned = binary.copy()
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0
    return cleaned


def _create_roi_mask(img_bgr: np.ndarray, ksize: int) -> np.ndarray:
    """Stage 8 — Retinal disc ROI mask.

    Converts to greyscale, applies Otsu thresholding, selects the largest
    contour (the retinal field-of-view), and smooths edges with morphology.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)
    return mask


def _smooth_and_binarise(image: np.ndarray, gauss_ksize: Tuple[int, int],
                         gauss_sigma: float,
                         threshold: int) -> np.ndarray:
    """Stage 9 — Light Gaussian smoothing + hard threshold.

    A small-kernel Gaussian softens any remaining jagged edges, then a
    hard threshold produces the final {0, 1} vessel mask.
    """
    smoothed = cv2.GaussianBlur(image, gauss_ksize, gauss_sigma)
    _, mask = cv2.threshold(smoothed, threshold, 1, cv2.THRESH_BINARY)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Public API  (required by evaluateSegment.py)
# ──────────────────────────────────────────────────────────────────────────────

def segmentImage(inputImg: np.ndarray) -> np.ndarray:
    """Segment retinal blood vessels from a fundus photograph.

    Parameters
    ----------
    inputImg : np.ndarray
        BGR colour image of shape ``(H, W, 3)``.

    Returns
    -------
    np.ndarray
        Binary mask of shape ``(H, W)`` with values in ``{0, 1}``:
        **1** = vessel, **0** = background.

    Algorithm (9 stages)
    --------------------
    1. **Green channel** — maximum vessel/background contrast.
    2. **Background subtraction** — morphological-closing-based illumination
       normalisation; vessels become uniformly bright.
    3. **CLAHE** — fine-grained adaptive histogram equalization (8x8).
    4. **Median filter** — impulse-noise suppression.
    5. **Mean-C threshold** — local-mean adaptive binarisation; vessel pixels
       must exceed the neighbourhood mean by *threshold_offset*.
    6. **Morphological close/open** — bridge vessel gaps, remove specks.
    7. **Area filter** — discard connected components < *min_vessel_area*.
    8. **ROI mask** — restrict output to the retinal disc.
    9. **Smooth + binarise** — Gaussian edge softening, then hard threshold
       to ``{0, 1}``.
    """
    cfg = _active_config

    # 1. Green channel
    green = _extract_green_channel(inputImg)

    # 2. Background subtraction  (vessels become bright)
    vessel = _subtract_background(green, cfg["bg_kernel_size"])

    # 3. CLAHE
    enhanced = _apply_clahe(vessel, cfg["clahe_clip_limit"],
                            cfg["clahe_grid_size"])

    # 4. Median filter
    filtered = _median_filter(enhanced, cfg["median_ksize"])

    # 5. Mean-C adaptive threshold
    binary = _mean_c_threshold(filtered,
                               cfg["mean_filter_size"],
                               cfg["threshold_offset"])

    # 6. Morphological refinement
    cleaned = _morphological_cleanup(binary, cfg["morph_kernel_size"])

    # 7. Area filtering
    cleaned = _filter_small_components(cleaned, cfg["min_vessel_area"])

    # 8. ROI mask
    roi = _create_roi_mask(inputImg, cfg["mask_kernel_size"])
    masked = cv2.bitwise_and(cleaned, roi)

    # 9. Smooth + binarise
    outputImg = _smooth_and_binarise(masked,
                                     cfg["gaussian_ksize"],
                                     cfg["gaussian_sigma"],
                                     cfg["binary_threshold"])
    return outputImg
