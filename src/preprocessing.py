# -*- coding: utf-8 -*-
"""
preprocessing.py — Image Enhancement Utilities for Retinal Fundus Images
=========================================================================

This module provides a composable set of preprocessing functions designed for
retinal fundus photography. Each function is self-contained and follows a
consistent interface: takes an image (ndarray) and returns a processed image.

Functions mirror the stages of the main segmentation pipeline but can be
used independently for experimentation or ablation studies.
"""

import cv2
import numpy as np
from typing import Tuple


def extract_channel(img_bgr: np.ndarray, channel: str = "green") -> np.ndarray:
    """Extract a single colour channel from a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR format (H, W, 3).
    channel : str
        One of ``'blue'``, ``'green'``, ``'red'``, or ``'gray'``.

    Returns
    -------
    np.ndarray
        Single-channel image (H, W), dtype uint8.
    """
    channel_map = {"blue": 0, "green": 1, "red": 2}
    if channel == "gray":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if channel not in channel_map:
        raise ValueError(f"Unknown channel '{channel}'. "
                         f"Choose from {list(channel_map) + ['gray']}")
    return img_bgr[:, :, channel_map[channel]]


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                grid_size: Tuple[int, int] = (3, 3)) -> np.ndarray:
    """Enhance local contrast with CLAHE.

    Contrast-Limited Adaptive Histogram Equalization divides the image into
    tiles and equalizes each tile's histogram independently, with a clip
    limit to prevent over-amplification of noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), dtype uint8.
    clip_limit : float
        Threshold for contrast limiting (higher = more contrast).
    grid_size : tuple of int
        Number of tiles in each dimension.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image, same shape and dtype.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def median_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Suppress impulse (salt-and-pepper) noise with a median filter.

    The median filter replaces each pixel with the median of its neighbourhood,
    effectively removing outlier pixels while preserving edges better than
    a mean filter.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W).
    ksize : int
        Kernel size (must be odd and ≥ 1).
    """
    if ksize < 3:
        return image  # ksize=1 is a no-op; skip entirely
    return cv2.medianBlur(image, ksize)


def gaussian_smooth(image: np.ndarray, ksize: Tuple[int, int] = (5, 5),
                    sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur for edge smoothing.

    Parameters
    ----------
    image : np.ndarray
        Input image (any dtype).
    ksize : tuple
        Kernel size (width, height); both must be odd.
    sigma : float
        Standard deviation of the Gaussian kernel.
    """
    return cv2.GaussianBlur(image, ksize, sigma)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize a floating-point image to [0, 255] uint8 range.

    Handles NaN values by replacing them with 0 before normalization.
    """
    img = image.astype(np.float64)
    img[np.isnan(img)] = 0
    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin) * 255.0
    return img.astype(np.uint8)


def create_retinal_mask(img_bgr: np.ndarray,
                        morph_ksize: int = 5) -> np.ndarray:
    """Create a binary mask isolating the circular retinal field-of-view.

    Uses Otsu thresholding to separate the retina from the black surround,
    then selects the largest connected component and refines with morphology.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input fundus image in BGR format.
    morph_ksize : int
        Kernel size for morphological smoothing of the mask boundary.

    Returns
    -------
    np.ndarray
        Binary mask (H, W) with 255 inside the retina, 0 outside.
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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask
