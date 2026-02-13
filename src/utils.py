# -*- coding: utf-8 -*-
"""
utils.py — Shared Utility Functions
=====================================
"""

import os
import time
import functools
from typing import List


def list_image_files(directory: str,
                     extensions: set = None) -> List[str]:
    """Return sorted list of image filenames in a directory.

    Parameters
    ----------
    directory : str
        Path to the image directory.
    extensions : set, optional
        Allowed file extensions (lowercase, with dot).
        Default: {'.png', '.jpg', '.jpeg', '.tif', '.bmp'}
    """
    if extensions is None:
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".bmp"}

    return sorted(
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in extensions
    )


def ensure_directory(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def timer(func):
    """Decorator that prints execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper
