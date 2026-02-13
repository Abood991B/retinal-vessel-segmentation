# -*- coding: utf-8 -*-
"""
analysis.py — Statistical Analysis & Reporting for Segmentation Results
========================================================================

Functions for computing aggregate statistics, identifying best/worst cases,
performing parameter sensitivity analysis, and exporting structured reports.
"""

import csv
import os
from typing import List, Dict, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Summary Statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary_stats(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, min, max, and median for each metric.

    Parameters
    ----------
    results : list of dict
        Each dict contains metric keys (precision, recall, f1, iou, etc.)
        and an 'image' key.

    Returns
    -------
    dict
        Mapping of ``metric_name → {mean, std, min, max, median}``.
    """
    metric_keys = [k for k in results[0] if k != "image"]
    summary = {}

    for key in metric_keys:
        values = np.array([r[key] for r in results])
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    return summary


def identify_best_worst(results: List[Dict],
                        metric: str = "f1") -> Tuple[Dict, Dict]:
    """Identify the best and worst performing images by a given metric.

    Parameters
    ----------
    results : list of dict
        Per-image metric results.
    metric : str
        Metric name to rank by (default: F1-score).

    Returns
    -------
    tuple of (best_result, worst_result)
    """
    sorted_results = sorted(results, key=lambda r: r[metric])
    return sorted_results[-1], sorted_results[0]


def rank_images(results: List[Dict], metric: str = "f1",
                ascending: bool = False) -> List[Dict]:
    """Rank images by a metric, with rank index added.

    Parameters
    ----------
    results : list of dict
        Per-image results.
    metric : str
        Metric to sort by.
    ascending : bool
        If True, worst comes first.

    Returns
    -------
    list of dict
        Sorted results with an added 'rank' key.
    """
    sorted_res = sorted(results, key=lambda r: r[metric], reverse=not ascending)
    for i, r in enumerate(sorted_res, 1):
        r["rank"] = i
    return sorted_res


# ──────────────────────────────────────────────────────────────────────────────
# Composite Score
# ──────────────────────────────────────────────────────────────────────────────

def compute_composite_score(results: List[Dict],
                            weights: Dict[str, float] = None) -> List[Dict]:
    """Add a weighted composite score to each result.

    Parameters
    ----------
    results : list of dict
        Per-image metrics.
    weights : dict, optional
        Metric weights. Default: equal weight for precision, recall, iou.

    Returns
    -------
    list of dict
        Original results with 'composite_score' appended.
    """
    if weights is None:
        weights = {"precision": 0.25, "recall": 0.25, "f1": 0.25, "iou": 0.25}

    for r in results:
        score = sum(r.get(k, 0) * w for k, w in weights.items())
        r["composite_score"] = round(score, 6)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def export_detailed_report(results: List[Dict], save_path: str) -> None:
    """Export a comprehensive CSV report with per-image and summary rows.

    The CSV includes per-image metrics, then appends rows for:
    MEAN, STD, MIN, MAX, MEDIAN.
    """
    if not results:
        return

    metric_keys = [k for k in results[0] if k != "image" and k != "rank"]
    fieldnames = ["image"] + metric_keys

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

        # Summary rows
        summary = compute_summary_stats(results)
        for stat_name in ["mean", "std", "min", "max", "median"]:
            row = {"image": stat_name.upper()}
            for key in metric_keys:
                row[key] = round(summary[key][stat_name], 6)
            writer.writerow(row)

    print(f"  [REPORT] Exported → {save_path}")


def format_console_table(results: List[Dict],
                         metrics: List[str] = None) -> str:
    """Format results as a clean console-printable table.

    Parameters
    ----------
    results : list of dict
    metrics : list of str, optional
        Metrics to include. Default: precision, recall, f1, iou, error.

    Returns
    -------
    str
        Formatted table string.
    """
    if metrics is None:
        metrics = ["precision", "recall", "f1", "iou", "error"]

    header = f"  {'Image':<14}" + "".join(f"{m:>12}" for m in metrics)
    separator = "  " + "-" * (14 + 12 * len(metrics))
    lines = [header, separator]

    for r in results:
        row = f"  {r['image']:<14}"
        row += "".join(f"{r.get(m, 0):12.4f}" for m in metrics)
        lines.append(row)

    # Average row
    lines.append(separator)
    avg_row = f"  {'AVERAGE':<14}"
    for m in metrics:
        avg_val = np.mean([r.get(m, 0) for r in results])
        avg_row += f"{avg_val:12.4f}"
    lines.append(avg_row)

    return "\n".join(lines)
