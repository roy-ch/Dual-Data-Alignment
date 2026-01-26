#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polar grouped bar chart (radar-like) for multi-benchmark comparison.

This script is a standardized version of the original plotting code.
- Configurable via CLI arguments
- Clean function structure
- Reproducible output naming
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np


# -------------------------
# Default (example) config
# -------------------------
DEFAULT_OFFSET = 30
DEFAULT_MIN_H = 2.0

DEFAULT_METHODS = [
    "UnivFD",
    "SAFE",
    "AIDE",
    "DRCT",
    "AlignedForensics",
    "Ours",
]

DEFAULT_METRICS = [
    "DDA-COCO",
    "DRCT-2M",
    "GenImage",
    "EvalGEN",
    "Synthbuster",
    "Chameleon",
    "WildRF",
    "SynthWildx",
    "Foren\nSynths",
    "AIGC\nDetection\nBenchmark",
    "BFree\nOnline",
]

DEFAULT_SCORES = [
    [52.4, 61.8, 64.1, 15.4, 67.8, 50.7, 55.3, 52.3, 77.7, 72.9, 49.0],
    [49.9, 59.3, 50.3, 1.1, 46.5, 59.2, 54.2, 49.1, 49.7, 50.3, 50.5],
    [50.0, 64.6, 61.2, 19.1, 53.9, 63.1, 58.4, 54.7, 59.4, 63.6, 53.1],
    [60.2, 90.5, 84.7, 77.8, 84.8, 56.6, 51.6, 55.1, 73.9, 81.4, 55.7],
    [86.5, 95.5, 79.0, 68.0, 77.4, 71.0, 80.1, 78.7, 53.9, 66.6, 68.5],
    [92.2, 98.1, 91.7, 97.2, 90.1, 82.4, 90.3, 90.9, 82.1, 87.8, 95.1],
]

# Shuffle order for metrics (and each score row) to improve readability
DEFAULT_SHUFFLE = [2, 0, 1, 5, 3, 7, 6, 4, 8, 9, 10]

# Colors (reversed in original)
DEFAULT_COLORS = [
    "#253494",  # deep blue / purple (highlight)
    "#4C91C0",
    "#A0B7D9",
    "#C7D8C8",
    "#F2D8CC",
    "#DBA893",
][::-1]


def sort_by_indices(indices: Sequence[int], values: Sequence):
    """Reorder list by given indices."""
    return [values[i] for i in indices]


def validate_inputs(
    methods: Sequence[str],
    metrics: Sequence[str],
    scores: Sequence[Sequence[float]],
):
    if len(scores) != len(methods):
        raise ValueError(
            f"len(scores) must equal len(methods), got {len(scores)} vs {len(methods)}"
        )
    for r, row in enumerate(scores):
        if len(row) != len(metrics):
            raise ValueError(
                f"Row {r} in scores has length {len(row)} but metrics length is {len(metrics)}"
            )


def plot_polar_grouped_bars(
    methods: List[str],
    metrics: List[str],
    scores: List[List[float]],
    *,
    shuffle: Optional[List[int]] = None,
    offset: float = DEFAULT_OFFSET,
    min_h: float = DEFAULT_MIN_H,
    colors: Optional[List[str]] = None,
    figsize: tuple = (20, 20),
    theta_offset: float = 3 * math.pi / 8,
    bar_width_ratio: float = 0.8,
    default_alpha: float = 0.6,
    highlight_color: str = "#253494",
    label_fontsize: int = 30,
    metric_fontsize: int = 21,
    metric_label_alpha: float = 0.6,
    legend_fontsize: int = 24,
    legend_title_fontsize: int = 30,
    output_path: Optional[Path] = None,
    dpi: int = 300,
):
    """
    Draw the chart and save to output_path if provided.

    Notes:
    - Uses an "inner radius" to create a donut-like hole
    - Bars are grouped by method for each metric
    """
    if colors is None:
        colors = DEFAULT_COLORS

    validate_inputs(methods, metrics, scores)

    # Apply shuffle (metric order)
    if shuffle is not None:
        metrics = sort_by_indices(shuffle, metrics)
        scores = [sort_by_indices(shuffle, row) for row in scores]

    n_metrics = len(metrics)
    n_methods = len(methods)

    # Polar angles for each metric
    angles = [i * 2 * math.pi / n_metrics for i in range(n_metrics)]

    # Inner hole size (keeps original intent)
    inner_radius = 35 - offset * 0.35

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.subplot(111, polar=True)

    ax.set_theta_direction(-1)
    ax.set_theta_offset(theta_offset)

    # Move origin to create inner hole
    ax.set_rorigin(-inner_radius)

    # Group bar width (per metric), then split by method
    bar_width = (2 * math.pi / n_metrics) * bar_width_ratio

    # Legend patches
    legend_patches = []

    for i, method in enumerate(methods):
        method_scores = scores[i]
        method_color = colors[i % len(colors)]

        # Split bar group around center
        method_offset = (i - n_methods / 2) * (bar_width / n_methods)

        # Highlight method color alpha logic
        bar_alpha = 1.0 if method_color.lower() == highlight_color.lower() else default_alpha
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=method_color, alpha=bar_alpha))

        for j, (angle, score) in enumerate(zip(angles, method_scores)):
            adjusted_angle = angle + method_offset
            height = (score - offset) if (score > offset) else min_h

            ax.bar(
                adjusted_angle,
                height,
                width=bar_width / n_methods,
                bottom=inner_radius,
                color=method_color,
                alpha=bar_alpha,
                edgecolor="white",
                linewidth=0.4,
            )

            # Value label inside bar (keep original rotation heuristics)
            if (score - offset) > 10:
                bar_center = inner_radius + (score - offset) - 9
                if j < 4:
                    rotation = -adjusted_angle * 180 / math.pi - 90 + 180 - 45 / 2
                else:
                    rotation = -adjusted_angle * 180 / math.pi - 90 - 45 / 2

                ax.text(
                    adjusted_angle,
                    bar_center,
                    f"{score}",
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                    color="white",
                    fontweight="bold",
                    rotation=rotation,
                )

    # Metric name labels near inner circle
    for i, metric in enumerate(metrics):
        angle = angles[i]
        r_position = 2 * inner_radius / 3

        angle_deg = np.degrees(-angle) + np.pi / 8
        if 90 < angle_deg < 270:
            rotation = angle_deg + 180 - 45 / 2
        else:
            rotation = angle_deg - 45 / 2

        ax.text(
            angle,
            r_position,
            metric,
            ha="center",
            va="center",
            fontsize=metric_fontsize,
            fontweight="bold",
            color="black",
            rotation=rotation,
            rotation_mode="anchor",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=metric_label_alpha,
                pad=2,
            ),
        )

    # Hide default ticks/grids/spines
    ax.set_xticks(angles)
    ax.set_xticklabels(["" for _ in metrics])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    plt.legend(
        legend_patches,
        methods,
        title_fontsize=legend_title_fontsize,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        fontsize=legend_fontsize,
        ncol=len(methods),
        frameon=False,
    )

    plt.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Draw a polar grouped bar chart (radar-like) from hard-coded benchmark scores."
    )
    p.add_argument("--out", type=str, default=None, help="Output image path, e.g. outputs/plot.png")
    p.add_argument("--offset", type=float, default=DEFAULT_OFFSET, help="Offset used to create inner hole & normalize heights")
    p.add_argument("--min-h", type=float, default=DEFAULT_MIN_H, help="Minimum bar height when score <= offset")
    p.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    p.add_argument("--no-shuffle", action="store_true", help="Disable metric shuffling")
    return p


def main():
    args = build_argparser().parse_args()

    methods = list(DEFAULT_METHODS)
    metrics = list(DEFAULT_METRICS)
    scores = [list(r) for r in DEFAULT_SCORES]

    shuffle = None if args.no_shuffle else list(DEFAULT_SHUFFLE)

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        out_path = Path(f"bfs_coco_benchmark_comparison_offset{int(args.offset)}.png")

    plot_polar_grouped_bars(
        methods=methods,
        metrics=metrics,
        scores=scores,
        shuffle=shuffle,
        offset=args.offset,
        min_h=args.min_h,
        output_path=out_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
