#!/usr/bin/env python3
"""Generate recognition dashboard charts."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DASHBOARD_DIR, RESULTS_DIR


COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#8BC34A", "#FFC107",
]


def _save(fig, name: str):
    out_dir = DASHBOARD_DIR / "recognition"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    results_path = RESULTS_DIR / "recognition_metrics.json"
    results = json.loads(results_path.read_text())

    models = list(results.keys())
    wer_values = [results[m]["micro_wer"] for m in models]

    # Sort by WER (ascending = better)
    order = np.argsort(wer_values)
    models = [models[i] for i in order]
    wer_values = [wer_values[i] for i in order]

    # Summary table
    fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.5 + 2)))
    ax.axis("off")
    rows = [[m, f"{results[m]['micro_wer']:.3f}", f"{results[m]['macro_wer']:.3f}"]
            for m in models]
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "WER (micro)", "WER (macro)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for j in range(3):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Recognition Benchmark — WER (lower is better)", fontsize=14, pad=20)
    _save(fig, "00_summary_table")

    # WER bar chart
    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.7)))
    bars = ax.barh(models, wer_values,
                   color=[COLORS[i % len(COLORS)] for i in range(len(models))])
    ax.set_xlim(0, max(wer_values) * 1.15)
    ax.set_xlabel("WER (lower is better)")
    ax.set_title("Word Error Rate by Model")
    for bar, val in zip(bars, wer_values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.invert_yaxis()
    _save(fig, "01_wer_by_model")

    print(f"Dashboard saved to {DASHBOARD_DIR / 'recognition'}")


if __name__ == "__main__":
    main()
