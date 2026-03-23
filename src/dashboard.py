"""Generate benchmark dashboard charts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import DASHBOARD_DIR, RESULTS_DIR


COLORS = [
    "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0",
    "#00BCD4", "#FF5722", "#607D8B", "#8BC34A", "#FFC107",
    "#3F51B5", "#795548", "#009688", "#CDDC39",
]


def load_results() -> dict:
    results_path = RESULTS_DIR / "metrics.json"
    return json.loads(results_path.read_text())


def _save(fig, name: str):
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = DASHBOARD_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def _sorted_bar(results: dict, metric: str, title: str, xlabel: str, filename: str):
    """Horizontal bar chart sorted by a given metric."""
    models = list(results.keys())
    values = [results[m]["aggregate"][metric] for m in models]

    order = np.argsort(values)[::-1]
    models = [models[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.7)))
    bars = ax.barh(models, values, color=[COLORS[i % len(COLORS)] for i in range(len(models))])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.invert_yaxis()
    _save(fig, filename)


def plot_summary_table(results: dict):
    """Summary table as an image."""
    models = list(results.keys())
    f1_scores = [results[m]["aggregate"]["f1"] for m in models]
    order = np.argsort(f1_scores)[::-1]
    models = [models[i] for i in order]

    rows = []
    for m in models:
        agg = results[m]["aggregate"]
        rows.append([
            m,
            f"{agg['precision']:.3f}",
            f"{agg['recall']:.3f}",
            f"{agg['f1']:.3f}",
            f"{agg.get('mean_area_coverage', 0):.3f}",
        ])

    fig, ax = plt.subplots(figsize=(12, max(4, len(models) * 0.5 + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Precision", "Recall", "F1", "Area Coverage"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for j in range(5):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Detection Benchmark Summary (IoU >= 0.5)", fontsize=14, pad=20)
    _save(fig, "00_summary_table")


def generate_all():
    """Generate all dashboard charts."""
    results = load_results()
    print("Generating dashboard...")

    plot_summary_table(results)

    _sorted_bar(results, "f1",
                "Detection F1 Score by Model",
                "F1 Score (IoU >= 0.5)",
                "01_f1_by_model")

    _sorted_bar(results, "precision",
                "Detection Precision by Model",
                "Precision (IoU >= 0.5)",
                "02_precision_by_model")

    _sorted_bar(results, "recall",
                "Detection Recall by Model",
                "Recall (IoU >= 0.5)",
                "03_recall_by_model")

    print(f"Dashboard saved to {DASHBOARD_DIR}")
