"""Shared styling, paths, and data-version checks for paper figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = PACKAGE_ROOT / "data" / "derived"
RAW_DIR = PACKAGE_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "figures" / "rebuilt"

# Restrained, conventional colours used consistently across all figures.
BLUE = "#2F6B9A"
LIGHT_BLUE = "#9EC3DD"
NAVY = "#173F5F"
ORANGE = "#D97732"
LIGHT_ORANGE = "#E8B07A"
GRAY = "#727272"
LIGHT_GRAY = "#D6D6D6"
DARK_GRAY = "#252525"


def output_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rebuilt PNG files (default: figures/rebuilt).",
    )
    return parser


def output_file(out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input is missing: {path}\n"
            "Run the current analysis script before rebuilding the figures."
        )
    return path


def assert_current_inputs() -> dict:
    """Reject stale derived files before any figure is drawn."""
    path = require_file(DERIVED_DIR / "latest_results.json")
    with path.open("r", encoding="utf-8") as stream:
        results = json.load(stream)
    data = results.get("data", {})
    waves = [str(value) for value in data.get("waves", [])]
    latest_window = [str(value) for value in data.get("latest_window", [])]
    checks = {
        "20 BTOS waves": len(waves) == 20,
        "latest BTOS period 202617": bool(waves) and waves[-1] == "202617",
        "eight-period headline window": len(latest_window) == 8,
        "headline window ends at 202617": bool(latest_window) and latest_window[-1] == "202617",
        "51 state-level units": data.get("units") == 51,
    }
    failures = [label for label, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError(
            "Derived plotting data are stale or inconsistent: " + "; ".join(failures)
        )
    return results


def set_style() -> None:
    """Apply the requested Arial, bold, large-text paper style."""
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        font_manager.findfont(
            font_manager.FontProperties(family="Arial", weight="bold"),
            fallback_to_default=False,
        )
    except ValueError as error:
        raise RuntimeError(
            "Arial regular and bold are required. Install Arial before plotting."
        ) from error

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 13,
            "font.weight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.edgecolor": DARK_GRAY,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def bold_ticks(ax) -> None:
    """Force tick labels created after rcParams setup to Arial bold."""
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily("Arial")
        label.set_fontweight("bold")


def save_png(fig, destination: Path) -> None:
    """Export only the requested high-resolution PNG."""
    fig.savefig(destination, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {destination}")
