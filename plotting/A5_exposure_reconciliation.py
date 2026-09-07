"""Rebuild the supplementary comparison of exposure-based outcomes."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    BLUE,
    DARK_GRAY,
    GRAY,
    ORANGE,
    DERIVED_DIR,
    assert_current_inputs,
    bold_ticks,
    output_file,
    output_parser,
    require_file,
    save_png,
    set_style,
)


SERIES = (
    ("bfs3_event_study.csv", "BFS, 3-digit industries", BLUE, "o"),
    ("monthly_event_all.csv", "BFS, 19 sectors", ORANGE, "s"),
    ("industry_event_study.csv", "QCEW, 3-digit industries", GRAY, "^"),
)


def main() -> None:
    args = output_parser("Rebuild supplementary Figure S3.").parse_args()
    set_style()
    assert_current_inputs()

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for filename, label, color, marker in SERIES:
        data = pd.read_csv(require_file(DERIVED_DIR / filename))
        required = {"year", "coef"}
        if not required.issubset(data.columns):
            raise ValueError(f"{filename} is missing {sorted(required - set(data.columns))}")
        data = data.loc[data["year"] >= 2015].sort_values("year")
        baseline = data.loc[data["year"].eq(2019), "coef"]
        if data.empty or len(baseline) != 1 or not np.isclose(baseline.iloc[0], 0.0):
            raise ValueError(f"{filename} must contain a zero-valued 2019 baseline.")
        ax.plot(
            data["year"],
            data["coef"],
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.9,
            label=label,
            zorder=3,
        )

    ax.axhline(0, color=DARK_GRAY, linewidth=1.0, zorder=1)
    ax.axvline(2022.91, color=DARK_GRAY, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(
        2023.03,
        0.04,
        "ChatGPT",
        transform=ax.get_xaxis_transform(),
        color=DARK_GRAY,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient on exposure x year")
    ax.set_xticks(np.arange(2016, 2027, 2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    bold_ticks(ax)
    fig.tight_layout(pad=0.8)
    save_png(fig, output_file(args.out, "Figure_A5.png"))


if __name__ == "__main__":
    main()
