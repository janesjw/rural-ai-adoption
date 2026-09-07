"""Rebuild the supplementary rural business-formation event study."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    BLUE,
    DARK_GRAY,
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


def main() -> None:
    args = output_parser("Rebuild supplementary Figure S1.").parse_args()
    set_style()
    assert_current_inputs()
    data = pd.read_csv(require_file(DERIVED_DIR / "FINAL_event_study.csv"))

    required = {"year", "coef", "se"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    data = data.loc[data["year"].between(2012, 2025)].sort_values("year")
    if set(data["year"]) != set(range(2012, 2026)):
        raise ValueError("Figure S1 requires annual estimates from 2012 through 2025.")
    baseline = data.loc[data["year"].eq(2019)]
    if len(baseline) != 1 or not np.isclose(baseline["coef"].iloc[0], 0.0):
        raise ValueError("Figure S1 must use 2019 as the zero baseline.")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.errorbar(
        data["year"],
        data["coef"],
        yerr=1.96 * data["se"],
        fmt="o-",
        markersize=5.5,
        linewidth=1.8,
        elinewidth=1.25,
        capsize=3.2,
        capthick=1.25,
        color=BLUE,
        ecolor=BLUE,
        zorder=3,
    )
    ax.axhline(0, color=DARK_GRAY, linewidth=1.0, zorder=1)
    ax.axvline(2022.91, color=ORANGE, linestyle="--", linewidth=1.5, zorder=2)
    ax.text(
        2023.03,
        0.04,
        "ChatGPT",
        transform=ax.get_xaxis_transform(),
        color=ORANGE,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Coefficient on exposure x year")
    ax.set_xticks(np.arange(2012, 2026, 2))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    bold_ticks(ax)
    fig.tight_layout(pad=0.8)
    save_png(fig, output_file(args.out, "Figure_A3.png"))


if __name__ == "__main__":
    main()
