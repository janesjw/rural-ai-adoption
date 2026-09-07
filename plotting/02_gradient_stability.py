"""Build Appendix Figure A2: wave-specific gradients and adoption levels."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    BLUE,
    LIGHT_BLUE,
    ORANGE,
    DARK_GRAY,
    DERIVED_DIR,
    assert_current_inputs,
    bold_ticks,
    output_file,
    output_parser,
    require_file,
    save_png,
    set_style,
)


def period_label(period: str) -> str:
    """Format a BTOS period identifier without calling it a calendar week."""
    period = str(period)
    return f"{period[:4]}-{period[4:]}"


def main() -> None:
    args = output_parser("Rebuild the wave-by-wave stability figure.").parse_args()
    set_style()
    results = assert_current_inputs()
    data = pd.read_csv(
        require_file(DERIVED_DIR / "gradient_by_wave.csv"),
        dtype={"period": str},
    ).sort_values("period").reset_index(drop=True)

    expected_periods = results["data"]["waves"]
    if data["period"].tolist() != expected_periods:
        raise ValueError("gradient_by_wave.csv does not contain the current 20 periods.")
    if len(data) != 20 or not (data["slope"] < 0).all():
        raise ValueError("The wave figure must contain 20 negative rural gradients.")
    if (data["lo"] > data["slope"]).any() or (data["hi"] < data["slope"]).any():
        raise ValueError("A reported estimate lies outside its confidence interval.")

    x = np.arange(len(data))
    errors = np.vstack(
        [
            data["slope"].to_numpy() - data["lo"].to_numpy(),
            data["hi"].to_numpy() - data["slope"].to_numpy(),
        ]
    )
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0], "hspace": 0.12},
    )

    top.axhline(0, color=DARK_GRAY, linewidth=0.9, zorder=1)
    top.errorbar(
        x,
        data["slope"],
        yerr=errors,
        fmt="o",
        linestyle="none",
        markersize=5.8,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.7,
        color=BLUE,
        ecolor=LIGHT_BLUE,
        elinewidth=1.5,
        capsize=2.8,
        capthick=1.2,
        zorder=3,
    )
    top.set_title("(a) Rural AI-use gradient by BTOS period", loc="left")
    top.set_ylabel("Gradient\n(pp per 1 pp rural share)")
    top.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    top.set_axisbelow(True)
    bold_ticks(top)

    bottom.plot(
        x,
        data["level"],
        color=ORANGE,
        linewidth=2.2,
        marker="o",
        markersize=5.2,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    bottom.set_title("(b) Mean reported AI use", loc="left")
    bottom.set_ylabel("AI use (%)")
    bottom.set_xlabel("BTOS period")
    bottom.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    bottom.set_axisbelow(True)

    tick_positions = list(range(0, len(data), 2))
    if tick_positions[-1] != len(data) - 1:
        tick_positions.append(len(data) - 1)
    bottom.set_xticks(tick_positions)
    bottom.set_xticklabels(
        [period_label(data.loc[index, "period"]) for index in tick_positions],
        rotation=40,
        ha="right",
    )
    bottom.set_xlim(-0.6, len(data) - 0.4)
    bold_ticks(bottom)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.96, bottom=0.16)

    save_png(fig, output_file(args.out, "Figure_2.png"))


if __name__ == "__main__":
    main()
