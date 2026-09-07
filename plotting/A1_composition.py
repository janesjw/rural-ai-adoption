"""Build Appendix Figure A1: additive composition accounting."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    BLUE,
    GRAY,
    NAVY,
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


ORDER = [
    "Observed",
    "Industry-predicted",
    "Size-predicted",
    "After both adjustments",
]
LABELS = {
    "Observed": "Observed gradient",
    "Industry-predicted": "Predicted by industry",
    "Size-predicted": "Predicted by firm size",
    "After both adjustments": "Remaining after both",
}
COLOURS = {
    "Observed": NAVY,
    "Industry-predicted": ORANGE,
    "Size-predicted": GRAY,
    "After both adjustments": BLUE,
}


def main() -> None:
    args = output_parser("Rebuild the composition-accounting figure.").parse_args()
    set_style()
    results = assert_current_inputs()
    data = pd.read_csv(require_file(DERIVED_DIR / "composition_summary.csv"))

    if set(data["measure"]) != set(ORDER):
        raise ValueError("composition_summary.csv does not contain the four expected rows.")
    data = data.set_index("measure").loc[ORDER].reset_index()
    slopes = data.set_index("measure")["slope"]
    if not np.isclose(
        slopes["Industry-predicted"]
        + slopes["Size-predicted"]
        + slopes["After both adjustments"],
        slopes["Observed"],
        atol=1e-12,
    ):
        raise ValueError("The additive composition decomposition does not sum exactly.")
    if not np.isclose(
        slopes["After both adjustments"],
        results["cross_section"]["adjusted_slope"],
        atol=1e-12,
    ):
        raise ValueError("Composition figure does not match latest_results.json.")

    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colours = [COLOURS[name] for name in data["measure"]]
    ax.barh(
        y,
        data["slope"],
        height=0.58,
        color=colours,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    for index, row in data.iterrows():
        long_bar = abs(row["slope"]) >= 0.04
        ax.text(
            row["slope"] / 2 if long_bar else row["slope"] - 0.003,
            index,
            f'{row["slope"]:.3f}',
            ha="center" if long_bar else "right",
            va="center",
            color="white" if long_bar else "#252525",
            fontsize=11,
            fontweight="bold",
            zorder=4,
        )

    ax.axvline(0, color="#252525", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[name] for name in data["measure"]])
    ax.invert_yaxis()
    ax.set_xlabel("AI-use slope (pp per 1 pp higher rural share)")
    ax.set_xlim(-0.112, 0.008)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    bold_ticks(ax)
    fig.tight_layout(pad=0.8)

    save_png(fig, output_file(args.out, "Figure_A1.png"))


if __name__ == "__main__":
    main()
