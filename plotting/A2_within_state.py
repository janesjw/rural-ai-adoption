"""Build the main-text within-state metro comparison."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    BLUE,
    GRAY,
    LIGHT_GRAY,
    DERIVED_DIR,
    assert_current_inputs,
    bold_ticks,
    output_file,
    output_parser,
    require_file,
    save_png,
    set_style,
)


STATE_NAMES = {
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "FL": "Florida",
    "GA": "Georgia",
    "MD": "Maryland",
    "MI": "Michigan",
    "TX": "Texas",
    "WA": "Washington",
}


def main() -> None:
    args = output_parser("Rebuild the within-state metro comparison.").parse_args()
    set_style()
    results = assert_current_inputs()
    data = pd.read_csv(require_file(DERIVED_DIR / "btos_within_state.csv"))

    required = {"State", "ai_msa", "ai_rest", "gap", "covered_metros"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if len(data) != 9 or set(data["State"]) != set(STATE_NAMES):
        raise ValueError("The within-state figure must contain the nine eligible states.")
    reported_gap = results["within_state"]["gap_mean"]
    if not np.isclose(data["gap"].mean(), reported_gap, atol=1e-10):
        raise ValueError("Within-state plotted data do not match latest_results.json.")

    data = data.sort_values("gap", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for index, row in data.iterrows():
        ax.plot(
            [row["ai_rest"], row["ai_msa"]],
            [index, index],
            color=LIGHT_GRAY,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(
        data["ai_rest"],
        y,
        s=66,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=1.7,
        zorder=3,
        label="Rest of state",
    )
    ax.scatter(
        data["ai_msa"],
        y,
        s=66,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
        label="Covered top-25 metro area(s)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([STATE_NAMES[state] for state in data["State"]])
    ax.set_xlabel("Businesses reporting AI use (%)")
    all_values = np.r_[data["ai_rest"].to_numpy(), data["ai_msa"].to_numpy()]
    ax.set_xlim(np.floor(all_values.min()) - 1, np.ceil(all_values.max()) + 1)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", ncol=1)
    bold_ticks(ax)
    fig.tight_layout(pad=0.8)

    save_png(fig, output_file(args.out, "Figure_A2.png"))


if __name__ == "__main__":
    main()
