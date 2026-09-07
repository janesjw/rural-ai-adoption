"""Build the main three-panel rural AI-adoption figure.

Panel (a) shows reported AI use, panel (b) removes additive industry and
firm-size composition, and panel (c) reports adjusted means by rurality
quartile without imposing a linear functional form.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import zipfile

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from _common import (
    BLUE,
    DARK_GRAY,
    DERIVED_DIR,
    RAW_DIR,
    assert_current_inputs,
    bold_ticks,
    output_file,
    output_parser,
    require_file,
    save_png,
    set_style,
)


@dataclass
class StateGeometry:
    state: str
    rings: list[np.ndarray]


CONUS_ALBERS = (29.5, 45.5, 23.0, -96.0)
ALASKA_ALBERS = (55.0, 65.0, 50.0, -154.0)
HAWAII_ALBERS = (8.0, 18.0, 3.0, -157.0)

# Approximate label anchors. Small northeastern states use point offsets below.
LABEL_LON_LAT = {
    "AL": (-86.8, 32.8), "AK": (-152.0, 64.0), "AZ": (-111.8, 34.2),
    "AR": (-92.3, 34.8), "CA": (-119.5, 37.1), "CO": (-105.5, 39.0),
    "CT": (-72.7, 41.6), "DE": (-75.5, 39.0), "DC": (-77.0, 38.9),
    "FL": (-82.4, 28.2), "GA": (-83.4, 32.7), "HI": (-157.5, 20.5),
    "ID": (-114.4, 44.2), "IL": (-89.2, 40.0), "IN": (-86.1, 40.0),
    "IA": (-93.5, 42.0), "KS": (-98.3, 38.5), "KY": (-85.3, 37.7),
    "LA": (-92.3, 31.0), "ME": (-69.0, 45.2), "MD": (-76.7, 39.0),
    "MA": (-71.8, 42.2), "MI": (-85.5, 44.2), "MN": (-94.3, 46.0),
    "MS": (-89.7, 32.7), "MO": (-92.5, 38.5), "MT": (-110.4, 47.0),
    "NE": (-99.8, 41.5), "NV": (-116.6, 39.2), "NH": (-71.6, 43.8),
    "NJ": (-74.6, 40.2), "NM": (-106.0, 34.5), "NY": (-75.5, 43.0),
    "NC": (-79.5, 35.5), "ND": (-100.5, 47.5), "OH": (-82.7, 40.3),
    "OK": (-97.5, 35.5), "OR": (-120.5, 44.0), "PA": (-77.7, 40.9),
    "RI": (-71.6, 41.7), "SC": (-80.9, 33.8), "SD": (-100.0, 44.5),
    "TN": (-86.2, 35.8), "TX": (-99.3, 31.2), "UT": (-111.6, 39.3),
    "VT": (-72.7, 44.0), "VA": (-78.5, 37.6), "WA": (-120.5, 47.4),
    "WV": (-80.6, 38.6), "WI": (-89.9, 44.5), "WY": (-107.5, 43.0),
}

SMALL_STATE_LABELS = {
    "NH": (0.985, 0.68),
    "VT": (0.985, 0.61),
    "MA": (0.985, 0.54),
    "RI": (0.985, 0.47),
    "CT": (0.985, 0.40),
    "NJ": (0.985, 0.32),
    "MD": (0.985, 0.24),
    "DE": (0.985, 0.17),
    "DC": (0.985, 0.10),
}


def read_dbf_state_codes(raw: bytes) -> list[str]:
    """Read the STUSPS field from the DBF bundled with the shapefile."""
    record_count = struct.unpack("<I", raw[4:8])[0]
    header_length = struct.unpack("<H", raw[8:10])[0]
    record_length = struct.unpack("<H", raw[10:12])[0]
    fields: list[tuple[str, int, int]] = []
    position = 32
    field_offset = 1
    while raw[position] != 0x0D:
        descriptor = raw[position:position + 32]
        name = descriptor[:11].split(b"\0")[0].decode("ascii")
        width = descriptor[16]
        fields.append((name, field_offset, width))
        field_offset += width
        position += 32
    state_field = next((field for field in fields if field[0] == "STUSPS"), None)
    if state_field is None:
        raise ValueError("The Census DBF file does not contain STUSPS.")
    _, start, width = state_field
    states = []
    for index in range(record_count):
        record_start = header_length + index * record_length
        record = raw[record_start:record_start + record_length]
        if record[:1] == b"*":
            raise ValueError("The Census DBF has a deleted record.")
        states.append(record[start:start + width].decode("ascii").strip())
    return states


def read_state_geometries(archive: Path) -> dict[str, StateGeometry]:
    """Read state polygons directly from the Census ZIP without GIS packages."""
    with zipfile.ZipFile(archive) as zipped:
        shape_member = next(name for name in zipped.namelist() if name.lower().endswith(".shp"))
        dbf_member = next(name for name in zipped.namelist() if name.lower().endswith(".dbf"))
        shape_data = zipped.read(shape_member)
        state_codes = read_dbf_state_codes(zipped.read(dbf_member))
    if struct.unpack("<i", shape_data[32:36])[0] != 5:
        raise ValueError("Expected Census polygon geometry (shape type 5).")

    geometries = {}
    position = 100
    record_index = 0
    while position < len(shape_data):
        content_words = struct.unpack(">i", shape_data[position + 4:position + 8])[0]
        content_start = position + 8
        content_end = content_start + 2 * content_words
        content = shape_data[content_start:content_end]
        if len(content) != 2 * content_words or record_index >= len(state_codes):
            raise ValueError("Census shapefile records are truncated or misaligned.")
        shape_type = struct.unpack("<i", content[:4])[0]
        rings = []
        if shape_type == 5:
            part_count, point_count = struct.unpack("<2i", content[36:44])
            parts = np.frombuffer(content, dtype="<i4", count=part_count, offset=44)
            point_offset = 44 + 4 * part_count
            points = np.frombuffer(
                content, dtype="<f8", count=2 * point_count, offset=point_offset
            ).reshape(-1, 2)
            for part_index, start in enumerate(parts):
                end = parts[part_index + 1] if part_index + 1 < part_count else point_count
                if end - start >= 3:
                    rings.append(points[start:end].copy())
        elif shape_type != 0:
            raise ValueError(f"Unsupported Census shape type: {shape_type}.")
        state = state_codes[record_index]
        geometries[state] = StateGeometry(state, rings)
        position = content_end
        record_index += 1
    if record_index != len(state_codes):
        raise ValueError("Census SHP and DBF record counts do not match.")
    return geometries


def albers_equal_area(
    points: np.ndarray,
    lat_1: float,
    lat_2: float,
    lat_0: float,
    lon_0: float,
) -> np.ndarray:
    """Project longitude-latitude points with spherical Albers equal area."""
    longitude = np.deg2rad(points[:, 0])
    latitude = np.deg2rad(points[:, 1])
    first, second, origin_lat, origin_lon = np.deg2rad(
        [lat_1, lat_2, lat_0, lon_0]
    )
    n = 0.5 * (np.sin(first) + np.sin(second))
    constant = np.cos(first) ** 2 + 2 * n * np.sin(first)
    rho = np.sqrt(np.maximum(constant - 2 * n * np.sin(latitude), 0)) / n
    rho_origin = np.sqrt(np.maximum(constant - 2 * n * np.sin(origin_lat), 0)) / n
    theta = n * (longitude - origin_lon)
    return np.column_stack([rho * np.sin(theta), rho_origin - rho * np.cos(theta)])


def add_state_polygons(ax, geometries, values, projection, cmap, norm) -> None:
    all_points = []
    for state, geometry in geometries.items():
        value = values.get(state, np.nan)
        colour = "#E3E3E3" if pd.isna(value) else cmap(norm(float(value)))
        for ring in geometry.rings:
            projected = albers_equal_area(ring, *projection)
            all_points.append(projected)
            ax.add_patch(
                Polygon(
                    projected,
                    closed=True,
                    facecolor=colour,
                    edgecolor="white",
                    linewidth=0.55,
                )
            )
    extent = np.vstack(all_points)
    x_pad = 0.015 * np.ptp(extent[:, 0])
    y_pad = 0.015 * np.ptp(extent[:, 1])
    ax.set_xlim(extent[:, 0].min() - x_pad, extent[:, 0].max() + x_pad)
    ax.set_ylim(extent[:, 1].min() - y_pad, extent[:, 1].max() + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def add_state_labels(ax, states, projection, inset=False) -> None:
    for state in states:
        lon_lat = LABEL_LON_LAT.get(state)
        if lon_lat is None:
            continue
        x, y = albers_equal_area(np.array([lon_lat]), *projection)[0]
        common = {
            "fontsize": 7.2 if not inset else 8.2,
            "fontfamily": "Arial",
            "fontweight": "bold",
            "color": DARK_GRAY,
            "zorder": 10,
        }
        if not inset and state in SMALL_STATE_LABELS:
            annotation = ax.annotate(
                state,
                xy=(x, y),
                xycoords="data",
                xytext=SMALL_STATE_LABELS[state],
                textcoords=ax.transAxes,
                ha="left",
                va="center",
                arrowprops={"arrowstyle": "-", "color": DARK_GRAY, "lw": 0.55},
                annotation_clip=False,
                **common,
            )
        else:
            annotation = ax.annotate(
                state,
                xy=(x, y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
                **common,
            )
        annotation.set_path_effects(
            [path_effects.withStroke(linewidth=1.4, foreground="white")]
        )


def plot_map(fig, geometries, values, title, cmap_name, divergent, y0, title_y):
    excluded = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}
    conus = {state: geo for state, geo in geometries.items() if state not in excluded}
    scale_values = values.dropna().to_numpy(float)
    if divergent:
        bound = float(np.abs(scale_values).max())
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    else:
        norm = Normalize(vmin=float(scale_values.min()), vmax=float(scale_values.max()))
    cmap = plt.get_cmap(cmap_name)

    ax = fig.add_axes([0.025, y0, 0.785, 0.275])
    add_state_polygons(ax, conus, values, CONUS_ALBERS, cmap, norm)
    add_state_labels(ax, conus, CONUS_ALBERS)
    fig.text(0.025, title_y, title, fontsize=14, fontweight="bold", va="bottom")

    for state, projection, box in (
        ("AK", ALASKA_ALBERS, [0.055, y0 + 0.012, 0.105, 0.075]),
        ("HI", HAWAII_ALBERS, [0.168, y0 + 0.012, 0.060, 0.046]),
    ):
        inset_ax = fig.add_axes(box)
        add_state_polygons(
            inset_ax, {state: geometries[state]}, values, projection, cmap, norm
        )
        add_state_labels(inset_ax, [state], projection, inset=True)

    colourbar_ax = fig.add_axes([0.845, y0 + 0.050, 0.019, 0.180])
    colourbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=colourbar_ax
    )
    colourbar.outline.set_linewidth(0.6)
    colourbar.ax.tick_params(labelsize=10, width=0.7)
    bold_ticks(colourbar.ax)


def main() -> None:
    args = output_parser("Rebuild the main three-panel AI-adoption figure.").parse_args()
    set_style()
    assert_current_inputs()

    frame = pd.read_csv(require_file(DERIVED_DIR / "state_frame.csv"))
    quartiles = pd.read_csv(require_file(DERIVED_DIR / "quartile_summary.csv"))
    if len(frame) != 51 or frame["State"].nunique() != 51:
        raise ValueError("state_frame.csv must contain all 51 state-level units.")
    if set(frame["available"]) - set(range(4, 9)):
        raise ValueError("Headline state estimates must average four to eight current waves.")
    if abs(frame["dev"].mean()) > 1e-10:
        raise ValueError("Composition-adjusted map deviations are not mean-centred.")
    if quartiles["n"].sum() != 51 or quartiles["quartile"].tolist() != [1, 2, 3, 4]:
        raise ValueError("quartile_summary.csv is incomplete or incorrectly ordered.")

    geometries = read_state_geometries(
        require_file(RAW_DIR / "cb_2023_us_state_20m.zip")
    )
    missing_geometry = set(frame["State"]) - set(geometries)
    if missing_geometry:
        raise ValueError(f"Missing Census geometries: {sorted(missing_geometry)}")

    fig = plt.figure(figsize=(7.2, 9.4))
    plot_map(
        fig,
        geometries,
        frame.set_index("State")["ai"],
        "(a) Businesses reporting AI use (%)",
        "Blues",
        False,
        0.665,
        0.950,
    )
    plot_map(
        fig,
        geometries,
        frame.set_index("State")["dev"],
        "(b) Deviation after industry and size adjustment (pp)",
        "RdBu_r",
        True,
        0.355,
        0.640,
    )

    ax = fig.add_axes([0.125, 0.070, 0.705, 0.205])
    x = quartiles["quartile"].to_numpy(int)
    y = quartiles["adjusted"].to_numpy(float)
    lower = y - quartiles["adjusted_lo"].to_numpy(float)
    upper = quartiles["adjusted_hi"].to_numpy(float) - y
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        linestyle="none",
        markersize=8.5,
        markerfacecolor=BLUE,
        markeredgecolor="white",
        markeredgewidth=0.9,
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.7,
        capsize=5,
        capthick=1.5,
        zorder=3,
    )
    for x_value, y_value in zip(x, y):
        ax.text(
            x_value,
            y_value + 0.55,
            f"{y_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Q1\nLeast rural", "Q2", "Q3", "Q4\nMost rural"]
    )
    ax.set_ylabel("Businesses reporting AI use (%)")
    ax.set_title(
        "(c) Adjusted mean AI use by rurality quartile (95% CI)",
        loc="left",
        pad=14,
    )
    ax.set_xlim(0.55, 4.45)
    ax.set_ylim(
        float(quartiles["adjusted_lo"].min()) - 0.8,
        float(quartiles["adjusted_hi"].max()) + 1.1,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    bold_ticks(ax)
    save_png(fig, output_file(args.out, "Figure_1.png"))


if __name__ == "__main__":
    main()
