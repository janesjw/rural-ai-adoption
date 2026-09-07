"""Rebuild the current rural-AI analysis inputs from compact official snapshots.

Outputs are the sole source data used by the plotting scripts. The script
contains numerical assertions so stale waves or the former QCEW sector-mapping
error cannot silently re-enter the figures.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
RAW = PACKAGE / "data" / "raw"
BTOS = RAW / "btos"
OUT = PACKAGE / "data" / "derived"
LATEST_PERIOD = "202617"
SECTORS = [
    "11", "21", "22", "23", "31", "42", "44", "48", "51", "52",
    "53", "54", "55", "56", "61", "62", "71", "72", "81",
]
QCEW_TO_BTOS = {"31-33": "31", "44-45": "44", "48-49": "48"}
SIZE_COLUMNS = [
    "n<5", "n5_9", "n10_19", "n20_49", "n50_99", "n100_249",
    "n250_499", "n500_999", "n1000",
]
SIZE_MAP = {
    "A": ["n<5"], "B": ["n5_9"], "C": ["n10_19"],
    "D": ["n20_49"], "E": ["n50_99"], "F": ["n100_249"],
    "G": ["n250_499", "n500_999", "n1000"],
}
STATE_MAP = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


def clean_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).replace(
            {"S": np.nan, "nan": np.nan, "": np.nan, ".": np.nan}
        ),
        errors="coerce",
    )


def response(path: Path, key: str) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_excel(path, sheet_name="Response Estimates")
    frame.columns = [str(column) for column in frame.columns]
    selected = frame[
        frame["Question"].astype(str).str.contains(
            "did this business use Artificial Intelligence", case=False, na=False
        )
        & frame["Answer"].astype(str).str.strip().str.lower().eq("yes")
    ].copy()
    selected[key] = clean_code(selected[key])
    periods = sorted(
        column for column in selected.columns
        if column.isdigit() and len(column) == 6 and numeric(selected[column]).notna().any()
    )
    long = selected.melt(id_vars=[key], value_vars=periods, var_name="period", value_name="value")
    long["value"] = numeric(long["value"])
    return long.dropna(subset=["value"]), periods


def ols(y, x_columns, weights=None):
    y = np.asarray(y, dtype=float)
    x_columns = [np.asarray(column, dtype=float) for column in x_columns]
    design = np.column_stack([np.ones(len(y)), *x_columns])
    n, k = design.shape
    weight = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    inverse = np.linalg.pinv(design.T @ (design * weight[:, None]))
    beta = inverse @ (design.T @ (y * weight))
    residual = y - design @ beta
    score = design * (weight * residual)[:, None]
    variance = inverse @ (score.T @ score) @ inverse * n / (n - k)
    standard_error = np.sqrt(np.diag(variance))
    mean = np.average(y, weights=weight)
    r_squared = 1 - np.sum(weight * residual**2) / np.sum(weight * (y - mean) ** 2)
    return beta, standard_error, r_squared


def clustered_wave_ols(frame: pd.DataFrame, outcome: str):
    wave_dummies = pd.get_dummies(frame["period"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(frame)), frame["r_est"].to_numpy(float), wave_dummies])
    y = frame[outcome].to_numpy(float)
    inverse = np.linalg.pinv(design.T @ design)
    beta = inverse @ design.T @ y
    residual = y - design @ beta
    meat = np.zeros((design.shape[1], design.shape[1]))
    groups = frame["State"].to_numpy()
    unique = np.unique(groups)
    for group in unique:
        score = design[groups == group].T @ residual[groups == group]
        meat += np.outer(score, score)
    n, k, group_count = len(frame), design.shape[1], len(unique)
    correction = (group_count / (group_count - 1)) * ((n - 1) / (n - k))
    variance = inverse @ meat @ inverse * correction
    standard_error = np.sqrt(np.diag(variance))
    return beta[1], standard_error[1], beta[1] / standard_error[1]


def load_btos():
    state, periods = response(BTOS / "btos_state.xlsx", "State")
    sector, _ = response(BTOS / "btos_sector.xlsx", "Sector")
    size, _ = response(BTOS / "btos_size.xlsx", "Empsize")
    msa, _ = response(BTOS / "btos_MSA.xlsx", "MSA")

    api = pd.read_csv(BTOS / "btos_ai_period107.csv", dtype=str)
    current = api[
        api["OPTION_TEXT"].eq("AI current")
        & api["ANSWER"].astype(str).str.strip().str.lower().eq("yes")
    ].copy()
    strata = ["STATE", "NAICS2", "MSA", "NAICS3", "EMPSIZE"]

    api_state = current[
        current["STATE"].notna()
        & current[["NAICS2", "MSA", "NAICS3", "EMPSIZE"]].isna().all(axis=1)
    ][["STATE", "ESTIMATE_PERCENTAGE"]].rename(
        columns={"STATE": "State", "ESTIMATE_PERCENTAGE": "value"}
    )
    api_sector = current[
        current["NAICS2"].notna()
        & current[["STATE", "MSA", "NAICS3", "EMPSIZE"]].isna().all(axis=1)
    ][["NAICS2", "ESTIMATE_PERCENTAGE"]].rename(
        columns={"NAICS2": "Sector", "ESTIMATE_PERCENTAGE": "value"}
    )
    api_size = current[
        current["EMPSIZE"].notna()
        & current[["STATE", "NAICS2", "MSA", "NAICS3"]].isna().all(axis=1)
    ][["EMPSIZE", "ESTIMATE_PERCENTAGE"]].rename(
        columns={"EMPSIZE": "Empsize", "ESTIMATE_PERCENTAGE": "value"}
    )
    for addition, key in ((api_state, "State"), (api_sector, "Sector"), (api_size, "Empsize")):
        addition[key] = clean_code(addition[key])
        addition["value"] = pd.to_numeric(addition["value"], errors="coerce")
        addition["period"] = LATEST_PERIOD

    state = pd.concat([state, api_state], ignore_index=True).dropna(subset=["value"])
    sector = pd.concat([sector, api_sector], ignore_index=True).dropna(subset=["value"])
    size = pd.concat([size, api_size], ignore_index=True).dropna(subset=["value"])
    return state, sector, size, msa, periods + [LATEST_PERIOD], current


def read_cbp() -> pd.DataFrame:
    path = RAW / "cbp23co.zip"
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith((".txt", ".csv")))
        with archive.open(member) as stream:
            return pd.read_csv(
                stream,
                dtype={"fipstate": str, "fipscty": str, "naics": str},
                low_memory=False,
            )


def county_inputs(cbp: pd.DataFrame):
    rucc = pd.read_excel(RAW / "rucc2023.xlsx", dtype=str).rename(
        columns={"FIPS": "fips", "RUCC_2023": "rucc", "Population_2020": "pop"}
    )
    rucc["fips"] = clean_code(rucc["fips"]).str.zfill(5)
    rucc["rucc"] = pd.to_numeric(rucc["rucc"], errors="coerce")
    rucc["pop"] = pd.to_numeric(rucc["pop"].astype(str).str.replace(",", ""), errors="coerce")

    counties = cbp[(cbp["naics"] == "------") & (cbp["fipscty"] != "999")].copy()
    counties["fips"] = counties["fipstate"].str.zfill(2) + counties["fipscty"].str.zfill(3)
    counties["State"] = counties["fipstate"].str.zfill(2).map(STATE_MAP)
    counties["est"] = pd.to_numeric(counties["est"], errors="coerce")
    for column in SIZE_COLUMNS:
        counties[column] = pd.to_numeric(counties[column], errors="coerce").fillna(0)
    counties = counties.dropna(subset=["State"]).merge(
        rucc[["fips", "rucc", "pop"]], on="fips", how="left", validate="one_to_one"
    )
    if counties["rucc"].isna().any():
        raise ValueError("CBP 2023 and RUCC 2023 county geographies do not fully match.")

    rows = []
    for state, group in counties.groupby("State"):
        rows.append({
            "State": state,
            "r_est": 100 * group.loc[group["rucc"] >= 4, "est"].sum() / group["est"].sum(),
            "r_pop": 100 * group.loc[group["rucc"] >= 4, "pop"].sum() / group["pop"].sum(),
            "n_est": group["est"].sum(),
        })
    rural = pd.DataFrame(rows)
    size_weights = counties.groupby("State")[SIZE_COLUMNS].sum()
    size_weights["total"] = size_weights[SIZE_COLUMNS].sum(axis=1)
    return counties, rural, size_weights


def qcew_weights() -> pd.DataFrame:
    frame = pd.read_csv(RAW / "qcew_2025_final_state_sector.csv", dtype={"area_fips": str, "industry_code": str})
    frame["State"] = frame["area_fips"].str.zfill(5).str[:2].map(STATE_MAP)
    frame["Sector"] = frame["industry_code"].str.strip().replace(QCEW_TO_BTOS)
    frame = frame[frame["State"].notna() & frame["Sector"].isin(SECTORS)].copy()
    frame["weight"] = frame["annual_avg_estabs"] / frame.groupby("State")["annual_avg_estabs"].transform("sum")
    published = frame.groupby("State")["Sector"].agg(set)
    missing = {
        (state, sector)
        for state, sectors in published.items()
        for sector in set(SECTORS) - sectors
    }
    # Final QCEW publishes no DC mining record at any aggregation level.
    # That cell has zero weight; every other missing cell remains an error.
    if missing != {("DC", "21")}:
        raise ValueError(
            f"Unexpected missing final-QCEW state-sector cells: {sorted(missing)}"
        )
    return frame[["State", "Sector", "weight"]]


def cbp_industry_weights(cbp: pd.DataFrame) -> pd.DataFrame:
    frame = cbp[(cbp["fipscty"] != "999") & cbp["naics"].str.match(r"^\d{2}----$", na=False)].copy()
    frame["State"] = frame["fipstate"].str.zfill(2).map(STATE_MAP)
    frame["Sector"] = frame["naics"].str[:2]
    frame["est"] = pd.to_numeric(frame["est"], errors="coerce")
    frame = frame[frame["State"].notna() & frame["Sector"].isin(SECTORS)]
    frame = frame.groupby(["State", "Sector"], as_index=False)["est"].sum()
    frame["weight"] = frame["est"] / frame.groupby("State")["est"].transform("sum")
    return frame[["State", "Sector", "weight"]]


def industry_prediction(weights: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    merged = weights.merge(rates[["Sector", "value"]], on="Sector", how="inner")
    return (
        merged.assign(component=merged["weight"] * merged["value"])
        .groupby("State", as_index=False)["component"].sum()
        .rename(columns={"component": "pred_ind"})
    )


def size_prediction(weights: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    rates = rates.set_index("Empsize")["value"]
    prediction = pd.Series(0.0, index=weights.index)
    for code, columns in SIZE_MAP.items():
        prediction += rates.loc[code] * weights[columns].sum(axis=1) / weights["total"]
    return prediction.rename("pred_size").reset_index()


def cross_section(window, state, sector, size, rural, industry_weights, size_weights):
    adoption = (
        state[state["period"].isin(window)]
        .groupby("State", as_index=False)
        .agg(ai=("value", "mean"), available=("period", "nunique"))
    )
    all_wave = state.groupby("State", as_index=False)["value"].mean().rename(columns={"value": "ai_all"})
    sector_rates = sector[sector["period"].isin(window)].groupby("Sector", as_index=False)["value"].mean()
    size_rates = size[size["period"].isin(window)].groupby("Empsize", as_index=False)["value"].mean()
    frame = adoption.merge(all_wave, on="State").merge(rural, on="State")
    frame = frame.merge(industry_prediction(industry_weights, sector_rates), on="State")
    frame = frame.merge(size_prediction(size_weights, size_rates), on="State")
    raw_adjusted = frame["ai"] - frame["pred_ind"] - frame["pred_size"]
    frame["dev"] = raw_adjusted - raw_adjusted.mean()
    frame["adjusted_ai"] = frame["ai"].mean() + frame["dev"]
    frame["resid"] = frame["adjusted_ai"]
    return frame.sort_values("State").reset_index(drop=True), sector_rates, size_rates


def quartile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["quartile"] = pd.qcut(data["r_est"], 4, labels=[1, 2, 3, 4])
    rows = []
    for quartile, group in data.groupby("quartile", observed=True):
        row = {"quartile": int(quartile), "n": len(group), "rural_mean": group["r_est"].mean()}
        for column, prefix in (("ai", "observed"), ("adjusted_ai", "adjusted")):
            mean = group[column].mean()
            se = group[column].std(ddof=1) / np.sqrt(len(group))
            row[prefix] = mean
            row[prefix + "_lo"] = mean - 1.96 * se
            row[prefix + "_hi"] = mean + 1.96 * se
        rows.append(row)
    return pd.DataFrame(rows)


def wave_results(periods, state, sector, size, rural, industry_weights, size_weights):
    rows = []
    pooled = []
    for period in periods:
        observed = state[state["period"] == period][["State", "value"]].rename(columns={"value": "ai"})
        industry_rates = sector[sector["period"] == period][["Sector", "value"]]
        size_rates = size[size["period"] == period][["Empsize", "value"]]
        frame = observed.merge(rural, on="State")
        frame = frame.merge(industry_prediction(industry_weights, industry_rates), on="State")
        frame = frame.merge(size_prediction(size_weights, size_rates), on="State")
        raw_adjusted = frame["ai"] - frame["pred_ind"] - frame["pred_size"]
        frame["adjusted"] = raw_adjusted - raw_adjusted.mean()
        frame["period"] = period
        pooled.append(frame)
        beta, standard_error, _ = ols(frame["ai"], [frame["r_est"]])
        rows.append({
            "period": period,
            "n": len(frame),
            "slope": beta[1],
            "se": standard_error[1],
            "lo": beta[1] - 1.96 * standard_error[1],
            "hi": beta[1] + 1.96 * standard_error[1],
            "level": frame["ai"].mean(),
            "t": beta[1] / standard_error[1],
        })
    return pd.DataFrame(rows), pd.concat(pooled, ignore_index=True)


def title_key(value: str) -> str:
    value = re.sub(r"\s+MSA$", "", str(value).strip())
    if "," not in value:
        return value.lower()
    cities, states = value.rsplit(",", 1)
    # CBSA title vintages can drop peripheral states while retaining the same
    # lead principal city and lead state (for example Chicago and New York).
    lead_state = states.strip().split("-")[0]
    return (cities.split("-")[0].strip() + "|" + lead_state).lower()


def within_state(state, periods, api_current, msa, counties):
    names = msa[["MSA"]].drop_duplicates().copy()
    names["key"] = names["MSA"].map(title_key)
    delineation = pd.read_excel(RAW / "cbsa_delineation_2023.xlsx", header=2, dtype=str)
    delineation.columns = [str(column).strip() for column in delineation.columns]
    delineation = delineation[
        delineation["Metropolitan/Micropolitan Statistical Area"].astype(str).str.contains(
            "Metropolitan", na=False
        )
    ].copy()
    titles = delineation[["CBSA Code", "CBSA Title"]].drop_duplicates().copy()
    titles["key"] = titles["CBSA Title"].map(title_key)
    names = names.merge(titles[["CBSA Code", "key"]], on="key", how="left")
    if names["CBSA Code"].notna().sum() != 25:
        raise ValueError("Could not match all 25 published BTOS metropolitan areas.")
    msa["code"] = msa["MSA"].map(names.set_index("MSA")["CBSA Code"])

    api_msa = api_current[
        api_current["MSA"].notna()
        & api_current[["STATE", "NAICS2", "NAICS3", "EMPSIZE"]].isna().all(axis=1)
    ][["MSA", "ESTIMATE_PERCENTAGE"]].rename(
        columns={"MSA": "code", "ESTIMATE_PERCENTAGE": "value"}
    )
    api_msa["code"] = clean_code(api_msa["code"])
    api_msa["value"] = pd.to_numeric(api_msa["value"], errors="coerce")
    api_msa["period"] = LATEST_PERIOD
    api_msa = api_msa.merge(names[["MSA", "CBSA Code"]], left_on="code", right_on="CBSA Code")
    combined = pd.concat(
        [msa[["MSA", "code", "period", "value"]], api_msa[["MSA", "code", "period", "value"]]],
        ignore_index=True,
    )
    combined = combined[combined["period"].isin(periods[-8:])].dropna(subset=["value"])
    means = combined.groupby(["MSA", "code"], as_index=False)["value"].mean().rename(
        columns={"value": "ai_msa"}
    )
    means["states"] = means["MSA"].apply(
        lambda value: re.search(r",\s*([A-Z\-]+)\s+MSA", value).group(1).split("-")
        if re.search(r",\s*([A-Z\-]+)\s+MSA", value) else []
    )
    single = means[means["states"].map(len).eq(1)].copy()
    single["State"] = single["states"].map(lambda value: value[0])

    delineation = delineation.dropna(subset=["FIPS State Code", "FIPS County Code"])
    delineation["fips"] = (
        clean_code(delineation["FIPS State Code"]).str.zfill(2)
        + clean_code(delineation["FIPS County Code"]).str.zfill(3)
    )
    county = counties.merge(delineation[["fips", "CBSA Code"]], on="fips", how="left")
    state_mean = state[state["period"].isin(periods[-8:])].groupby("State")["value"].mean()
    rows = []
    for state_code, group in single.groupby("State"):
        total = county[county["State"] == state_code]
        metro_codes = set(group["code"])
        in_metro = total[total["CBSA Code"].isin(metro_codes)]
        n_state = total["est"].sum()
        n_metro = in_metro["est"].sum()
        if n_metro <= 0 or n_state <= n_metro or state_code not in state_mean.index:
            continue
        weights = [total.loc[total["CBSA Code"] == code, "est"].sum() for code in group["code"]]
        ai_metro = np.average(group["ai_msa"], weights=weights)
        ai_rest = (state_mean.loc[state_code] * n_state - ai_metro * n_metro) / (n_state - n_metro)
        rest = total[~total["CBSA Code"].isin(metro_codes)]
        rural_rest = 100 * rest.loc[rest["rucc"] >= 4, "est"].sum() / rest["est"].sum()
        rows.append({
            "State": state_code,
            "ai_msa": ai_metro,
            "ai_rest": ai_rest,
            "gap": ai_metro - ai_rest,
            "rural_rest": rural_rest,
            "covered_metros": len(group),
        })
    return pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)


def main() -> None:
    state, sector, size, msa, periods, api_current = load_btos()
    if len(periods) != 20 or periods[-1] != LATEST_PERIOD:
        raise ValueError(f"Expected 20 waves through {LATEST_PERIOD}; found {periods}.")
    cbp = read_cbp()
    counties, rural, size_weights = county_inputs(cbp)
    qcew = qcew_weights()
    frame, sector_rates, size_rates = cross_section(
        periods[-8:], state, sector, size, rural, qcew, size_weights
    )
    cbp_frame, _, _ = cross_section(
        periods[-8:], state, sector, size, rural, cbp_industry_weights(cbp), size_weights
    )
    if len(frame) != 51 or frame["State"].nunique() != 51:
        raise ValueError("The latest cross-section must contain 50 states plus DC.")

    raw, raw_se, raw_r2 = ols(frame["ai"], [frame["r_est"]])
    industry, _, _ = ols(frame["pred_ind"], [frame["r_est"]])
    size_beta, _, _ = ols(frame["pred_size"], [frame["r_est"]])
    adjusted, adjusted_se, adjusted_r2 = ols(frame["adjusted_ai"], [frame["r_est"]])
    cbp_adjusted, cbp_adjusted_se, _ = ols(cbp_frame["adjusted_ai"], [cbp_frame["r_est"]])
    quartiles = quartile_summary(frame)
    waves, pooled = wave_results(periods, state, sector, size, rural, qcew, size_weights)
    pooled_raw = clustered_wave_ols(pooled, "ai")
    pooled_adjusted = clustered_wave_ols(pooled, "adjusted")
    within = within_state(state, periods, api_current, msa, counties)

    balanced = frame[frame["available"] == 8]
    balanced_raw, balanced_raw_se, _ = ols(balanced["ai"], [balanced["r_est"]])
    balanced_adjusted, balanced_adjusted_se, _ = ols(balanced["adjusted_ai"], [balanced["r_est"]])

    broadband = pd.read_csv(RAW / "chr2026_state_broadband.csv")
    education = pd.read_csv(RAW / "acs2024_state_ba.csv")
    complements = frame.merge(broadband, on="State").merge(education, on="State")
    comp_beta, comp_se, comp_r2 = ols(
        complements["ai"],
        [complements["r_est"], complements["broadband_share"], complements["pct_ba"]],
    )
    complements.to_csv(OUT / "complements_51.csv", index=False)

    composition = pd.DataFrame([
        {"measure": "Observed", "slope": raw[1], "share_of_raw": 1.0},
        {"measure": "Industry-predicted", "slope": industry[1], "share_of_raw": industry[1] / raw[1]},
        {"measure": "Size-predicted", "slope": size_beta[1], "share_of_raw": size_beta[1] / raw[1]},
        {"measure": "After both adjustments", "slope": adjusted[1], "share_of_raw": adjusted[1] / raw[1]},
    ])

    frame.to_csv(OUT / "state_frame.csv", index=False)
    quartiles.to_csv(OUT / "quartile_summary.csv", index=False)
    waves.to_csv(OUT / "gradient_by_wave.csv", index=False)
    within.to_csv(OUT / "btos_within_state.csv", index=False)
    composition.to_csv(OUT / "composition_summary.csv", index=False)

    results = {
        "data": {
            "waves": periods,
            "latest_window": periods[-8:],
            "units": len(frame),
            "balanced_units": len(balanced),
        },
        "cross_section": {
            "raw_slope": raw[1], "raw_se_hc1": raw_se[1], "raw_t": raw[1] / raw_se[1], "raw_r2": raw_r2,
            "industry_slope": industry[1], "industry_share": industry[1] / raw[1],
            "size_slope": size_beta[1], "size_share": size_beta[1] / raw[1],
            "adjusted_slope": adjusted[1], "adjusted_se_hc1": adjusted_se[1],
            "adjusted_t": adjusted[1] / adjusted_se[1], "adjusted_r2": adjusted_r2,
            "adjusted_share": adjusted[1] / raw[1],
            "cbp_adjusted_slope": cbp_adjusted[1],
            "cbp_adjusted_t": cbp_adjusted[1] / cbp_adjusted_se[1],
            "cbp_adjusted_share": cbp_adjusted[1] / raw[1],
        },
        "balanced_eight": {
            "raw_slope": balanced_raw[1], "raw_t": balanced_raw[1] / balanced_raw_se[1],
            "adjusted_slope": balanced_adjusted[1],
            "adjusted_t": balanced_adjusted[1] / balanced_adjusted_se[1],
        },
        "pooled_twenty": {
            "n": len(pooled),
            "raw_slope": pooled_raw[0], "raw_cluster_se": pooled_raw[1], "raw_t": pooled_raw[2],
            "adjusted_slope": pooled_adjusted[0], "adjusted_cluster_se": pooled_adjusted[1],
            "adjusted_t": pooled_adjusted[2],
            "negative_waves": int((waves["slope"] < 0).sum()),
            "waves_t_below_minus_two": int((waves["t"] < -2).sum()),
        },
        "within_state": {
            "states": len(within),
            "metro_mean": within["ai_msa"].mean(),
            "rest_mean": within["ai_rest"].mean(),
            "gap_mean": within["gap"].mean(),
            "positive": int((within["gap"] > 0).sum()),
            "paired_t": within["gap"].mean() / (within["gap"].std(ddof=1) / np.sqrt(len(within))),
        },
        "complements": {
            "r2": comp_r2,
            "rural_slope": comp_beta[1], "rural_t": comp_beta[1] / comp_se[1],
            "broadband_coefficient": comp_beta[2], "broadband_t": comp_beta[2] / comp_se[2],
            "broadband_sd": complements["broadband_share"].std(ddof=1),
            "broadband_one_sd_effect": comp_beta[2] * complements["broadband_share"].std(ddof=1),
            "ba_coefficient": comp_beta[3], "ba_t": comp_beta[3] / comp_se[3],
        },
        "sector_rates": dict(zip(sector_rates["Sector"], sector_rates["value"])),
        "size_rates": dict(zip(size_rates["Empsize"], size_rates["value"])),
    }
    (OUT / "latest_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    expected = {
        "raw": -0.097680,
        "adjusted": -0.074044,
        "pooled_raw": -0.084271,
        "within_gap": 4.3958,
    }
    checks = {
        "raw": raw[1],
        "adjusted": adjusted[1],
        "pooled_raw": pooled_raw[0],
        "within_gap": within["gap"].mean(),
    }
    for key, target in expected.items():
        if not np.isclose(checks[key], target, atol=5e-5):
            raise AssertionError(f"{key}: expected approximately {target}, obtained {checks[key]}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
