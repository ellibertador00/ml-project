#!/usr/bin/env python3
"""
Merge USDA corn yield (yearly, state-level) with NASA POWER weather (monthly -> growing season yearly).

INPUT FILES (same folder):
  1) usda_corn_yield_clean.csv
     columns: Year,State,corn_yield_bu_acre
  2) usa_state_monthly_weather_2000_2025.csv
     columns: State,Year,Month,T2M,PRECTOTCORR

OUTPUT FILES:
  - usa_state_yearly_weather_growing_2000_2025.csv
  - final_corn_yield_weather.csv

Run:
  python merge_yield_weather.py
"""

import pandas as pd

YIELD_FILE = "usda_corn_yield_clean.csv"
WEATHER_MONTHLY_FILE = "usa_state_monthly_weather_2000_2025.csv"

OUT_WEATHER_GROWING = "usa_state_yearly_weather_growing_2000_2025.csv"
OUT_FINAL = "final_corn_yield_weather.csv"

# Corn growing season (USA typical): Apr–Sep
GROWING_MONTHS = [4, 5, 6, 7, 8, 9]


def main():
    # -----------------------
    # Load yield data
    # -----------------------
    yield_df = pd.read_csv(YIELD_FILE)
    for c in ["Year", "State", "corn_yield_bu_acre"]:
        if c not in yield_df.columns:
            raise ValueError(f"Yield file missing column '{c}'. Found: {list(yield_df.columns)}")

    yield_df["State"] = yield_df["State"].astype(str).str.strip().str.upper()
    yield_df["Year"] = pd.to_numeric(yield_df["Year"], errors="coerce")
    yield_df["corn_yield_bu_acre"] = pd.to_numeric(yield_df["corn_yield_bu_acre"], errors="coerce")
    yield_df = yield_df.dropna(subset=["State", "Year", "corn_yield_bu_acre"]).copy()
    yield_df["Year"] = yield_df["Year"].astype(int)
    yield_df = yield_df.drop_duplicates(subset=["State", "Year"])

    print("=== YIELD DATA ===")
    print("Rows:", len(yield_df))
    print("States:", yield_df["State"].nunique())
    print("Years:", yield_df["Year"].min(), "to", yield_df["Year"].max())
    print(yield_df.head(5).to_string(index=False))
    print()

    # -----------------------
    # Load monthly weather
    # -----------------------
    w = pd.read_csv(WEATHER_MONTHLY_FILE)
    for c in ["State", "Year", "Month", "T2M", "PRECTOTCORR"]:
        if c not in w.columns:
            raise ValueError(f"Weather file missing column '{c}'. Found: {list(w.columns)}")

    w["State"] = w["State"].astype(str).str.strip().str.upper()
    w["Year"] = pd.to_numeric(w["Year"], errors="coerce")
    w["Month"] = pd.to_numeric(w["Month"], errors="coerce")
    w["T2M"] = pd.to_numeric(w["T2M"], errors="coerce")
    w["PRECTOTCORR"] = pd.to_numeric(w["PRECTOTCORR"], errors="coerce")

    w = w.dropna(subset=["State", "Year", "Month", "T2M", "PRECTOTCORR"]).copy()
    w["Year"] = w["Year"].astype(int)
    w["Month"] = w["Month"].astype(int)

    print("=== WEATHER MONTHLY ===")
    print("Rows:", len(w))
    print("States:", w["State"].nunique())
    print("Years:", w["Year"].min(), "to", w["Year"].max())
    print(w.head(5).to_string(index=False))
    print()

    # -----------------------
    # Aggregate to growing season per (State, Year)
    # -----------------------
    w_growing = (
        w[w["Month"].isin(GROWING_MONTHS)]
        .groupby(["State", "Year"], as_index=False)
        .agg(
            avg_temp_growing=("T2M", "mean"),
            total_rain_growing=("PRECTOTCORR", "sum")
        )
    )

    w_growing.to_csv(OUT_WEATHER_GROWING, index=False)

    print("=== WEATHER GROWING-SEASON (YEARLY) ===")
    print("Rows:", len(w_growing))
    print("States:", w_growing["State"].nunique())
    print("Years:", w_growing["Year"].min(), "to", w_growing["Year"].max())
    print(w_growing.head(5).to_string(index=False))
    print(f"✅ Saved: {OUT_WEATHER_GROWING}")
    print()

    # -----------------------
    # Merge yield + weather
    # -----------------------
    final_df = yield_df.merge(w_growing, on=["State", "Year"], how="inner")
    final_df = final_df.dropna().drop_duplicates(subset=["State", "Year"]).sort_values(["State", "Year"]).reset_index(drop=True)
    final_df.to_csv(OUT_FINAL, index=False)

    print("=== FINAL MERGED DATASET ===")
    print("Rows:", len(final_df))
    print("States:", final_df["State"].nunique())
    print("Years:", final_df["Year"].min(), "to", final_df["Year"].max())
    print(final_df.head(10).to_string(index=False))
    print(f"✅ Saved: {OUT_FINAL}")


if __name__ == "__main__":
    main()

