import pandas as pd
import calendar

WEATHER_FILE = "usa_state_monthly_weather_2000_2025.csv"
YIELD_FILE = "usda_corn_yield_clean.csv"
OUTPUT_FILE = "final_corn_yield_weather_fixed.csv"

GROWING_MONTHS = [4, 5, 6, 7, 8, 9]


def get_days_in_month(year, month):
    """Get number of days in a given month/year."""
    return calendar.monthrange(year, month)[1]


def main():
    # Load monthly weather
    w = pd.read_csv(WEATHER_FILE)
    w["State"] = w["State"].astype(str).str.strip().str.upper()
    w["Year"] = pd.to_numeric(w["Year"], errors="coerce")
    w["Month"] = pd.to_numeric(w["Month"], errors="coerce")
    w["T2M"] = pd.to_numeric(w["T2M"], errors="coerce")
    w["PRECTOTCORR"] = pd.to_numeric(w["PRECTOTCORR"], errors="coerce")
    
    w = w.dropna(subset=["State", "Year", "Month", "T2M", "PRECTOTCORR"]).copy()
    w["Year"] = w["Year"].astype(int)
    w["Month"] = w["Month"].astype(int)
    
    # Convert PRECTOTCORR (mm/day) to monthly rainfall (mm/month)
    w["monthly_rain_mm"] = w.apply(
        lambda row: row["PRECTOTCORR"] * get_days_in_month(row["Year"], row["Month"]),
        axis=1
    )
    
    # Filter to growing season and aggregate
    w_growing = (
        w[w["Month"].isin(GROWING_MONTHS)]
        .groupby(["State", "Year"], as_index=False)
        .agg(
            avg_temp_growing=("T2M", "mean"),
            total_rain_growing=("monthly_rain_mm", "sum")
        )
    )
    
    # Load yield data
    y = pd.read_csv(YIELD_FILE)
    y["State"] = y["State"].astype(str).str.strip().str.upper()
    y["Year"] = pd.to_numeric(y["Year"], errors="coerce")
    y["corn_yield_bu_acre"] = pd.to_numeric(y["corn_yield_bu_acre"], errors="coerce")
    y = y.dropna(subset=["State", "Year", "corn_yield_bu_acre"]).copy()
    y["Year"] = y["Year"].astype(int)
    
    # Merge
    final = y.merge(w_growing, on=["State", "Year"], how="inner")
    final = final.dropna().drop_duplicates(subset=["State", "Year"]).sort_values(["State", "Year"]).reset_index(drop=True)
    
    # Print statistics
    print("Dataset shape:", final.shape)
    print("States count:", final["State"].nunique())
    print("Year range:", final["Year"].min(), "to", final["Year"].max())
    print("Rainfall min:", final["total_rain_growing"].min(), "mm")
    print("Rainfall max:", final["total_rain_growing"].max(), "mm")
    
    # Save
    final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

