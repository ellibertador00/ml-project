import pandas as pd
import requests
import time

# ==============================
# CONFIG
# ==============================
START_YEAR = 2000
END_YEAR   = 2025

# NASA POWER monthly endpoint
POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Variables:
# T2M = Temperature at 2m (°C)
# PRECTOTCORR = Corrected precipitation (mm)
PARAMS = "T2M,PRECTOTCORR"

# Growing season months (corn typical): Apr-Sep
GROWING_MONTHS = [4,5,6,7,8,9]

# Polite delay between requests
SLEEP_SEC = 0.25

# ==============================
# STATE CENTROID COORDS (lat, lon)
# Use only the states that exist in your yield file
# ==============================
STATE_COORDS = {
    "ALABAMA": (32.806671, -86.791130),
    "ALASKA": (61.370716, -152.404419),
    "ARIZONA": (33.729759, -111.431221),
    "ARKANSAS": (34.969704, -92.373123),
    "CALIFORNIA": (36.116203, -119.681564),
    "COLORADO": (39.059811, -105.311104),
    "CONNECTICUT": (41.597782, -72.755371),
    "DELAWARE": (39.318523, -75.507141),
    "FLORIDA": (27.766279, -81.686783),
    "GEORGIA": (33.040619, -83.643074),
    "HAWAII": (21.094318, -157.498337),
    "IDAHO": (44.240459, -114.478828),
    "ILLINOIS": (40.349457, -88.986137),
    "INDIANA": (39.849426, -86.258278),
    "IOWA": (42.011539, -93.210526),
    "KANSAS": (38.526600, -96.726486),
    "KENTUCKY": (37.668140, -84.670067),
    "LOUISIANA": (31.169546, -91.867805),
    "MAINE": (44.693947, -69.381927),
    "MARYLAND": (39.063946, -76.802101),
    "MASSACHUSETTS": (42.230171, -71.530106),
    "MICHIGAN": (43.326618, -84.536095),
    "MINNESOTA": (45.694454, -93.900192),
    "MISSISSIPPI": (32.741646, -89.678696),
    "MISSOURI": (38.456085, -92.288368),
    "MONTANA": (46.921925, -110.454353),
    "NEBRASKA": (41.125370, -98.268082),
    "NEVADA": (38.313515, -117.055374),
    "NEW HAMPSHIRE": (43.452492, -71.563896),
    "NEW JERSEY": (40.298904, -74.521011),
    "NEW MEXICO": (34.840515, -106.248482),
    "NEW YORK": (42.165726, -74.948051),
    "NORTH CAROLINA": (35.630066, -79.806419),
    "NORTH DAKOTA": (47.528912, -99.784012),
    "OHIO": (40.388783, -82.764915),
    "OKLAHOMA": (35.565342, -96.928917),
    "OREGON": (44.572021, -122.070938),
    "PENNSYLVANIA": (40.590752, -77.209755),
    "RHODE ISLAND": (41.680893, -71.511780),
    "SOUTH CAROLINA": (33.856892, -80.945007),
    "SOUTH DAKOTA": (44.299782, -99.438828),
    "TENNESSEE": (35.747845, -86.692345),
    "TEXAS": (31.054487, -97.563461),
    "UTAH": (40.150032, -111.862434),
    "VERMONT": (44.045876, -72.710686),
    "VIRGINIA": (37.769337, -78.169968),
    "WASHINGTON": (47.400902, -121.490494),
    "WEST VIRGINIA": (38.491226, -80.954453),
    "WISCONSIN": (44.268543, -89.616508),
    "WYOMING": (42.755966, -107.302490),
}

# ==============================
# OPTIONAL: Load your yield file to automatically select states
# (If you don't have it loaded yet, you can manually set states_to_use)
# ==============================
YIELD_FILE = "usda_corn_yield_clean.csv"  # change if needed

try:
    y = pd.read_csv(YIELD_FILE)
    y["State"] = y["State"].astype(str).str.strip().str.upper()
    states_to_use = sorted(set(y["State"]) & set(STATE_COORDS.keys()))
    missing = sorted(set(y["State"]) - set(STATE_COORDS.keys()))
    if missing:
        print("⚠️ Missing coords for these states (will be skipped):", missing[:10], "..." if len(missing)>10 else "")
    print(f"Using {len(states_to_use)} states from yield file.")
except Exception as e:
    print("Could not read yield file; using ALL states in STATE_COORDS. Error:", e)
    states_to_use = sorted(STATE_COORDS.keys())

# ==============================
# Fetch NASA POWER monthly weather
# ==============================
def fetch_power_monthly(lat, lon, start_year, end_year):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start_year,
        "end": end_year,
        "community": "AG",
        "parameters": PARAMS,
        "format": "JSON",
    }
    r = requests.get(POWER_URL, params=params, timeout=40)
    r.raise_for_status()
    return r.json()["properties"]["parameter"]

rows = []
failed = []

for i, st in enumerate(states_to_use, start=1):
    lat, lon = STATE_COORDS[st]
    try:
        data = fetch_power_monthly(lat, lon, START_YEAR, END_YEAR)
        t2m = data["T2M"]
        pr  = data["PRECTOTCORR"]

        for year in range(START_YEAR, END_YEAR + 1):
            for month in range(1, 13):
                key = f"{year}{month:02d}"
                rows.append({
                    "State": st,
                    "Year": year,
                    "Month": month,
                    "T2M": t2m.get(key),
                    "PRECTOTCORR": pr.get(key),
                })

        print(f"[{i}/{len(states_to_use)}] OK {st}")
        time.sleep(SLEEP_SEC)

    except Exception as e:
        print(f"[{i}/{len(states_to_use)}] FAIL {st}: {e}")
        failed.append((st, str(e)))

weather_df = pd.DataFrame(rows)

# Clean numeric
weather_df["T2M"] = pd.to_numeric(weather_df["T2M"], errors="coerce")
weather_df["PRECTOTCORR"] = pd.to_numeric(weather_df["PRECTOTCORR"], errors="coerce")

# Save monthly
weather_df.to_csv("usa_state_monthly_weather_2000_2025.csv", index=False)

print("\nMonthly weather shape:", weather_df.shape)
print("Failed states:", failed[:5], "..." if len(failed)>5 else "")

# ==============================
# Growing season aggregation (Apr-Sep)
# ==============================
weather_growing = (
    weather_df[weather_df["Month"].isin(GROWING_MONTHS)]
    .groupby(["State", "Year"], as_index=False)
    .agg(
        avg_temp_growing=("T2M", "mean"),
        total_rain_growing=("PRECTOTCORR", "sum"),
    )
)

weather_growing.to_csv("usa_state_yearly_weather_growing_2000_2025.csv", index=False)

print("Growing-season yearly weather shape:", weather_growing.shape)
weather_growing.head()

