import pandas as pd

# ====== CONFIG (change filename if needed) ======
INPUT_CSV  = "usda_corn_yield_state_year.csv"   # <-- your downloaded USDA file name
OUTPUT_CSV = "usda_corn_yield_clean.csv"

# ====== LOAD ======
df = pd.read_csv(INPUT_CSV)

# ====== KEEP ONLY WHAT WE NEED ======
# Keep: Year, State, Value (yield)
keep_cols = ["Year", "State", "Value"]
missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}. Found columns: {list(df.columns)}")

df = df[keep_cols].copy()

# ====== CLEAN + RENAME ======
df.rename(columns={"Value": "corn_yield_bu_acre"}, inplace=True)

# Convert year to int
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# Clean yield values:
# - remove commas
# - convert to numeric
df["corn_yield_bu_acre"] = (
    df["corn_yield_bu_acre"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["corn_yield_bu_acre"] = pd.to_numeric(df["corn_yield_bu_acre"], errors="coerce")

# Normalize state names (optional but helps)
df["State"] = df["State"].astype(str).str.strip().str.upper()

# Drop rows with missing Year or Yield
df = df.dropna(subset=["Year", "corn_yield_bu_acre"]).copy()

# Convert Year to regular int after dropping NaNs
df["Year"] = df["Year"].astype(int)

# Remove duplicates (just in case)
df = df.drop_duplicates(subset=["Year", "State"])

# Sort for readability
df = df.sort_values(["State", "Year"]).reset_index(drop=True)

# ====== SAVE ======
df.to_csv(OUTPUT_CSV, index=False)

print("✅ Cleaned yield dataset saved:", OUTPUT_CSV)
print("Shape:", df.shape)
print(df.head(10))

