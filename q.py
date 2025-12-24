import pandas as pd

df = pd.read_csv("final_corn_yield_weather.csv")

print("Shape:", df.shape)
print("States:", df["State"].nunique())
print("Years:", df["Year"].min(), "to", df["Year"].max())

print("\nDescribe numeric columns:")
print(df[["corn_yield_bu_acre","avg_temp_growing","total_rain_growing"]].describe())

print("\nMissing values:")
print(df.isna().sum())
