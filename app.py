import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# PAGE CONFIG + LIGHT UI CSS
# =========================
st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

st.markdown("""
<style>
/* tighten spacing */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
/* make metric cards feel nicer */
div[data-testid="stMetric"] {background: rgba(255,255,255,0.03); padding: 10px 12px; border-radius: 14px;}
/* sidebar spacing */
section[data-testid="stSidebar"] .block-container {padding-top: 1.2rem;}
/* subtle header style */
h1, h2, h3 {letter-spacing: 0.2px;}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD ASSETS (LOCAL FILES)
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("final_corn_yield_weather_fixed.csv")


@st.cache_resource
def load_model():
    return joblib.load("random_forest_crop_yield_model.joblib")


df = load_data()
model = load_model()


# =========================
# HEADER
# =========================
st.title("🌾 Weather-Based Crop Yield Prediction")
st.caption("Random Forest Regression • State-level U.S. corn yield forecasting using growing-season temperature and rainfall (local dataset + saved model).")

# quick dataset badges
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("States", f"{df['State'].nunique():,}")
c3.metric("Years", f"{df['Year'].min()}–{df['Year'].max()}")
c4.metric("Target", "bu/acre")


# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.title("Controls")
st.sidebar.caption("Choose inputs and generate a prediction.")

states = sorted(df["State"].unique().tolist())
state = st.sidebar.selectbox("State", states)

year_min, year_max = int(df["Year"].min()), int(df["Year"].max()) + 2
year = st.sidebar.number_input("Year", min_value=year_min, max_value=year_max, value=year_max)

# sensible defaults from selected state's history
state_df = df[df["State"] == state]
default_temp = float(state_df["avg_temp_growing"].mean())
default_rain = float(state_df["total_rain_growing"].mean())

temp_min, temp_max = float(df["avg_temp_growing"].min()), float(df["avg_temp_growing"].max())
rain_min, rain_max = float(df["total_rain_growing"].min()), float(df["total_rain_growing"].max())

avg_temp = st.sidebar.slider(
    "Avg Temp (Growing Season) [°C]",
    temp_min, temp_max, default_temp
)

total_rain = st.sidebar.slider(
    "Total Rain (Growing Season) [mm]",
    rain_min, rain_max, default_rain
)

predict_btn = st.sidebar.button("Predict Yield ✅", use_container_width=True)

with st.sidebar.expander("What do these inputs mean?"):
    st.write("- **Avg Temp (Growing Season):** average temperature during the corn growing months.")
    st.write("- **Total Rain (Growing Season):** total rainfall during the same growing months.")
    st.write("- **State:** used as a categorical feature (encoded).")
    st.write("- **Year:** captures long-term trends and technology improvements.")


# =========================
# MAIN LAYOUT
# =========================
left, right = st.columns([1.05, 1.95], gap="large")


# =========================
# LEFT: PREDICTION CARD + STATE SUMMARY
# =========================
with left:
    st.subheader("🔮 Prediction")

    input_df = pd.DataFrame({
        "State": [state],
        "Year": [int(year)],
        "avg_temp_growing": [float(avg_temp)],
        "total_rain_growing": [float(total_rain)]
    })

    if predict_btn:
        pred = float(model.predict(input_df)[0])
        st.metric("Predicted Corn Yield (bu/acre)", f"{pred:.2f}")
        st.success("Prediction generated successfully.")
    else:
        st.info("Set inputs and click **Predict Yield**.")

    st.write("**Input used**")
    st.dataframe(input_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("📌 State Snapshot")

    # show state historical range for context
    state_hist = df[df["State"] == state].sort_values("Year")
    st.write(f"Historical yield summary for **{state}**:")
    s1, s2, s3 = st.columns(3)
    s1.metric("Mean", f"{state_hist['corn_yield_bu_acre'].mean():.1f}")
    s2.metric("Min", f"{state_hist['corn_yield_bu_acre'].min():.1f}")
    s3.metric("Max", f"{state_hist['corn_yield_bu_acre'].max():.1f}")


# =========================
# RIGHT: PERFORMANCE + ANALYSIS TABS
# =========================
with right:
    st.subheader("📊 Model Results & Analysis")

    # Test set: 2020–2025 (same as training notebook)
    test_df = df[df["Year"] >= 2020].copy()
    X_test = test_df[["State", "Year", "avg_temp_growing", "total_rain_growing"]]
    y_test = test_df["corn_yield_bu_acre"]

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.2f}")
    m2.metric("RMSE", f"{rmse:.2f}")
    m3.metric("R²", f"{r2:.3f}")

    tabs = st.tabs(["Actual vs Predicted", "Residuals", "Feature Importance", "Trends"])

    # ---- Tab 1: Actual vs Predicted
    with tabs[0]:
        fig = plt.figure(figsize=(5.5, 5.2))
        plt.scatter(y_test, y_pred, alpha=0.45)
        mn, mx = float(y_test.min()), float(y_test.max())
        plt.plot([mn, mx], [mn, mx], "r--")
        plt.xlabel("Actual Yield")
        plt.ylabel("Predicted Yield")
        plt.title("Actual vs Predicted (Test Years)")
        st.pyplot(fig, use_container_width=True)

        st.caption("Points closer to the diagonal indicate better prediction accuracy.")

    # ---- Tab 2: Residuals
    with tabs[1]:
        residuals = y_test - y_pred
        fig = plt.figure(figsize=(6.2, 4.2))
        plt.scatter(y_pred, residuals, alpha=0.45)
        plt.axhline(0, color="red")
        plt.xlabel("Predicted Yield")
        plt.ylabel("Residual (Actual − Predicted)")
        plt.title("Residual Plot (Test Years)")
        st.pyplot(fig, use_container_width=True)

        st.caption("Residuals centered around 0 indicate no major systematic bias.")

    # ---- Tab 3: Feature Importance
    with tabs[2]:
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.named_steps["model"].feature_importances_

        fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi = fi.sort_values("Importance", ascending=False).head(15)

        st.write("Top 15 important features:")
        st.dataframe(fi, hide_index=True, use_container_width=True)

        fig = plt.figure(figsize=(7, 4.2))
        plt.barh(fi["Feature"][::-1], fi["Importance"][::-1])
        plt.title("Top 15 Feature Importances")
        st.pyplot(fig, use_container_width=True)

        st.caption("Higher importance means the feature contributes more to the model’s decisions.")

    # ---- Tab 4: Trends
    with tabs[3]:
        st.write("Yield trend for selected state:")
        trend = df[df["State"] == state].sort_values("Year")

        fig = plt.figure(figsize=(10, 3.2))
        plt.plot(trend["Year"], trend["corn_yield_bu_acre"])
        plt.xlabel("Year")
        plt.ylabel("Corn Yield (bu/acre)")
        plt.title(f"{state} - Corn Yield Over Time")
        st.pyplot(fig, use_container_width=True)

        st.caption("This shows historical yield behavior and supports interpretation of predictions.")


# =========================
# FOOTER
# =========================
st.divider()
st.caption("Note: This demo uses a locally saved Random Forest model and a local dataset (no external APIs).")

