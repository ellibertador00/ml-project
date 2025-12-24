��� Weather-Based Crop Yield Prediction
Master Project Checklist
✅ Phase 1: Problem Definition & Planning

 Define the problem clearly:

Predict annual corn yield (bu/acre) using growing-season weather data

 Identify stakeholders:

Farmers, planners, researchers

 Define scope:

USA, state-level analysis

Years: 2000–2025

Crop: Corn

 State assumptions:

State centroid represents regional weather

Growing season: April–September

No soil or NDVI data in current version

 List limitations:

State-level aggregation

No field-level variability

Weather-only features

✅ Phase 2: Dataset Collection

 Collect crop yield data:

Source: USDA NASS QuickStats

Level: State × Year

Variable: Corn yield (bu/acre)

 Collect weather data:

Source: NASA POWER

Variables:

Average temperature (T2M)

Precipitation (PRECTOTCORR)

Resolution: Monthly

 Justify data sources:

Government / satellite-based

Public, reproducible, reliable

✅ Phase 3: Data Preprocessing

 Clean USDA yield data:

Remove non-numeric values

Standardize state names

Remove duplicates

 Clean NASA weather data:

Ensure numeric columns

Validate month/year ranges

 Convert rainfall units:

PRECTOTCORR (mm/day) → monthly rainfall (mm)

Multiply by number of days per month

 Aggregate growing-season features:

Avg temperature (Apr–Sep)

Total rainfall (Apr–Sep)

 Merge datasets:

Join on (State, Year)

 Save final dataset:

final_corn_yield_weather_fixed.csv

✅ Phase 4: Dataset Validation & EDA

 Verify dataset shape:

Rows ≈ 1200+

States ≈ 48

Years: 2000–2025

 Check missing values (none allowed)

 Generate EDA plots:

Yield trend over years

Yield distribution histogram

Rainfall vs yield scatter plot

Temperature vs yield scatter plot

Correlation heatmap

Boxplot of yield by state

 Summarize key insights from EDA

✅ Phase 5: Baseline Modeling

 Implement baseline model:

Predict mean yield

 Evaluate baseline:

MAE

RMSE

R²

 (Optional) State-wise mean baseline

 Interpret baseline results:

Demonstrate need for ML models

✅ Phase 6: Machine Learning Models

 Split dataset:

Train/Test (80/20) or year-based split

 Feature encoding:

One-hot encode State

 Train models:

Linear Regression

Ridge Regression

Random Forest

 Evaluate each model:

MAE

RMSE

R²

 Compare results in a table

 Select best-performing model

✅ Phase 7: Error Reduction Techniques

 Improve feature engineering:

Correct rainfall aggregation

 Hyperparameter tuning:

Random Forest depth, estimators

 Prevent data leakage:

Year-aware split

 Analyze residuals:

Error patterns

 Justify chosen techniques in report

✅ Phase 8: Model Interpretation

 Feature importance plot

 Predicted vs actual scatter plot

 Residual error plot

 Explain model behavior in plain language

✅ Phase 9: Demonstration (Streamlit App)

 Create Streamlit UI:

Dataset overview

EDA visualization section

Model comparison section

 Prediction interface:

Select State

Select Year

Input temperature & rainfall

Display predicted yield

 Add explanation text:

Model assumptions

Limitations

 Ensure app runs locally

✅ Phase 10: Documentation & Submission

 Create README.md:

Project overview

Data sources

Methodology

How to run code

 Add screenshots:

EDA plots

Streamlit demo

 Add future work section:

NDVI integration

Soil data

Higher-resolution modeling

 Ensure reproducibility:

requirements.txt

Clear folder structure

 Final review against course guidelines
