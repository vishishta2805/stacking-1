import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor
)

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("housing 3.csv")

print("\nDataset Loaded Successfully")

# =========================
# HANDLE MISSING VALUES
# =========================

df.dropna(inplace=True)

# =========================
# FEATURES & TARGET
# =========================

X = df.drop(
    "median_house_value",
    axis=1
)

y = df["median_house_value"]

# =========================
# HANDLE CATEGORICAL COLUMN
# =========================

X = pd.get_dummies(
    X,
    columns=['ocean_proximity'],
    drop_first=True
)

# =========================
# CHECK DATA SHAPE
# =========================

print("\nFeature Shape:", X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)

# =========================
# RANDOM FOREST REGRESSOR
# =========================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# =========================
# XGBOOST REGRESSOR
# =========================

xgb_model = XGBRegressor(
    random_state=42,
    objective='reg:squarederror'
)

# =========================
# TRAIN INDIVIDUAL MODELS
# =========================

print("\nTraining Random Forest...")

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest Training Complete")

print("\nTraining XGBoost...")

xgb_model.fit(
    X_train,
    y_train
)

print("XGBoost Training Complete")

# =========================
# INDIVIDUAL PREDICTIONS
# =========================

rf_preds = rf_model.predict(
    X_test
)

xgb_preds = xgb_model.predict(
    X_test
)

# =========================
# INDIVIDUAL R2 SCORES
# =========================

rf_r2 = r2_score(
    y_test,
    rf_preds
)

xgb_r2 = r2_score(
    y_test,
    xgb_preds
)

print(f"\nRandom Forest R2 Score: {rf_r2:.4f}")

print(f"XGBoost R2 Score: {xgb_r2:.4f}")

# =========================
# STACKING REGRESSOR
# =========================

base_models = [

    ('rf', rf_model),

    ('xgb', xgb_model)
]

meta_model = LinearRegression()

stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# =========================
# TRAIN STACKING MODEL
# =========================

print("\nTraining Stacking Regressor...")

stack_model.fit(
    X_train,
    y_train
)

print("Stacking Training Complete")

# =========================
# STACKING PREDICTIONS
# =========================

stack_preds = stack_model.predict(
    X_test
)

# =========================
# EVALUATION METRICS
# =========================

mae = mean_absolute_error(
    y_test,
    stack_preds
)

mse = mean_squared_error(
    y_test,
    stack_preds
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    stack_preds
)

print(f"\nMean Absolute Error: {mae:.2f}")

print(f"Mean Squared Error: {mse:.2f}")

print(f"Root Mean Squared Error: {rmse:.2f}")

print(f"R2 Score: {r2:.4f}")

# =========================
# SAVE MODEL & SCALER
# =========================

joblib.dump(
    stack_model,
    "stacking_regression_model.pkl"
)

joblib.dump(
    scaler,
    "regression_scaler.pkl"
)

print("\nModel Saved Successfully")