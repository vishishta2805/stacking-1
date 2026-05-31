import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

from sklearn.metrics import accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("breast-cancer.csv")

print("\nDataset Loaded Successfully")

# =========================
# DROP UNWANTED COLUMN
# =========================

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# =========================
# ENCODE TARGET COLUMN
# =========================

label_encoder = LabelEncoder()

# M = Malignant
# B = Benign

df['diagnosis'] = label_encoder.fit_transform(
    df['diagnosis']
)

# =========================
# SELECT FEATURES
# =========================

selected_features = [

    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean'
]

X = df[selected_features]

y = df['diagnosis']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =========================
# RANDOM FOREST MODEL
# =========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =========================
# XGBOOST MODEL
# =========================

xgb_model = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

# =========================
# TRAIN INDIVIDUAL MODELS
# =========================

print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

print("Random Forest Training Complete")

print("\nTraining XGBoost...")

xgb_model.fit(X_train, y_train)

print("XGBoost Training Complete")

# =========================
# INDIVIDUAL PREDICTIONS
# =========================

rf_preds = rf_model.predict(X_test)

xgb_preds = xgb_model.predict(X_test)

# =========================
# INDIVIDUAL ACCURACIES
# =========================

rf_acc = accuracy_score(y_test, rf_preds)

xgb_acc = accuracy_score(y_test, xgb_preds)

print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")

print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# =========================
# STACKING CLASSIFIER
# =========================

base_models = [

    ('rf', rf_model),

    ('xgb', xgb_model)
]

meta_model = LogisticRegression()

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# =========================
# TRAIN STACKING MODEL
# =========================

print("\nTraining Stacking Classifier...")

stack_model.fit(X_train, y_train)

print("Stacking Training Complete")

# =========================
# STACKING PREDICTIONS
# =========================

stack_preds = stack_model.predict(X_test)

# =========================
# STACKING ACCURACY
# =========================

stack_acc = accuracy_score(
    y_test,
    stack_preds
)

print(f"\nStacking Accuracy: {stack_acc:.4f}")

# =========================
# SAVE MODEL & SCALER
# =========================

joblib.dump(
    stack_model,
    "stacking_model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nModel Saved Successfully")