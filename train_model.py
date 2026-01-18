import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:\\Users\\naras\\OneDrive\\Desktop\\CVD_Project\\CVD_cleaned.csv")
print("Dataset loaded:", df.shape)


possible_targets = ['cardio', 'target', 'outcome', 'y', 'heart_disease'] # Added 'heart_disease'
target_col = None

for col in df.columns:
    if col.lower() in possible_targets:
        target_col = col
        break

# If no common target names are found, default to 'Heart_Disease' if it exists, 
# otherwise fallback to the last column as a last resort.
if target_col is None:
    if 'Heart_Disease' in df.columns:
        target_col = 'Heart_Disease'
    else:
        target_col = df.columns[-1]

print("Target column:", target_col)

X = df.drop(columns=[target_col])
y = df[target_col]



categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

print("Categorical columns:", list(categorical_cols))
print("Numerical columns:", list(numerical_cols))


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",  # ⭐ IMPORTANT
        random_state=42,
        n_jobs=-1
    ))
])

df_cleaned = pd.concat([X, y], axis=1).dropna(subset=[target_col])
X_cleaned = df_cleaned.drop(columns=[target_col])
y_cleaned = df_cleaned[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y_cleaned, test_size=0.2, stratify=y_cleaned, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "cvd_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Model trained and saved successfully")