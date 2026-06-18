# train_model.py
# ✅ Single file jo sab kuch handle karta hai:
#    encoder, scaler, feature columns, aur model training

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv("Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ─────────────────────────────────────────
# 2. Drop Useless Constant Columns
#    (ye columns HR dataset mein always same hoti hain)
# ─────────────────────────────────────────
cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# ─────────────────────────────────────────
# 3. Encode Target Variable
# ─────────────────────────────────────────
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# ─────────────────────────────────────────
# 4. Separate Features and Target
# ─────────────────────────────────────────
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# ─────────────────────────────────────────
# 5. Encode Categorical Columns
# ─────────────────────────────────────────
label_encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Save encoder
joblib.dump(label_encoders, "Model/encoder.pkl")
print("✅ encoder.pkl saved!")

# ─────────────────────────────────────────
# 6. Save Feature Column Order
#    (inference ke time same order chahiye)
# ─────────────────────────────────────────
joblib.dump(X.columns.tolist(), "Model/feature_columns.pkl")
print("✅ feature_columns.pkl saved!")

# ─────────────────────────────────────────
# 7. Train-Test Split (stratify = balanced split)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ─────────────────────────────────────────
# 8. Scaling (ONLY fit on train, transform on both)
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)        # ⚠️ fit nahi, sirf transform

joblib.dump(scaler, "Model/scaler.pkl")
print("✅ scaler.pkl saved!")

# ─────────────────────────────────────────
# 9. Handle Class Imbalance with SMOTE
#    (ONLY on training data — test data touch nahi hoga)
# ─────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"✅ SMOTE done | Train size: {X_train_smote.shape[0]} samples")

# ─────────────────────────────────────────
# 10. Hyperparameter Tuning with GridSearchCV
#     (F1 score — imbalanced data ke liye best metric)
# ─────────────────────────────────────────
param_grid = {
    "n_estimators":     [100, 200],
    "max_depth":        [10, 20, None],
    "min_samples_split":[2, 5],
    "min_samples_leaf": [1, 2]
}

rf = RandomForestClassifier(class_weight="balanced", random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_smote, y_train_smote)

best_model = grid_search.best_estimator_
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# ─────────────────────────────────────────
# 11. Evaluation on Test Set
# ─────────────────────────────────────────
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy : {round(accuracy, 4)}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stay", "Attrition"]))

# ─────────────────────────────────────────
# 12. Save Best Model
# ─────────────────────────────────────────
joblib.dump(best_model, "Model/best_model.pkl")
print("\n✅ best_model.pkl saved successfully!")
print("\n🎯 Training Complete! Saved files:")
print("   Model/encoder.pkl")
print("   Model/scaler.pkl")
print("   Model/feature_columns.pkl")
print("   Model/best_model.pkl")
