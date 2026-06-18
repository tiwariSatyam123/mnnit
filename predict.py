# predict.py
# ✅ Trained model se prediction karne ke liye file
#    Sirf is file ko use karo jab kisi employee ka attrition predict karna ho

import pandas as pd
import joblib

# ─────────────────────────────────────────
# Load Saved Artifacts
# ─────────────────────────────────────────
encoder         = joblib.load("Model/encoder.pkl")
scaler          = joblib.load("Model/scaler.pkl")
feature_columns = joblib.load("Model/feature_columns.pkl")
model           = joblib.load("Model/best_model.pkl")


def predict_attrition(input_dict: dict) -> dict:
    """
    Ek employee ka data dict mein do, prediction milega.

    Parameters:
        input_dict (dict): Employee features as key-value pairs

    Returns:
        dict: {
            "prediction": "Attrition" or "Stay",
            "probability_attrition": float (0-1),
            "probability_stay": float (0-1)
        }
    """

    # Step 1: DataFrame banao
    df = pd.DataFrame([input_dict])

    # Step 2: Constant columns drop karo (agar user ne diye toh)
    cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    # Step 3: Categorical columns encode karo (same encoder jo training mein use hua)
    for col, le in encoder.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Step 4: Feature order match karo (bahut important!)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Step 5: Scale karo (same scaler jo training mein use hua)
    df_scaled = scaler.transform(df)

    # Step 6: Predict
    prediction    = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]

    return {
        "prediction":            "Attrition" if prediction == 1 else "Stay",
        "probability_attrition": round(probabilities[1], 4),
        "probability_stay":      round(probabilities[0], 4)
    }


# ─────────────────────────────────────────
# Example Usage
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Ek sample employee ka data (apna data yahan dalo)
    sample_employee = {
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 800,
        "Department": "Research & Development",
        "DistanceFromHome": 5,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 2,
        "OverTime": "No",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 5,
        "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3
    }

    result = predict_attrition(sample_employee)

    print("\n🔮 Prediction Result:")
    print(f"   → Prediction          : {result['prediction']}")
    print(f"   → Attrition Chance    : {result['probability_attrition'] * 100:.1f}%")
    print(f"   → Stay Chance         : {result['probability_stay'] * 100:.1f}%")
