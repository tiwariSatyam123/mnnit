# app.py — HR Attrition Predictor | Streamlit App

import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# ─────────────────────────────────────────
# Auto-Train if Model not found
# ─────────────────────────────────────────
@st.cache_resource
def get_model():
    base      = os.path.dirname(__file__)
    model_dir = os.path.join(base, "Model")
    os.makedirs(model_dir, exist_ok=True)

    model_path   = os.path.join(model_dir, "best_model.pkl")
    encoder_path = os.path.join(model_dir, "encoder.pkl")
    scaler_path  = os.path.join(model_dir, "scaler.pkl")
    feat_path    = os.path.join(model_dir, "feature_columns.pkl")

    # Agar pkl files already hain toh load karo
    if all(os.path.exists(p) for p in [model_path, encoder_path, scaler_path, feat_path]):
        encoder         = joblib.load(encoder_path)
        scaler          = joblib.load(scaler_path)
        feature_columns = joblib.load(feat_path)
        model           = joblib.load(model_path)
        return encoder, scaler, feature_columns, model

    # Nahi hain toh CSV se train karo
    csv_path = os.path.join(base, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)

    cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    label_encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_s, y_train_s = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_s, y_train_s)

    joblib.dump(label_encoders,  encoder_path)
    joblib.dump(scaler,          scaler_path)
    joblib.dump(feature_columns, feat_path)
    joblib.dump(model,           model_path)

    return label_encoders, scaler, feature_columns, model

# ─────────────────────────────────────────
# Load / Train
# ─────────────────────────────────────────
st.title("👥 HR Employee Attrition Predictor")
st.markdown("##### Employee ka data bharo — AI predict karega ki wo company chhod sakta hai ya nahi")
st.divider()

with st.spinner("🔄 Model load ho raha hai... (pehli baar 1-2 min lag sakte hain)"):
    result_artifacts = get_model()

if result_artifacts is None:
    st.error("❌ Dataset file nahi mili! Repo mein `WA_Fn-UseC_-HR-Employee-Attrition.csv` honi chahiye.")
    st.stop()

encoder, scaler, feature_columns, model = result_artifacts

# ─────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────
def predict_attrition(input_dict):
    df = pd.DataFrame([input_dict])
    cols_to_drop = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    for col, le in encoder.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    df = df.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df)

    prediction    = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]

    return {
        "prediction":            "Attrition" if prediction == 1 else "Stay",
        "probability_attrition": round(float(probabilities[1]), 4),
        "probability_stay":      round(float(probabilities[0]), 4)
    }

# ─────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────
with st.form("employee_form"):
    st.subheader("📋 Employee Details")

    c1, c2, c3, c4 = st.columns(4)
    age            = c1.number_input("Age",             min_value=18, max_value=60,  value=30)
    gender         = c2.selectbox("Gender",             ["Male", "Female"])
    marital_status = c3.selectbox("Marital Status",     ["Single", "Married", "Divorced"])
    education      = c4.selectbox("Education Level",    [1,2,3,4,5],
                       format_func=lambda x:{1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])

    c1, c2, c3, c4 = st.columns(4)
    department     = c1.selectbox("Department",         ["Sales","Research & Development","Human Resources"])
    job_role       = c2.selectbox("Job Role",           [
                       "Sales Executive","Research Scientist","Laboratory Technician",
                       "Manufacturing Director","Healthcare Representative","Manager",
                       "Sales Representative","Research Director","Human Resources"])
    job_level      = c3.selectbox("Job Level",          [1,2,3,4,5])
    business_travel= c4.selectbox("Business Travel",    ["Non-Travel","Travel_Rarely","Travel_Frequently"])

    c1, c2, c3, c4 = st.columns(4)
    job_satisfaction   = c1.selectbox("Job Satisfaction",        [1,2,3,4],
                           format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    env_satisfaction   = c2.selectbox("Environment Satisfaction",[1,2,3,4],
                           format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    relationship_sat   = c3.selectbox("Relationship Satisfaction",[1,2,3,4],
                           format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    work_life_balance  = c4.selectbox("Work-Life Balance",       [1,2,3,4],
                           format_func=lambda x:{1:"Bad",2:"Good",3:"Better",4:"Best"}[x])

    c1, c2, c3, c4 = st.columns(4)
    job_involvement    = c1.selectbox("Job Involvement",  [1,2,3,4],
                           format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    performance_rating = c2.selectbox("Performance Rating",[3,4],
                           format_func=lambda x:{3:"Excellent",4:"Outstanding"}[x])
    overtime           = c3.selectbox("OverTime",         ["Yes","No"])
    education_field    = c4.selectbox("Education Field",  [
                           "Life Sciences","Other","Medical","Marketing",
                           "Technical Degree","Human Resources"])

    st.markdown("##### 💰 Financial Details")
    c1, c2, c3, c4 = st.columns(4)
    monthly_income     = c1.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=500)
    daily_rate         = c2.number_input("Daily Rate ($)",     min_value=100,  max_value=1500,  value=800)
    hourly_rate        = c3.number_input("Hourly Rate ($)",    min_value=30,   max_value=100,   value=65)
    monthly_rate       = c4.number_input("Monthly Rate ($)",   min_value=2000, max_value=27000, value=15000, step=500)
    c1, c2, _, _ = st.columns(4)
    percent_salary_hike= c1.number_input("Salary Hike (%)",   min_value=11,   max_value=25,    value=14)
    stock_option_level = c2.selectbox("Stock Option Level",    [0,1,2,3])

    st.markdown("##### 🗂️ Experience Details")
    c1, c2, c3, c4 = st.columns(4)
    total_working_years= c1.number_input("Total Working Years",   min_value=0, max_value=40, value=10)
    years_at_company   = c2.number_input("Years at Company",      min_value=0, max_value=40, value=5)
    years_in_role      = c3.number_input("Years in Current Role", min_value=0, max_value=20, value=3)
    years_since_promo  = c4.number_input("Years Since Promotion", min_value=0, max_value=15, value=1)
    c1, c2, c3, c4 = st.columns(4)
    years_with_manager = c1.number_input("Years With Manager",    min_value=0, max_value=20, value=3)
    num_companies      = c2.number_input("No. of Companies",      min_value=0, max_value=10, value=2)
    training_times     = c3.number_input("Training Times (Last Year)", min_value=0, max_value=6, value=3)
    distance_from_home = c4.number_input("Distance From Home (km)",    min_value=1, max_value=30, value=5)

    st.divider()
    submitted = st.form_submit_button("🔮 Predict Attrition", use_container_width=True, type="primary")

# ─────────────────────────────────────────
# Result
# ─────────────────────────────────────────
if submitted:
    input_data = {
        "Age": age, "BusinessTravel": business_travel, "DailyRate": daily_rate,
        "Department": department, "DistanceFromHome": distance_from_home,
        "Education": education, "EducationField": education_field,
        "EnvironmentSatisfaction": env_satisfaction, "Gender": gender,
        "HourlyRate": hourly_rate, "JobInvolvement": job_involvement,
        "JobLevel": job_level, "JobRole": job_role, "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status, "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate, "NumCompaniesWorked": num_companies,
        "OverTime": overtime, "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_sat,
        "StockOptionLevel": stock_option_level, "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times, "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_role,
        "YearsSinceLastPromotion": years_since_promo, "YearsWithCurrManager": years_with_manager,
    }

    with st.spinner("Predicting..."):
        result = predict_attrition(input_data)

    st.divider()
    st.subheader("📊 Prediction Result")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        if result["prediction"] == "Attrition":
            st.error("⚠️ HIGH RISK — Employee May Leave")
        else:
            st.success("✅ LOW RISK — Employee Likely to Stay")

        attrition_pct = result["probability_attrition"] * 100
        stay_pct      = result["probability_stay"] * 100
        st.metric("Attrition Probability", f"{attrition_pct:.1f}%")
        st.metric("Stay Probability",       f"{stay_pct:.1f}%")

    with col_right:
        st.markdown("**Risk Gauge**")
        st.progress(result["probability_attrition"], text=f"Attrition Risk: {attrition_pct:.1f}%")

        if attrition_pct >= 70:
            st.warning("🔴 **Critical Risk** — HR ko immediately action lena chahiye")
        elif attrition_pct >= 40:
            st.warning("🟡 **Moderate Risk** — Employee ko engage karo")
        else:
            st.info("🟢 **Low Risk** — Employee satisfied lag raha hai")

        st.markdown("---")
        summary_df = pd.DataFrame({
            "Feature": ["Age","Department","Job Role","Monthly Income","OverTime",
                        "Job Satisfaction","Work-Life Balance","Years at Company"],
            "Value":   [age, department, job_role, f"${monthly_income:,}", overtime,
                        {1:"Low",2:"Medium",3:"High",4:"Very High"}[job_satisfaction],
                        {1:"Bad",2:"Good",3:"Better",4:"Best"}[work_life_balance],
                        years_at_company]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

st.divider()
st.caption("Built with ❤️ using Random Forest + SMOTE | IBM HR Analytics Dataset")
