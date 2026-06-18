# app.py — HR Attrition Predictor | Streamlit App

import streamlit as st
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide"
)

# ─────────────────────────────────────────
# Load Model Artifacts
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    model_dir = os.path.join(base, "Model")
    encoder         = joblib.load(os.path.join(model_dir, "encoder.pkl"))
    scaler          = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))
    model           = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    return encoder, scaler, feature_columns, model

try:
    encoder, scaler, feature_columns, model = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

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
# UI — Header
# ─────────────────────────────────────────
st.title("👥 HR Employee Attrition Predictor")
st.markdown("##### Employee ka data bharo — AI predict karega ki wo company chhod sakta hai ya nahi")
st.divider()

if not model_loaded:
    st.error(f"❌ Model load nahi hua. Pehle `train_model.py` run karo.\n\nError: {model_error}")
    st.stop()

# ─────────────────────────────────────────
# UI — Input Form
# ─────────────────────────────────────────
with st.form("employee_form"):

    st.subheader("📋 Employee Details")

    # ── Row 1: Basic Info ──────────────────
    c1, c2, c3, c4 = st.columns(4)
    age              = c1.number_input("Age",                    min_value=18, max_value=60,  value=30)
    gender           = c2.selectbox("Gender",                   ["Male", "Female"])
    marital_status   = c3.selectbox("Marital Status",           ["Single", "Married", "Divorced"])
    education        = c4.selectbox("Education Level",          [1, 2, 3, 4, 5],
                                    format_func=lambda x: {1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])

    # ── Row 2: Job Info ────────────────────
    c1, c2, c3, c4 = st.columns(4)
    department       = c1.selectbox("Department",               ["Sales", "Research & Development", "Human Resources"])
    job_role         = c2.selectbox("Job Role",                 [
                                        "Sales Executive", "Research Scientist", "Laboratory Technician",
                                        "Manufacturing Director", "Healthcare Representative", "Manager",
                                        "Sales Representative", "Research Director", "Human Resources"
                                    ])
    job_level        = c3.selectbox("Job Level",                [1, 2, 3, 4, 5])
    business_travel  = c4.selectbox("Business Travel",          ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])

    # ── Row 3: Satisfaction ────────────────
    c1, c2, c3, c4 = st.columns(4)
    job_satisfaction      = c1.selectbox("Job Satisfaction",        [1,2,3,4],
                                         format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    env_satisfaction      = c2.selectbox("Environment Satisfaction",[1,2,3,4],
                                         format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    relationship_sat      = c3.selectbox("Relationship Satisfaction",[1,2,3,4],
                                         format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    work_life_balance     = c4.selectbox("Work-Life Balance",       [1,2,3,4],
                                         format_func=lambda x:{1:"Bad",2:"Good",3:"Better",4:"Best"}[x])

    # ── Row 4: Performance ─────────────────
    c1, c2, c3, c4 = st.columns(4)
    job_involvement       = c1.selectbox("Job Involvement",         [1,2,3,4],
                                         format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    performance_rating    = c2.selectbox("Performance Rating",      [3, 4],
                                         format_func=lambda x:{3:"Excellent",4:"Outstanding"}[x])
    overtime              = c3.selectbox("OverTime",                ["Yes", "No"])
    education_field       = c4.selectbox("Education Field",         [
                                             "Life Sciences", "Other", "Medical", "Marketing",
                                             "Technical Degree", "Human Resources"
                                         ])

    # ── Row 5: Financial ──────────────────
    st.markdown("##### 💰 Financial Details")
    c1, c2, c3, c4 = st.columns(4)
    monthly_income        = c1.number_input("Monthly Income ($)",   min_value=1000, max_value=20000, value=5000, step=500)
    daily_rate            = c2.number_input("Daily Rate ($)",       min_value=100,  max_value=1500,  value=800)
    hourly_rate           = c3.number_input("Hourly Rate ($)",      min_value=30,   max_value=100,   value=65)
    monthly_rate          = c4.number_input("Monthly Rate ($)",     min_value=2000, max_value=27000, value=15000, step=500)
    c1, c2, _, _ = st.columns(4)
    percent_salary_hike   = c1.number_input("Salary Hike (%)",     min_value=11,   max_value=25,    value=14)
    stock_option_level    = c2.selectbox("Stock Option Level",      [0, 1, 2, 3])

    # ── Row 6: Experience ─────────────────
    st.markdown("##### 🗂️ Experience Details")
    c1, c2, c3, c4 = st.columns(4)
    total_working_years   = c1.number_input("Total Working Years",  min_value=0, max_value=40, value=10)
    years_at_company      = c2.number_input("Years at Company",     min_value=0, max_value=40, value=5)
    years_in_role         = c3.number_input("Years in Current Role",min_value=0, max_value=20, value=3)
    years_since_promo     = c4.number_input("Years Since Promotion",min_value=0, max_value=15, value=1)
    c1, c2, c3, c4 = st.columns(4)
    years_with_manager    = c1.number_input("Years With Manager",   min_value=0, max_value=20, value=3)
    num_companies         = c2.number_input("No. of Companies Worked",min_value=0,max_value=10,value=2)
    training_times        = c3.number_input("Training Times (Last Year)", min_value=0, max_value=6, value=3)
    distance_from_home    = c4.number_input("Distance From Home (km)", min_value=1, max_value=30, value=5)

    # ── Submit ─────────────────────────────
    st.divider()
    submitted = st.form_submit_button("🔮 Predict Attrition", use_container_width=True, type="primary")

# ─────────────────────────────────────────
# Prediction & Result
# ─────────────────────────────────────────
if submitted:
    input_data = {
        "Age":                     age,
        "BusinessTravel":          business_travel,
        "DailyRate":               daily_rate,
        "Department":              department,
        "DistanceFromHome":        distance_from_home,
        "Education":               education,
        "EducationField":          education_field,
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender":                  gender,
        "HourlyRate":              hourly_rate,
        "JobInvolvement":          job_involvement,
        "JobLevel":                job_level,
        "JobRole":                 job_role,
        "JobSatisfaction":         job_satisfaction,
        "MaritalStatus":           marital_status,
        "MonthlyIncome":           monthly_income,
        "MonthlyRate":             monthly_rate,
        "NumCompaniesWorked":      num_companies,
        "OverTime":                overtime,
        "PercentSalaryHike":       percent_salary_hike,
        "PerformanceRating":       performance_rating,
        "RelationshipSatisfaction":relationship_sat,
        "StockOptionLevel":        stock_option_level,
        "TotalWorkingYears":       total_working_years,
        "TrainingTimesLastYear":   training_times,
        "WorkLifeBalance":         work_life_balance,
        "YearsAtCompany":          years_at_company,
        "YearsInCurrentRole":      years_in_role,
        "YearsSinceLastPromotion": years_since_promo,
        "YearsWithCurrManager":    years_with_manager,
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
            st.warning("🟡 **Moderate Risk** — Employee ko engage karo, check-in karo")
        else:
            st.info("🟢 **Low Risk** — Employee satisfied lag raha hai")

        st.markdown("---")
        st.markdown("**Input Summary**")
        summary_df = pd.DataFrame({
            "Feature": ["Age", "Department", "Job Role", "Monthly Income", "OverTime",
                        "Job Satisfaction", "Work-Life Balance", "Years at Company"],
            "Value":   [age, department, job_role, f"${monthly_income:,}", overtime,
                        {1:"Low",2:"Medium",3:"High",4:"Very High"}[job_satisfaction],
                        {1:"Bad",2:"Good",3:"Better",4:"Best"}[work_life_balance],
                        years_at_company]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ using Random Forest + SMOTE | IBM HR Analytics Dataset")
