import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="CVD Risk Predictor",
    layout="centered"
)

st.title(" Cardiovascular Disease Risk Predictor")
st.write("Enter patient details to predict overall health risk")

st.divider()

# ==============================
# Load Model
# ==============================
model = joblib.load("cvd_model.pkl")

preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["classifier"]

explainer = shap.TreeExplainer(rf_model)

# ==============================
# USER INPUTS (RAW FEATURES)
# ==============================
General_Health = st.selectbox(
    "General Health",
    ["Excellent", "Very Good", "Good", "Fair", "Poor"]
)

Checkup = st.selectbox(
    "Last Medical Checkup",
    [
        "Within the past year",
        "Within the past 2 years",
        "Within the past 5 years",
        "5 or more years ago",
        "Never"
    ]
)

Exercise = st.selectbox("Exercise", ["Yes", "No"])
Skin_Cancer = st.selectbox("Skin Cancer", ["Yes", "No"])
Other_Cancer = st.selectbox("Other Cancer", ["Yes", "No"])
Depression = st.selectbox("Depression", ["Yes", "No"])
Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
Arthritis = st.selectbox("Arthritis", ["Yes", "No"])
Sex = st.selectbox("Sex", ["Male", "Female"])

Age_Category = st.selectbox(
    "Age Category",
    [
        "18-24","25-29","30-34","35-39","40-44","45-49",
        "50-54","55-59","60-64","65-69","70-74","75-79","80+"
    ]
)

Height = st.number_input("Height (cm)", 100, 220, 170)
Weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
BMI = st.number_input("BMI", 10.0, 50.0, 23.0)

Smoking_History = st.selectbox("Smoking History", ["No", "Former", "Yes"])
Alcohol = st.number_input("Alcohol Consumption (days/month)", 0, 30, 2)
Fruit = st.number_input("Fruit Consumption (days/month)", 0, 30, 5)
Vegetables = st.number_input("Green Vegetables Consumption (days/month)", 0, 30, 5)
Fried = st.number_input("Fried Potato Consumption (days/month)", 0, 30, 2)

# ==============================
# CREATE INPUT DATAFRAME
# ==============================
input_df = pd.DataFrame([{
    "General_Health": General_Health,
    "Checkup": Checkup,
    "Exercise": Exercise,
    "Skin_Cancer": Skin_Cancer,
    "Other_Cancer": Other_Cancer,
    "Depression": Depression,
    "Diabetes": Diabetes,
    "Arthritis": Arthritis,
    "Sex": Sex,
    "Age_Category": Age_Category,
    "Height_(cm)": Height,
    "Weight_(kg)": Weight,
    "BMI": BMI,
    "Smoking_History": Smoking_History,
    "Alcohol_Consumption": Alcohol,
    "Fruit_Consumption": Fruit,
    "Green_Vegetables_Consumption": Vegetables,
    "FriedPotato_Consumption": Fried
}])

# ==============================
# PREDICTION
# ==============================
if st.button("Predict Health Status"):

    prob = model.predict_proba(input_df)[0][1]

    if prob < 0.33:
        result = "🟢 Good"
        color = "green"
    elif prob < 0.66:
        result = "🟡 Average"
        color = "orange"
    else:
        result = "🔴 Bad"
        color = "red"

    st.markdown(f"### **Health Status:** :{color}[{result}]")
    st.write(f"Risk Probability: **{prob:.2f}**")

   # ==============================
# SHAP EXPLANATION (FINAL FIX)
# ==============================
st.subheader("🧠 Why this prediction? (Explainable AI)")

processed_input = preprocessor.transform(input_df)
shap_values = explainer.shap_values(processed_input)

# Handle binary classifier
if isinstance(shap_values, list):
    shap_impact = shap_values[1]
else:
    shap_impact = shap_values

# Take first sample
shap_impact = shap_impact[0]

# Flatten
shap_impact = np.array(shap_impact).flatten()

feature_names = preprocessor.get_feature_names_out()

# ✅ ALIGN LENGTHS (CRITICAL FIX)
min_len = min(len(feature_names), len(shap_impact))

shap_df = pd.DataFrame({
    "Feature": feature_names[:min_len],
    "Impact": shap_impact[:min_len]
})

shap_df["Absolute Impact"] = shap_df["Impact"].abs()
shap_df = shap_df.sort_values("Absolute Impact", ascending=False).head(8)

st.bar_chart(shap_df.set_index("Feature")["Impact"])
