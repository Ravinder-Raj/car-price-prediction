# app.py — complete file
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ■■ PAGE CONFIG ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Must be the FIRST streamlit command in the file
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="",
    layout="wide"
)

# ■■ LOAD MODEL ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# @st.cache_resource loads the model ONCE and keeps it in memory
# Without this, the model reloads every time the user interacts
@st.cache_resource
def load_model():
    model   = joblib.load("models/best_model.joblib")
    columns = joblib.load("models/feature_columns.joblib")
    with open("models/results.json") as f:
        results = json.load(f)
    return model, columns, results

model, feature_cols, results = load_model()

# ■■ HEADER ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.title("Car Price Prediction System")
st.markdown("*Production ML model trained on 205 real car records*")
st.markdown("---")

# ■■ SIDEBAR INPUTS ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.sidebar.header("Enter Car Specs")

horsepower = st.sidebar.slider("Horsepower",   48,  288, 100)
enginesize  = st.sidebar.slider("Engine size",  61,  326, 120)
curbweight  = st.sidebar.slider("Curb weight", 1488, 4066, 2500)
citympg     = st.sidebar.slider("City MPG",    13,   49,   25)
highwaympg  = st.sidebar.slider("Highway MPG", 16,   54,   30)

# ■■ PREDICT ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
if st.sidebar.button("Predict Price", type="primary"):
    # Build input row with same columns model was trained on
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Fill in the values the user provided
    for col, val in [
        ("horsepower", horsepower), ("enginesize", enginesize),
        ("curbweight", curbweight), ("citympg",    citympg),
        ("highwaympg", highwaympg)
    ]:
        if col in input_df.columns:
            input_df[col] = val

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${prediction:,.0f}")

# ■■ MODEL COMPARISON TABLE ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.header("Model Performance")
results_df = pd.DataFrame(results).T
st.dataframe(
    results_df.style.highlight_max(subset=["R2"], color="lightgreen")
)

# ■■ VISUALISATIONS ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
st.header("Data Insights")
df = pd.read_csv("data/Car_data.csv")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price distribution")
    fig, ax = plt.subplots()
    ax.hist(df["price"], bins=30, color="#378ADD", edgecolor="white")
    ax.set_xlabel("Price ($)")
    st.pyplot(fig)

with col2:
    st.subheader("Horsepower vs Price")
    fig, ax = plt.subplots()
    ax.scatter(df["horsepower"], df["price"], alpha=0.6, color="#1D9E75")
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Price ($)")
    st.pyplot(fig)