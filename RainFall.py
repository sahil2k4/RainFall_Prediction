import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

# Install xgboost once (on first run)
os.system("pip install xgboost")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🌧️ Indian Rainfall Data Analysis & Prediction")

# Load data function
def load_data():
    paths = ['./rainfaLLIndia.csv', '../DataSets/rainfaLLIndia.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("❌ Data file not found. Please upload 'rainfaLLIndia.csv' or place it in the working directory.")
    return None

df = load_data()
if df is None:
    st.stop()

# Basic cleaning and feature engineering
df.rename(columns={'subdivision': 'Sub_Division'}, inplace=True)
df["AVG_RAINFALL"] = df[["JUN", "JUL", "AUG", "SEP"]].mean(axis=1)
df.sort_values(by=["Sub_Division", "YEAR"], inplace=True)
df["YOY_CHANGE"] = df.groupby("Sub_Division")["JUN-SEP"].diff()
df["LAG_1"] = df.groupby("Sub_Division")["JUN-SEP"].shift(1)
df["LAG_2"] = df.groupby("Sub_Division")["JUN-SEP"].shift(2)

# Drop NaNs from engineered columns for model training
df.dropna(subset=["AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Encode subdivisions
label_enc = LabelEncoder()
df['Sub_Division_Encoded'] = label_enc.fit_transform(df['Sub_Division'])

# Show sample data
st.subheader("🔍 Sample of Rainfall Data")
st.dataframe(df.head())

# Prepare data for model
features = ["YEAR", "AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2", "Sub_Division_Encoded"]
target = "JUN-SEP"
X = df[features]
y = df[target]

# Train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("🧠 Model Performance on Test Data")
st.write(f"R² Score: {r2:.4f}")
st.write(f"MAE: {mae:.2f} mm")
st.write(f"RMSE: {rmse:.2f} mm")

# Prediction input
st.subheader("🎯 Predict JUN-SEP Rainfall for Future Year")
subdivision_input = st.selectbox("Select Subdivision", sorted(df["Sub_Division"].unique()))
future_year_input = st.slider("Select Future Year", 2023, 2040, 2025)

if st.button("Predict"):
    sub_df = df[df["Sub_Division"] == subdivision_input].sort_values("YEAR")
    if sub_df.empty:
        st.error("Subdivision data not found.")
    else:
        last_rec = sub_df.iloc[-1]
        input_dict = {
            "YEAR": future_year_input,
            "AVG_RAINFALL": last_rec["AVG_RAINFALL"],
            "YOY_CHANGE": last_rec["YOY_CHANGE"],
            "LAG_1": last_rec["JUN-SEP"],
            "LAG_2": last_rec["LAG_1"],
            "Sub_Division_Encoded": label_enc.transform([subdivision_input])[0]
        }
        input_df = pd.DataFrame([input_dict])
        prediction = rf_model.predict(input_df)[0]
        st.success(f"🌧️ Predicted JUN-SEP Rainfall for {subdivision_input} in {future_year_input}: {prediction:.2f} mm")

# Visualization 1: Year-wise average JUN-SEP rainfall
st.subheader("📈 Year-wise Average JUN-SEP Rainfall in India")
yearly = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()
fig1 = px.line(yearly, x="YEAR", y="JUN-SEP", markers=True, title="India: Year-wise Average Rainfall (JUN–SEP)")
st.plotly_chart(fig1, use_container_width=True)

# Visualization 2: Rainfall distribution for top 10 subdivisions
st.subheader("📦 Rainfall Distribution by Top 10 Subdivisions")
top_subs = df["Sub_Division"].value_counts().head(10).index
top_df = df[df["Sub_Division"].isin(top_subs)]
fig2 = px.box(top_df, x="Sub_Division", y="JUN-SEP", color="Sub_Division",
              title="JUN-SEP Rainfall Distribution by Top 10 Subdivisions")
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)
