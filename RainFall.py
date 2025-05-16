import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🌧️ Indian Rainfall Analysis & Prediction Dashboard")

# Load CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('rainfaLLIndia.csv')
        return df
    except FileNotFoundError:
        st.error("rainfaLLIndia.csv not found. Please upload it.")
        return None

df = load_data()
if df is None:
    st.stop()

# Rename and feature engineering
df.rename(columns={'subdivision': 'Sub_Division'}, inplace=True)
df["AVG_RAINFALL"] = df[["JUN", "JUL", "AUG", "SEP"]].mean(axis=1)
df.sort_values(by=["Sub_Division", "YEAR"], inplace=True)
df["YOY_CHANGE"] = df.groupby("Sub_Division")["JUN-SEP"].diff()
df["LAG_1"] = df.groupby("Sub_Division")["JUN-SEP"].shift(1)
df["LAG_2"] = df.groupby("Sub_Division")["JUN-SEP"].shift(2)

# Drop rows with missing values after lag features
df.dropna(subset=["AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Encode Sub_Division
le = LabelEncoder()
df["Sub_Division_Encoded"] = le.fit_transform(df["Sub_Division"])

# Model preparation
features = ["YEAR", "AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2", "Sub_Division_Encoded"]
target = "JUN-SEP"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊 Model Performance")
st.write(f"**R² Score**: {r2:.3f}")
st.write(f"**MAE**: {mae:.2f} mm")
st.write(f"**RMSE**: {rmse:.2f} mm")

# User Input for Prediction
st.subheader("🎯 Predict Future Rainfall")
selected_division = st.selectbox("Choose Subdivision", sorted(df["Sub_Division"].unique()))
future_year = st.slider("Select Future Year", 2023, 2040, 2025)

if st.button("Predict Rainfall"):
    sub_df = df[df["Sub_Division"] == selected_division].sort_values("YEAR")
    latest = sub_df.iloc[-1]

    input_data = pd.DataFrame([{
        "YEAR": future_year,
        "AVG_RAINFALL": latest["AVG_RAINFALL"],
        "YOY_CHANGE": latest["YOY_CHANGE"],
        "LAG_1": latest["JUN-SEP"],
        "LAG_2": latest["LAG_1"],
        "Sub_Division_Encoded": le.transform([selected_division])[0]
    }])
    
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted JUN-SEP Rainfall for **{selected_division}** in **{future_year}**: **{prediction:.2f} mm**")

# --- Visualizations ---
st.subheader("📈 Visualizations")

# 1. Year-wise Average Rainfall
st.markdown("### India: Year-wise Average JUN-SEP Rainfall")
year_avg = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()
fig1 = px.line(year_avg, x="YEAR", y="JUN-SEP", markers=True)
st.plotly_chart(fig1, use_container_width=True)

# 2. Rainfall Distribution by Top Subdivisions
st.markdown("### Rainfall Distribution: Top 10 Subdivisions")
top_subs = df["Sub_Division"].value_counts().head(10).index
top_df = df[df["Sub_Division"].isin(top_subs)]
fig2 = px.box(top_df, x="Sub_Division", y="JUN-SEP", color="Sub_Division")
st.plotly_chart(fig2, use_container_width=True)

# 3. Correlation Heatmap
st.markdown("### Feature Correlation Heatmap")
import seaborn as sns
import matplotlib.pyplot as plt
corr = df[["YEAR", "AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2", "JUN-SEP"]].corr()
fig3, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
st.pyplot(fig3)
