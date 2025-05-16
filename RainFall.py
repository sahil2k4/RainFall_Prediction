import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Title
st.title("🌧️ Indian Rainfall Forecasting and Visualization")

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv("rainfall_data.csv")  # Replace with your dataset filename
    df["YEAR"] = df["YEAR"].astype(int)
    return df

df = load_data()

# Sidebar - User input
st.sidebar.header("🔧 User Input for Prediction")
subdivisions = df["SUBDIVISION"].unique()
selected_subdivision = st.sidebar.selectbox("Select Subdivision", sorted(subdivisions))
future_year = st.sidebar.number_input("Enter Future Year (e.g., 2026)", min_value=2025, max_value=2100, step=1)

# Filter data by selected subdivision
df_sub = df[df["SUBDIVISION"] == selected_subdivision]

# Feature engineering
df_sub["Prev_Annual"] = df_sub["ANNUAL"].shift(1)
df_sub.dropna(inplace=True)

# Model Training
X = df_sub[["YEAR", "Prev_Annual"]]
y = df_sub["ANNUAL"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Prepare input for prediction
if not df_sub.empty:
    latest_year = df_sub["YEAR"].max()
    latest_annual = df_sub[df_sub["YEAR"] == latest_year]["ANNUAL"].values[0]
    future_input = pd.DataFrame([[future_year, latest_annual]], columns=["YEAR", "Prev_Annual"])
    prediction = model.predict(future_input)[0]
    st.subheader(f"📈 Predicted Annual Rainfall for {selected_subdivision} in {future_year}:")
    st.success(f"{prediction:.2f} mm")

# Visualization 1: Year-wise Annual Rainfall for Selected Subdivision
st.subheader(f"📊 Year-wise Annual Rainfall: {selected_subdivision}")
fig1 = px.line(df_sub, x="YEAR", y="ANNUAL", title=f"Annual Rainfall in {selected_subdivision}", markers=True, template=None)
st.plotly_chart(fig1)

# Visualization 2: JUN-SEP rainfall trend (India-level average)
st.subheader("🌧️ India-wide JUN–SEP Rainfall Trend")
jun_sep_avg = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()
fig2 = px.line(jun_sep_avg, x="YEAR", y="JUN-SEP", title="India: Year-wise Average JUN–SEP Rainfall", markers=True, template=None)
st.plotly_chart(fig2)

# Visualization 3: Top 5 wettest subdivisions
st.subheader("🏆 Top 5 Wettest Subdivisions (Avg Annual)")
top5 = df.groupby("SUBDIVISION")["ANNUAL"].mean().sort_values(ascending=False).head(5).index
df_top5 = df[df["SUBDIVISION"].isin(top5)]
fig3 = px.line(df_top5, x="YEAR", y="ANNUAL", color="SUBDIVISION", title="Top 5 Wettest Subdivisions Over Years", markers=True, template=None)
st.plotly_chart(fig3)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Plotly")

