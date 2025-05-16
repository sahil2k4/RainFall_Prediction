import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor 

# Load your cleaned and prepared dataset (used for model input)
@st.cache_data
def load_data():
    return pd.read_csv("df_model.csv")  # Ensure this has all features used during training

df = load_data()

# Load the label encoder and model (already trained and saved)
le = joblib.load("label_encoder.pkl")  # Save LabelEncoder using joblib
model = joblib.load("rainfall_model.pkl")  # Save model using joblib

# UI title
st.title("🌧️ Indian Rainfall Forecasting App")

# UI inputs
subdivision = st.selectbox("Select Subdivision", df["Sub_Division"].unique())
future_year = st.slider("Select Future Year", min_value=2023, max_value=2040, step=1, value=2025)

# Prediction logic
latest_data = df[df["Sub_Division"] == subdivision].sort_values(by="YEAR").iloc[-1]

input_data = {
    "YEAR": future_year,
    "AVG_RAINFALL": latest_data["AVG_RAINFALL"],
    "YOY_CHANGE": latest_data["YOY_CHANGE"],
    "LAG_1": latest_data["JUN-SEP"],
    "LAG_2": latest_data["LAG_1"],
    "Sub_Division_Encoded": le.transform([subdivision])[0]
}

input_df = pd.DataFrame([input_data])
predicted_rainfall = model.predict(input_df)[0]

# Display result
st.success(f"📅 Predicted Rainfall for {subdivision} in {future_year}: **{predicted_rainfall:.2f} mm**")

# Trend plot for selected subdivision
st.subheader(f"📈 Historical Rainfall Trend - {subdivision}")
sub_df = df[df["Sub_Division"] == subdivision].sort_values(by="YEAR")

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=sub_df, x="YEAR", y="JUN-SEP", label="Actual", marker="o")
ax.axvline(future_year, linestyle='--', color='red', label="Prediction Year")
ax.scatter(future_year, predicted_rainfall, color='green', s=100, label="Predicted")
plt.ylabel("Rainfall (mm)")
plt.title(f"Rainfall Trend for {subdivision}")
plt.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Developed by [Your Name] | Data Source: IMD")
