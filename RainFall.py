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



import matplotlib.pyplot as plt

# ================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("🌧️ Indian Rainfall Analysis & Prediction Dashboard")

# Load your CSV file (adjust path if needed)
@st.cache_data
def load_data():
    df = pd.read_csv('rainfaLLIndia.csv')  # your actual file path here
    return df

df = load_data()

# Basic preprocessing assuming your columns:
# 'YEAR', 'JUN-SEP', 'Sub_Division'

# Year-wise average rainfall
yearly_trend = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()

st.header("📈 Rainfall Trend Analysis")

# Plot year-wise average rainfall
st.subheader("1️⃣ Year-wise Average JUN–SEP Rainfall")
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(yearly_trend["YEAR"], yearly_trend["JUN-SEP"], marker='o', color='teal', linewidth=2)
ax1.set_title("Year-wise Average JUN-SEP Rainfall in India", fontsize=16)
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Rainfall (mm)")
ax1.grid(True)
st.pyplot(fig1)

# Subdivision-wise trends for top 6 subdivisions by average rainfall
st.subheader("2️⃣ Top 6 Subdivisions: JUN–SEP Rainfall Over the Years")

top_subs = df.groupby("Sub_Division")["JUN-SEP"].mean().nlargest(6).index.tolist()
sub_df = df[df["Sub_Division"].isin(top_subs)]

fig2, ax2 = plt.subplots(figsize=(10, 6))
for division in top_subs:
    division_data = sub_df[sub_df["Sub_Division"] == division]
    ax2.plot(division_data["YEAR"], division_data["JUN-SEP"], marker='o', label=division)

ax2.set_title("Top 6 Subdivisions - Year-wise JUN-SEP Rainfall", fontsize=16)
ax2.set_xlabel("Year")
ax2.set_ylabel("Rainfall (mm)")
ax2.legend(title="Subdivision", fontsize=8)
ax2.grid(True)
st.pyplot(fig2)

