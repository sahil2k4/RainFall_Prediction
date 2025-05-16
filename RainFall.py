import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

# Run pip install in Streamlit
st.write("Installing xgboost...")
os.system("pip install xgboost")  # Will install on first run, ignored later

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from scipy.stats import zscore
import statsmodels.api as sm

st.title("🌧️ Indian Rainfall Data Analysis & Prediction")

# Load data with path check
def load_data():
    paths = [
        '../DataSets/rainfaLLIndia.csv',
        './rainfaLLIndia.csv',
        'rainfaLLIndia.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("❌ Data file not found. Please upload 'rainfaLLIndia.csv' or place it in the working directory.")
    return None

df = load_data()
if df is None:
    st.stop()

st.subheader("🔍 Raw Data Sample")
st.dataframe(df.head())

# Rename column for consistency
df.rename(columns={'subdivision': 'Sub_Division'}, inplace=True)

# Identify duplicates
st.subheader("📊 Duplicate Rows Info")
exact_duplicates = df[df.duplicated()]
subset_duplicates = df[df.duplicated(subset=['Sub_Division', 'YEAR'], keep=False)]
duplicate_counts = df.groupby(['Sub_Division', 'YEAR']).size().reset_index(name='count')
duplicate_rows = duplicate_counts[duplicate_counts['count'] > 1]

st.write(f"Exact duplicate rows: {len(exact_duplicates)}")
st.write(f"Rows with duplicate subdivision and YEAR: {len(subset_duplicates)}")
st.write("Grouped duplicate subdivision and YEAR pairs:")
st.dataframe(duplicate_rows)

# Feature engineering
df["AVG_RAINFALL"] = df[["JUN", "JUL", "AUG", "SEP"]].mean(axis=1)
df.sort_values(by=["Sub_Division", "YEAR"], inplace=True)
df["YOY_CHANGE"] = df.groupby("Sub_Division")["JUN-SEP"].diff()
df["LAG_1"] = df.groupby("Sub_Division")["JUN-SEP"].shift(1)
df["LAG_2"] = df.groupby("Sub_Division")["JUN-SEP"].shift(2)

def categorize_rainfall(mm):
    if mm < 500:
        return "Low"
    elif mm <= 1000:
        return "Normal"
    else:
        return "High"

df["RAINFALL_CATEGORY_MM"] = df["JUN-SEP"].apply(categorize_rainfall)

# Encode subdivisions
label_enc = LabelEncoder()
df['Sub_Division_Encoded'] = label_enc.fit_transform(df['Sub_Division'])

# Drop rows with NaNs in key engineered columns
df.dropna(subset=["AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2"], inplace=True)
df.reset_index(drop=True, inplace=True)

st.subheader("🛠️ Feature Engineered Data Sample")
st.dataframe(df.head())

# Year-wise average rainfall
st.subheader("📈 Year-wise Average JUN-SEP Rainfall")
yearly_trend = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()
fig = px.line(yearly_trend, x="YEAR", y="JUN-SEP", markers=True, 
              title="Year-wise Average JUN-SEP Rainfall in India", 
              labels={"YEAR":"Year","JUN-SEP":"Avg Rainfall (mm)"}, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Distribution boxplot for top subdivisions
st.subheader("📦 Rainfall Distribution by Top Subdivisions")
top_subs = df["Sub_Division"].value_counts().head(10).index
filtered_df = df[df["Sub_Division"].isin(top_subs)]
fig2 = px.box(filtered_df, x="Sub_Division", y="JUN-SEP", color="Sub_Division", 
              title="JUN-SEP Rainfall Distribution by Top 10 Subdivisions", template="plotly_white")
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap
st.subheader("🔗 Correlation Matrix Heatmap")
rainfall_cols = ["JUN", "JUL", "AUG", "SEP", "JUN-SEP"]
corr_matrix = df[rainfall_cols].corr()
fig3, ax = plt.subplots(figsize=(8,6))
sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix of Rainfall Months and Total (JUN-SEP)")
st.pyplot(fig3)

# Model training
st.subheader("🧠 Train Random Forest Regression Model")

features = ["YEAR", "AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2", "Sub_Division_Encoded"]
target = "JUN-SEP"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.write(f"R² Score: {r2:.4f}")
st.write(f"MAE: {mae:.2f} mm")
st.write(f"RMSE: {rmse:.2f} mm")

# Actual vs Predicted plot
fig4, ax4 = plt.subplots(figsize=(10,6))
ax4.scatter(y_test, y_pred, alpha=0.6, color='teal', edgecolors='black')
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax4.set_xlabel("Actual JUN-SEP Rainfall (mm)")
ax4.set_ylabel("Predicted JUN-SEP Rainfall (mm)")
ax4.set_title("Actual vs Predicted JUN-SEP Rainfall")
ax4.grid(True)
st.pyplot(fig4)

# Interactive prediction
st.subheader("🎯 Predict JUN-SEP Rainfall")

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

# Clustering
st.subheader("🔍 Clustering Subdivisions Based on Average Rainfall")

df_cluster = df.groupby("Sub_Division")[["JUN", "JUL", "AUG", "SEP", "JUN-SEP"]].mean().reset_index()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[["JUN", "JUL", "AUG", "SEP", "JUN-SEP"]])

# KMeans Elbow Method
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig5, ax5 = plt.subplots()
ax5.plot(range(2, 10), inertias, marker='o')
ax5.set_xlabel("Number of Clusters (k)")
ax5.set_ylabel("Inertia")
ax5.set_title("Elbow Method for KMeans Clustering")
ax5.grid(True)
st.pyplot(fig5)

# Apply clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.5, min_samples=3)
df_cluster["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

# Plot cluster assignment
fig6, ax6 = plt.subplots(figsize=(10,5))
sb.scatterplot(x=df_cluster["JUN-SEP"], y=df_cluster["JUL"], hue=df_cluster["KMeans_Cluster"],
               palette="tab10", s=100, ax=ax6)
ax6.set_title("KMeans Clusters of Subdivisions Based on Rainfall")
ax6.set_xlabel("Average JUN-SEP Rainfall")
ax6.set_ylabel("Average JUL Rainfall")
ax6.legend(title="Cluster")
ax6.grid(True)
st.pyplot(fig6)

# Interactive cluster view
fig7 = px.scatter(df_cluster, x="JUN-SEP", y="JUL", color="KMeans_Cluster",
                  hover_name="Sub_Division", title="KMeans Clustering of Subdivisions (Interactive)",
                  labels={"JUN-SEP": "Avg JUN-SEP Rainfall", "JUL": "Avg JUL Rainfall"},
                  template="plotly_white")
st.plotly_chart(fig7, use_container_width=True)

# Final output
st.subheader("📋 Cluster Assignments Table")
st.dataframe(df_cluster[["Sub_Division", "KMeans_Cluster", "DBSCAN_Cluster"]])
