import streamlit as st
import os
import warnings
warnings.filterwarnings('ignore')

# Install XGBoost (runs once, ignored on re-run)
st.write("🔧 Installing dependencies (if not already)...")
os.system("pip install xgboost")

# Import packages
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

# Title
st.title("🌧️ Indian Rainfall Data Analysis & Prediction")

# Load data
def load_data():
    paths = [
        '../DataSets/rainfaLLIndia.csv',
        './rainfaLLIndia.csv',
        'rainfaLLIndia.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("❌ Data file not found. Please upload 'rainfaLLIndia.csv'.")
    return None

df = load_data()
if df is None:
    st.stop()

# Initial Data Preview
st.subheader("🔍 Raw Data Sample")
st.dataframe(df.head())

# Clean column names
df.rename(columns={'subdivision': 'Sub_Division'}, inplace=True)

# Duplicates
st.subheader("📊 Duplicate Rows Info")
exact_duplicates = df[df.duplicated()]
subset_duplicates = df[df.duplicated(subset=['Sub_Division', 'YEAR'], keep=False)]
duplicate_counts = df.groupby(['Sub_Division', 'YEAR']).size().reset_index(name='count')
duplicate_rows = duplicate_counts[duplicate_counts['count'] > 1]

st.write(f"Exact duplicate rows: {len(exact_duplicates)}")
st.write(f"Rows with duplicate subdivision and YEAR: {len(subset_duplicates)}")
st.dataframe(duplicate_rows)

# Feature Engineering
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

# Drop NA for model
df.dropna(subset=["AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2"], inplace=True)
df.reset_index(drop=True, inplace=True)

st.subheader("🛠️ Feature Engineered Data Sample")
st.dataframe(df.head())

# Trend Line
st.subheader("📈 Year-wise Average JUN-SEP Rainfall")
yearly_avg = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()
fig = px.line(yearly_avg, x="YEAR", y="JUN-SEP", markers=True,
              title="Year-wise Average JUN-SEP Rainfall", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Boxplot of Top Subdivisions
st.subheader("📦 Rainfall Distribution by Top Subdivisions")
top_subs = df["Sub_Division"].value_counts().head(10).index
filtered = df[df["Sub_Division"].isin(top_subs)]
fig2 = px.box(filtered, x="Sub_Division", y="JUN-SEP", color="Sub_Division",
              title="Top 10 Subdivisions - JUN-SEP Rainfall Distribution", template="plotly_white")
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# Correlation Heatmap
st.subheader("🔗 Correlation Matrix Heatmap")
cols = ["JUN", "JUL", "AUG", "SEP", "JUN-SEP"]
fig3, ax = plt.subplots(figsize=(8,6))
sb.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
st.pyplot(fig3)

# Modeling
st.subheader("🧠 Train Random Forest Regression Model")
features = ["YEAR", "AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2", "Sub_Division_Encoded"]
X = df[features]
y = df["JUN-SEP"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
st.write(f"R² Score: {r2_score(y_test, y_pred):.4f}")
st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} mm")
st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f} mm")

# Actual vs Predicted
fig4, ax4 = plt.subplots()
ax4.scatter(y_test, y_pred, color='teal', edgecolor='black')
ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax4.set_xlabel("Actual")
ax4.set_ylabel("Predicted")
ax4.set_title("Actual vs Predicted JUN-SEP Rainfall")
st.pyplot(fig4)

# Prediction UI
st.subheader("🎯 Predict Future Rainfall")
subdivision_input = st.selectbox("Choose Subdivision", sorted(df["Sub_Division"].unique()))
future_year = st.slider("Select Year", 2023, 2040, 2025)

if st.button("Predict"):
    sub_data = df[df["Sub_Division"] == subdivision_input].sort_values("YEAR")
    if not sub_data.empty:
        last = sub_data.iloc[-1]
        input_vals = {
            "YEAR": future_year,
            "AVG_RAINFALL": last["AVG_RAINFALL"],
            "YOY_CHANGE": last["YOY_CHANGE"],
            "LAG_1": last["JUN-SEP"],
            "LAG_2": last["LAG_1"],
            "Sub_Division_Encoded": label_enc.transform([subdivision_input])[0]
        }
        input_df = pd.DataFrame([input_vals])
        pred = rf.predict(input_df)[0]
        st.success(f"🌧️ Predicted Rainfall for {subdivision_input} in {future_year}: {pred:.2f} mm")
    else:
        st.error("No data for selected subdivision.")

# Clustering
st.subheader("🔍 Clustering Subdivisions Based on Rainfall")
df_cluster = df.groupby("Sub_Division")[["JUN", "JUL", "AUG", "SEP", "JUN-SEP"]].mean().reset_index()
X_clust = StandardScaler().fit_transform(df_cluster.drop("Sub_Division", axis=1))

# Elbow method
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_clust)
    inertias.append(km.inertia_)

fig5, ax5 = plt.subplots()
ax5.plot(range(2, 10), inertias, marker='o')
ax5.set_title("Elbow Method for KMeans")
ax5.set_xlabel("k")
ax5.set_ylabel("Inertia")
st.pyplot(fig5)

# Apply clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster["KMeans_Cluster"] = kmeans.fit_predict(X_clust)

dbscan = DBSCAN(eps=1.5, min_samples=3)
df_cluster["DBSCAN_Cluster"] = dbscan.fit_predict(X_clust)

# Static scatter plot
fig6, ax6 = plt.subplots()
sb.scatterplot(data=df_cluster, x="JUN-SEP", y="JUL", hue="KMeans_Cluster", ax=ax6, s=100)
ax6.set_title("KMeans Clustering")
st.pyplot(fig6)

# Interactive
fig7 = px.scatter(df_cluster, x="JUN-SEP", y="JUL", color="KMeans_Cluster",
                  hover_name="Sub_Division", title="KMeans Clustering (Interactive)")
st.plotly_chart(fig7, use_container_width=True)

# Final Table
st.subheader("📋 Cluster Assignments Table")
st.dataframe(df_cluster[["Sub_Division", "KMeans_Cluster", "DBSCAN_Cluster"]])
