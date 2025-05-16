import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(layout="wide")
st.title("🌧️ Indian Rainfall Prediction & Visualization App")

# Load Data
def load_data():
    paths = ['../DataSets/rainfaLLIndia.csv', './rainfaLLIndia.csv', 'rainfaLLIndia.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    st.error("❌ Data file not found. Please upload 'rainfaLLIndia.csv'.")
    return None

df = load_data()
if df is None:
    st.stop()

# Preprocessing
df.rename(columns={'subdivision': 'Sub_Division'}, inplace=True)
df["AVG_RAINFALL"] = df[["JUN", "JUL", "AUG", "SEP"]].mean(axis=1)
df.sort_values(by=["Sub_Division", "YEAR"], inplace=True)
df["YOY_CHANGE"] = df.groupby("Sub_Division")["JUN-SEP"].diff()
df["LAG_1"] = df.groupby("Sub_Division")["JUN-SEP"].shift(1)
df["LAG_2"] = df.groupby("Sub_Division")["JUN-SEP"].shift(2)
df["RAINFALL_CATEGORY_MM"] = df["JUN-SEP"].apply(lambda x: "Low" if x < 500 else "Normal" if x <= 1000 else "High")
df['Sub_Division_Encoded'] = LabelEncoder().fit_transform(df['Sub_Division'])
df.dropna(subset=["AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2"], inplace=True)
df.reset_index(drop=True, inplace=True)

# Model Training
features = ["YEAR", "AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2", "Sub_Division_Encoded"]
target = "JUN-SEP"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 🔮 Prediction Section
# -------------------------------
st.header("🔮 Predict Future JUN–SEP Rainfall")
subdivision_input = st.selectbox("Select Subdivision", sorted(df["Sub_Division"].unique()))
future_year = st.slider("Select Future Year", 2023, 2040, 2025)

if st.button("Predict Rainfall"):
    sub_df = df[df["Sub_Division"] == subdivision_input].sort_values("YEAR")
    if not sub_df.empty:
        last_rec = sub_df.iloc[-1]
        label_enc = LabelEncoder()
        label_enc.fit(df["Sub_Division"])
        input_dict = {
            "YEAR": future_year,
            "AVG_RAINFALL": last_rec["AVG_RAINFALL"],
            "YOY_CHANGE": last_rec["YOY_CHANGE"],
            "LAG_1": last_rec["JUN-SEP"],
            "LAG_2": last_rec["LAG_1"],
            "Sub_Division_Encoded": label_enc.transform([subdivision_input])[0]
        }
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        st.success(f"🌧️ Predicted JUN–SEP Rainfall for **{subdivision_input}** in **{future_year}**: `{prediction:.2f} mm`")
    else:
        st.error("Data not found for selected subdivision.")

# -------------------------------
# 📊 Visualization Section
# -------------------------------
st.header("📊 Rainfall Visualizations")

# Year-wise trend
st.subheader("📈 Year-wise Average JUN–SEP Rainfall")
yearly = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()
fig1 = px.line(yearly, x="YEAR", y="JUN-SEP", markers=True, title="India: Year-wise Average Rainfall (JUN–SEP)", template="plotly_dark")
st.plotly_chart(fig1, use_container_width=True)

# Boxplot for top subdivisions
st.subheader("📦 Top 10 Subdivisions - Rainfall Distribution")
top_subs = df["Sub_Division"].value_counts().head(10).index
top_df = df[df["Sub_Division"].isin(top_subs)]
fig2 = px.box(top_df, x="Sub_Division", y="JUN-SEP", color="Sub_Division",
              title="JUN–SEP Rainfall Distribution (Top 10 Subdivisions)", template="plotly_white")
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

# Correlation heatmap
st.subheader("🔗 Correlation Heatmap")
corr_cols = ["JUN", "JUL", "AUG", "SEP", "JUN-SEP"]
fig3, ax = plt.subplots(figsize=(8,6))
sb.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig3)

# Rainfall Category Histogram
st.subheader("📊 Rainfall Category Distribution")
fig4 = px.histogram(df, x="RAINFALL_CATEGORY_MM", color="RAINFALL_CATEGORY_MM",
                    title="Distribution of Rainfall Categories (JUN–SEP)", template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

# Clustering
st.subheader("🔍 Clustering Subdivisions")
df_cluster = df.groupby("Sub_Division")[["JUN", "JUL", "AUG", "SEP", "JUN-SEP"]].mean().reset_index()
scaled = StandardScaler().fit_transform(df_cluster.drop("Sub_Division", axis=1))
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster["KMeans_Cluster"] = kmeans.fit_predict(scaled)

dbscan = DBSCAN(eps=1.5, min_samples=3)
df_cluster["DBSCAN_Cluster"] = dbscan.fit_predict(scaled)

fig5 = px.scatter(df_cluster, x="JUN-SEP", y="JUL", color="KMeans_Cluster",
                  hover_name="Sub_Division", title="KMeans Clustering (Avg Rainfall)", template="plotly_white")
st.plotly_chart(fig5, use_container_width=True)

fig6 = px.scatter(df_cluster, x="JUN-SEP", y="JUL", color="DBSCAN_Cluster",
                  hover_name="Sub_Division", title="DBSCAN Clustering (Avg Rainfall)", template="plotly_white")
st.plotly_chart(fig6, use_container_width=True)

# Cluster Table
st.subheader("📋 Cluster Assignment Table")
st.dataframe(df_cluster[["Sub_Division", "KMeans_Cluster", "DBSCAN_Cluster"]])
