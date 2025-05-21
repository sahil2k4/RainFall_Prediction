import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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


import plotly.graph_objects as go

# Actual vs Predicted Scatter Plot using Plotly
scatter_fig = go.Figure()

# Scatter points
scatter_fig.add_trace(go.Scatter(
    x=y_test,
    y=y_pred,
    mode='markers',
    marker=dict(color='teal', size=8, line=dict(width=1, color='black')),
    name='Predicted vs Actual'
))

# Reference line y = x
scatter_fig.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines',
    line=dict(dash='dash', color='red'),
    name='Ideal Line (y = x)'
))

# Layout settings
scatter_fig.update_layout(
    title="📉 Actual vs Predicted JUN–SEP Rainfall",
    xaxis_title="Actual JUN–SEP Rainfall (mm)",
    yaxis_title="Predicted JUN–SEP Rainfall (mm)",
    template="plotly_white",
    width=800,
    height=600
)

# Show in Streamlit
st.plotly_chart(scatter_fig, use_container_width=True)


import matplotlib.pyplot as plt

# ================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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

# 1️⃣ Year-wise Average JUN–SEP Rainfall
st.subheader("1️⃣ Year-wise Average JUN–SEP Rainfall")

# Prepare yearly trend data
yearly_trend = df.groupby("YEAR")["JUN-SEP"].mean().reset_index()

# Plot using Plotly
fig1 = px.line(
    yearly_trend,
    x="YEAR",
    y="JUN-SEP",
    markers=True,
    title="Year-wise Average JUN–SEP Rainfall in India",
    labels={"YEAR": "Year", "JUN-SEP": "Avg Rainfall (mm)"},
    color_discrete_sequence=["teal"]
)

fig1.update_traces(line=dict(width=3))
fig1.update_layout(title_x=0.5, template="simple_white")

# Display plot in Streamlit
st.plotly_chart(fig1, use_container_width=True)

# Header
st.markdown("### 📌 Key Insights: JUN–SEP Rainfall in India")

# Stylized insights using Markdown and emojis
st.markdown("""
<div style='font-size:16px; line-height:1.6;'>

🔹 **No clear long-term trend** – Rainfall hasn't consistently increased or decreased since **1901**.  
🔹 **High year-to-year variation** – Indicates strong influence of **monsoon irregularities**.  
🔹 **More extreme highs/lows post-2000** – Suggests **rising climate volatility**.  
🔹 **Recent dry years** – Notably low rainfall around **2002**, **2015–2019**.

</div>
""", unsafe_allow_html=True)



# Subdivision-wise trends for top 6 subdivisions by average rainfall
st.subheader("2️⃣ Top 6 Subdivisions: JUN–SEP Rainfall Over the Years")

# Get top 6 subdivisions with most data
top_subdivisions = df["subdivision"].value_counts().head(6).index

# Initialize figure
fig2 = go.Figure()

# Add a trace for each subdivision
for sub in top_subdivisions:
    sub_data = df[df["subdivision"] == sub]
    fig2.add_trace(go.Scatter(
        x=sub_data["YEAR"],
        y=sub_data["JUN-SEP"],
        mode='lines+markers',
        name=sub
    ))

# Customize layout
fig2.update_layout(
    title="📊 Year-wise JUN-SEP Rainfall Trend for Top 6 Subdivisions",
    xaxis_title="Year",
    yaxis_title="Rainfall (mm)",
    legend_title="Subdivision",
    template="plotly_white",
    hovermode="x unified"
)
st.plotly_chart(fig2)


# Individual insight points with markdown formatting
st.markdown("""
- **_Arunachal Pradesh (17)_** consistently records the **highest rainfall**, with a **rising trend post-2000** — _a potential sign of climate intensification_.
- **_Konkan & Goa (28)_** shows **high but fluctuating rainfall**, maintaining a **stable overall trend** — _characteristic of coastal monsoon behavior_.
- **_Vidarbha (31)_** and **_West Rajasthan (35)_** experience **moderate and relatively steady rainfall**, with **subtle increases in recent years**.
- **_Madhya Maharashtra (29)_** and **_Marathwada (30)_** receive the **lowest rainfall**, with **frequent drought-like dips** — _critical for agricultural planning and water resource management_.
- **Most subdivisions** exhibit **greater year-to-year variability**, suggesting a trend of **increasing climate instability**.
""")

import plotly.express as px
import streamlit as st

st.subheader("3️⃣ Distribution of JUN–SEP Rainfall Across India (All Years)")

# Create histogram using Plotly
fig = px.histogram( 
    df,
    x="JUN-SEP",
    nbins=50,
    title="🌧️ Distribution of JUN–SEP Rainfall Across India (All Years)",
    labels={"JUN-SEP": "Rainfall (mm)"},
    template="plotly_white",
    color_discrete_sequence=["steelblue"]
)

fig.update_layout(
    bargap=0.1,
    title_x=0.5
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Insight points with stylized markdown
st.markdown("""
- **Most common rainfall**: Majority of records fall between **500–1000 mm** during JUN–SEP.
- **Right-skewed distribution**: Indicates that **high rainfall (2000+ mm)** is **rare** but does occur in some regions.
- **Low rainfall events (<400 mm)** are also **frequent**, pointing to **drought-prone areas**.
- **Significant variation**: Reflects **diverse climatic conditions** across India’s subdivisions.
""")

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.header("🔍 DBSCAN Clustering on Rainfall Data")

# Select features for clustering
features = ["JUN-SEP", "JUL"]
df_cluster = df[["subdivision"] + features].groupby("subdivision").mean().reset_index()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[features])

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=3)
df_cluster["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

# Replace -1 with "Noise" for visualization
df_cluster["Cluster_Label"] = df_cluster["DBSCAN_Cluster"].apply(lambda x: f"Cluster {x}" if x != -1 else "Noise")

# Plot using Plotly
fig = px.scatter(
    df_cluster,
    x="JUN-SEP",
    y="JUL",
    color="Cluster_Label",
    hover_name="subdivision",
    title="📍 DBSCAN Clusters of Subdivisions Based on Rainfall",
    labels={"JUN-SEP": "Avg JUN-SEP Rainfall (mm)", "JUL": "Avg JUL Rainfall (mm)"},
    width=900,
    height=600,
)

fig.update_traces(marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")))
fig.update_layout(title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

# Show cluster assignments
st.subheader("📊 Clustered Subdivisions (DBSCAN)")
st.dataframe(df_cluster[["subdivision", "DBSCAN_Cluster"]].sort_values("DBSCAN_Cluster"), use_container_width=True)

import streamlit as st
import plotly.graph_objects as go
import statsmodels.api as sm

st.header("📈 National Rainfall Trend Analysis (JUN–SEP)")

# Group by year to get national average
df_national = df.groupby("YEAR")[["JUN-SEP"]].mean().reset_index()

# Build trendline using OLS
X = sm.add_constant(df_national["YEAR"])
model = sm.OLS(df_national["JUN-SEP"], X).fit()
df_national["TRENDLINE"] = model.predict(X)

# Create interactive Plotly figure
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=df_national["YEAR"],
    y=df_national["JUN-SEP"],
    mode='lines+markers',
    name='Avg Rainfall (JUN-SEP)',
    line=dict(color='blue')
))

fig1.add_trace(go.Scatter(
    x=df_national["YEAR"],
    y=df_national["TRENDLINE"],
    mode='lines',
    name='Trendline',
    line=dict(color='red', dash='dash')
))

# Update layout
fig1.update_layout(
    title='📈 National Rainfall Trend (JUN–SEP)',
    xaxis_title='Year',
    yaxis_title='Average Rainfall (mm)',
    template='plotly_white',
    title_x=0.5
)

# Show in Streamlit
st.plotly_chart(fig1, use_container_width=True)
st.markdown("### 📊 **Additional Insights: National Trendline Observations**")

st.markdown("""
- 📉 There is a **slight declining trend** in **average rainfall over the years**, visible in the _red trendline_.
- 📊 **High year-to-year variability** is evident, marked by **frequent spikes and drops**.
- ⚠️ Despite occasional **extreme rainfall years**, there is **no consistent upward trend**, highlighting ongoing **climate instability**.
""")
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Assuming df is already loaded and has YEAR + JUN-SEP columns
st.header("📉 Smoothed Long-Term Rainfall Trend (10-Year Rolling Average)")

# Group by year to get national average JUN–SEP rainfall
df_national = df.groupby("YEAR")[["JUN-SEP"]].mean().reset_index()

# --- 10-Year Rolling Average ---
df_national["ROLLING_MEAN"] = df_national["JUN-SEP"].rolling(window=10).mean()

# Plot using Plotly
fig2 = go.Figure()

# Annual Rainfall
fig2.add_trace(go.Scatter(
    x=df_national["YEAR"],
    y=df_national["JUN-SEP"],
    mode='lines',
    name='Annual Avg',
    line=dict(color='gray', width=1)
))

# 10-Year Rolling Avg
fig2.add_trace(go.Scatter(
    x=df_national["YEAR"],
    y=df_national["ROLLING_MEAN"],
    mode='lines',
    name='10-Year Rolling Avg',
    line=dict(color='orange', width=3)
))

fig2.update_layout(
    title='📉 Smoothed Long-Term Rainfall Trend (10-Year Rolling Avg)',
    xaxis_title='Year',
    yaxis_title='Rainfall (mm)',
    template='plotly_white',
    title_x=0.5
)

# Show chart
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
- 🔼 **Peak around 1960s**: India saw the **highest sustained rainfall** during the 1960s.
- 📉 **Gradual decline post-1970s**: The 10-year average shows a **steady decrease** in rainfall from the 1970s onward.
- 🔁 **Reduced long-term variability**: Rolling average **smooths out spikes**, revealing a **consistent weakening trend** in monsoon strength.
- ⚠️ Despite occasional **extreme rainfall years**, there is **no consistent upward trend**, pointing to possible **climate instability**.
""")
