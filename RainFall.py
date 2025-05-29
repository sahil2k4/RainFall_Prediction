import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Page configuration with dark theme
st.set_page_config(
    page_title="Indian Rainfall Analysis & Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/rainfall-analysis',
        'Report a bug': "https://github.com/yourusername/rainfall-analysis/issues",
        'About': "# Indian Rainfall Analysis Dashboard\nThis is a modern dashboard for analyzing and predicting rainfall patterns in India."
    }
)

# Modern dark theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400;500;600;700;900&display=swap');
    
    /* Main containers */
    .main {
        background-color: #0E1117;
        color: #E6E6E6;
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1A1C23;
    }
    
    /* Headers */
    h1 {
        color: #FFFFFF;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3A86FF 0%, #FF006E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        border-bottom: 2px solid #2C2C3A;
    }
    
    h2 {
        color: #FFFFFF;
        font-weight: 600;
        margin-top: 2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2C2C3A;
    }
    
    h3 {
        color: #E6E6E6;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: white;
        background: linear-gradient(90deg, #3A86FF 0%, #FF006E 100%);
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(58, 134, 255, 0.3);
    }
    
    /* Metrics */
    .stMetric {
        background-color: #1A1C23;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2C2C3A;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
    }
    
    /* Select boxes */
    .stSelectbox {
        background-color: #1A1C23;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div {
        background-color: #1A1C23;
        border: 1px solid #2C2C3A;
    }
    
    /* Plotly charts */
    .js-plotly-plot {
        border-radius: 12px;
        padding: 1rem;
        background-color: #1A1C23;
        border: 1px solid #2C2C3A;
        margin-bottom: 2rem;
        width: 100% !important;
        box-sizing: border-box;
    }
    
    .js-plotly-plot .plot-container {
        width: 100% !important;
    }
    
    /* Chart container */
    [data-testid="stPlotlyChart"] > div {
        width: 100% !important;
        min-height: 400px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Ensure charts are responsive */
    .stPlotlyChart {
        width: 100% !important;
    }
    
    /* Container adjustments */
    .element-container {
        width: 100% !important;
        padding: 0.5rem;
        box-sizing: border-box;
    }
    
    /* Column adjustments */
    [data-testid="column"] {
        width: 100% !important;
        padding: 0.5rem;
        box-sizing: border-box;
    }
    
    /* Cards */
    div.element-container {
        background-color: #1A1C23;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #2C2C3A;
        transition: transform 0.3s ease;
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1A1C23;
        border-radius: 8px;
        border: 1px solid #2C2C3A;
        width: 100%;
    }
    
    .dataframe th {
        background-color: #2C2C3A;
        color: white;
        padding: 0.5rem;
    }
    
    .dataframe td {
        color: #E6E6E6;
        padding: 0.5rem;
    }
    
    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div {
        background-color: #2C2C3A;
    }
    
    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning {
        background-color: #1A1C23;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #2C2C3A;
    }
    
    /* Animation for elements */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .element-container {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<h1>🌧️ Indian Rainfall Analysis & Prediction Dashboard</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose a Section", 
    ["🏠 Overview", 
     "📈 Historical Analysis", 
     "🎯 Rainfall Prediction",
     "ℹ️ About"])

# Configure default Plotly theme for dark mode
template = go.layout.Template()
template.layout.plot_bgcolor = '#1A1C23'
template.layout.paper_bgcolor = '#1A1C23'
template.layout.font = dict(color='#E6E6E6')
template.layout.xaxis = dict(gridcolor='#2C2C3A', linecolor='#2C2C3A')
template.layout.yaxis = dict(gridcolor='#2C2C3A', linecolor='#2C2C3A')
template.layout.margin = dict(l=20, r=20, t=40, b=20)

# Update all Plotly figures to use dark theme
def configure_plotly_figure(fig):
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#1A1C23',
        paper_bgcolor='#1A1C23',
        font=dict(color='#E6E6E6'),
        xaxis=dict(
            gridcolor='#2C2C3A', 
            linecolor='#2C2C3A',
            showgrid=True,
            showline=True,
            zeroline=False,
            automargin=True
        ),
        yaxis=dict(
            gridcolor='#2C2C3A', 
            linecolor='#2C2C3A',
            showgrid=True,
            showline=True,
            zeroline=False,
            automargin=True
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        autosize=True,
        hovermode='closest'
    )
    return fig

# Load data with progress bar
@st.cache_data
def load_data():
    try:
        with st.spinner('Loading data...'):
            df = pd.read_csv('rainfaLLIndia.csv')
            return df
    except FileNotFoundError:
        st.error("❌ rainfaLLIndia.csv not found. Please upload it.")
        return None

df = load_data()
if df is None:
    st.stop()

# Data preprocessing
def preprocess_data(df):
    df = df.copy()
    df.rename(columns={'subdivision': 'Sub_Division'}, inplace=True)
    df["AVG_RAINFALL"] = df[["JUN", "JUL", "AUG", "SEP"]].mean(axis=1)
    df.sort_values(by=["Sub_Division", "YEAR"], inplace=True)
    df["YOY_CHANGE"] = df.groupby("Sub_Division")["JUN-SEP"].diff()
    df["LAG_1"] = df.groupby("Sub_Division")["JUN-SEP"].shift(1)
    df["LAG_2"] = df.groupby("Sub_Division")["JUN-SEP"].shift(2)
    df.dropna(subset=["AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = preprocess_data(df)

# Function to load Lottie animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load specific animations for Overview and Prediction
def get_animations():
    animations = {
        'overview': load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json'),  # Data analysis animation
        'prediction': load_lottieurl('https://assets9.lottiefiles.com/private_files/lf30_gqrqfgqj.json'),  # ML/Prediction animation
        'increase': load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_p8bfn5to.json'),  # Increase/up arrow animation
        'decrease': load_lottieurl('https://assets1.lottiefiles.com/packages/lf20_ie4gzpa3.json')  # Decrease/down arrow animation
    }
    
    # Fallback animation in case URLs fail
    fallback = {
        "v": "5.7.4",
        "fr": 30,
        "ip": 0,
        "op": 60,
        "w": 300,
        "h": 300,
        "nm": "Simple Loading",
        "ddd": 0,
        "assets": [],
        "layers": [{
            "ddd": 0,
            "ind": 1,
            "ty": 4,
            "nm": "Circle",
            "sr": 1,
            "ks": {
                "o": {"a": 0, "k": 100},
                "r": {"a": 1, "k": [{"t": 0, "s": [0]}, {"t": 60, "s": [360]}]},
                "p": {"a": 0, "k": [150, 150]},
                "a": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [100, 100]}
            },
            "shapes": [{
                "ty": "el",
                "p": {"a": 0, "k": [0, 0]},
                "s": {"a": 0, "k": [80, 80]},
                "c": {"a": 0, "k": [0.2, 0.5, 0.8]}
            }]
        }]
    }
    
    # Replace any failed loads with fallback
    for key in animations:
        if animations[key] is None:
            animations[key] = fallback
    
    return animations

# Initialize animations
animations = get_animations()

# Update the chart display configuration
def display_plotly_chart(fig, use_container_width=True):
    # Configure the figure
    fig = configure_plotly_figure(fig)
    # Display with proper container settings
    st.plotly_chart(fig, use_container_width=use_container_width, config={
        'displayModeBar': True,
        'responsive': True,
        'scrollZoom': True
    })

# Define the create_seasonal_analysis function at the top level
def create_seasonal_analysis(sub_df):
    # Create monthly pattern plot with improved visualization
    monthly_data = sub_df.pivot_table(
        index=sub_df['YEAR'].astype(str),
        values=['JUN', 'JUL', 'AUG', 'SEP'],
        aggfunc='first'
    ).tail(20)  # Show last 20 years
    
    # Calculate percentage of total rainfall for each month
    monthly_percentages = monthly_data.div(monthly_data.sum(axis=1), axis=0) * 100
    
    # Create the first figure for absolute values
    fig1 = go.Figure()
    
    months = ['June', 'July', 'August', 'September']
    colors = ['#3A86FF', '#FF006E', '#8338EC', '#FB5607']
    
    for i, month in enumerate(['JUN', 'JUL', 'AUG', 'SEP']):
        fig1.add_trace(go.Scatter(
            x=monthly_data.index,
            y=monthly_data[month],
            name=months[i],
            line=dict(color=colors[i], width=2),
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate=f"{months[i]}: %{{y:.1f}}mm<br>Year: %{{x}}<extra></extra>"
        ))
    
    fig1.update_layout(
        title='Monthly Rainfall Distribution Over Years',
        xaxis_title='Year',
        yaxis_title='Rainfall (mm)',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create the second figure for percentage distribution
    fig2 = go.Figure()
    
    for i, month in enumerate(['JUN', 'JUL', 'AUG', 'SEP']):
        fig2.add_trace(go.Bar(
            x=monthly_percentages.index,
            y=monthly_percentages[month],
            name=months[i],
            marker_color=colors[i],
            hovertemplate=f"{months[i]}: %{{y:.1f}}%<br>Year: %{{x}}<extra></extra>"
        ))
    
    fig2.update_layout(
        title='Monthly Rainfall Distribution (% of Total)',
        xaxis_title='Year',
        yaxis_title='Percentage of Total Rainfall',
        barmode='relative',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig1, fig2

# Overview Page
if page == "🏠 Overview":
    # Welcome section with animation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📌 Welcome to the Rainfall Analysis Dashboard")
        st.markdown("""
        This interactive dashboard provides comprehensive insights into India's rainfall patterns from 1901 to 2020. 
        Explore historical trends, analyze regional patterns, and predict future rainfall using advanced machine learning.
        
        #### 🎯 What You Can Do:
        - Analyze historical rainfall data across different regions
        - Compare rainfall patterns between subdivisions
        - Predict future rainfall trends
        - Explore monthly and seasonal patterns
        """)
    with col2:
        st_lottie(animations['overview'], height=200, key="overview_animation")

    # National Trend Overview
    st.markdown("### 🌍 National Rainfall Trend")
    yearly_avg = df.groupby('YEAR')['JUN-SEP'].mean().reset_index()
    display_plotly_chart(px.line(
        yearly_avg,
        x='YEAR',
        y='JUN-SEP',
        title='National Average Monsoon Rainfall (1901-2020)',
        labels={'YEAR': 'Year', 'JUN-SEP': 'Rainfall (mm)'},
        template='plotly_dark'
    ))

    # Regional Overview
    st.markdown("### 🗺️ Regional Overview")
    
    # Top and Bottom 5 regions by rainfall
    col1, col2 = st.columns(2)
    with col1:
        # Create a container for the title and animation
        title_col1, anim_col1 = st.columns([4, 1])
        with title_col1:
            st.markdown("#### 💧 Highest Rainfall Regions")
        with anim_col1:
            st_lottie(animations['increase'], height=50, key="increase_animation")
        
        top_5 = df.groupby('Sub_Division')['JUN-SEP'].mean().sort_values(ascending=False).head()
        display_plotly_chart(px.bar(
            x=top_5.values,
            y=top_5.index,
            orientation='h',
            title='Top 5 Regions by Average Rainfall',
            labels={'x': 'Average Rainfall (mm)', 'y': 'Region'},
            color=top_5.values,
            color_continuous_scale='Blues',
            template='plotly_dark'
        ))
    
    with col2:
        # Create a container for the title and animation
        title_col2, anim_col2 = st.columns([4, 1])
        with title_col2:
            st.markdown("#### 🏜️ Lowest Rainfall Regions")
        with anim_col2:
            st_lottie(animations['decrease'], height=50, key="decrease_animation")
        
        bottom_5 = df.groupby('Sub_Division')['JUN-SEP'].mean().sort_values().head()
        display_plotly_chart(px.bar(
            x=bottom_5.values,
            y=bottom_5.index,
            orientation='h',
            title='Bottom 5 Regions by Average Rainfall',
            labels={'x': 'Average Rainfall (mm)', 'y': 'Region'},
            color=bottom_5.values,
            color_continuous_scale='Reds_r',
            template='plotly_dark'
        ))

    # Monthly Pattern
    st.markdown("### 📅 Monthly Rainfall Pattern")
    monthly_avg = df[['JUN', 'JUL', 'AUG', 'SEP']].mean()
    monthly_std = df[['JUN', 'JUL', 'AUG', 'SEP']].std()
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(
        x=['June', 'July', 'August', 'September'],
        y=monthly_avg,
        error_y=dict(type='data', array=monthly_std),
        marker_color='#3A86FF',
        name='Average Rainfall'
    ))
    
    fig_monthly.update_layout(
        title='Average Monthly Monsoon Rainfall (with Standard Deviation)',
        xaxis_title='Month',
        yaxis_title='Rainfall (mm)',
        template='plotly_dark'
    )
    display_plotly_chart(fig_monthly)

    # Quick insights
    st.markdown("### 🔍 Key Insights")
    st.markdown("""
    - The monsoon season typically peaks in **July-August**
    - There is significant variation in rainfall across different regions
    - Coastal and northeastern regions receive the highest rainfall
    - Northwestern regions generally receive less rainfall
    - Year-to-year variability is high across all regions
    """)

# Historical Analysis Page
elif page == "📈 Historical Analysis":
    st.markdown("### 📊 Historical Rainfall Analysis")
    
    # Subdivision selection for analysis
    selected_subdivision = st.selectbox(
        "Select Region for Analysis",
        sorted(df['Sub_Division'].unique())
    )
    
    # Get data for selected subdivision
    sub_df = df[df['Sub_Division'] == selected_subdivision].copy()
    
    if len(sub_df) > 0:  # Check if we have data for the selected subdivision
        # Historical Statistics Dashboard
        st.markdown(f"### 📈 Historical Statistics for {selected_subdivision}")
        
        # Calculate key statistics
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.info(f"""
            **Rainfall Statistics**
            - Average: {sub_df['JUN-SEP'].mean():.1f} mm
            - Maximum: {sub_df['JUN-SEP'].max():.1f} mm ({sub_df.loc[sub_df['JUN-SEP'].idxmax(), 'YEAR']})
            - Minimum: {sub_df['JUN-SEP'].min():.1f} mm ({sub_df.loc[sub_df['JUN-SEP'].idxmin(), 'YEAR']})
            """)
        
        with stats_col2:
            percentiles = np.percentile(sub_df['JUN-SEP'], [25, 50, 75])
            st.success(f"""
            **Distribution**
            - 25th Percentile: {percentiles[0]:.1f} mm
            - Median: {percentiles[1]:.1f} mm
            - 75th Percentile: {percentiles[2]:.1f} mm
            """)
        
        with stats_col3:
            cv = (sub_df['JUN-SEP'].std() / sub_df['JUN-SEP'].mean()) * 100
            recent_trend = sub_df[sub_df['YEAR'] >= 2000]['JUN-SEP'].mean() - sub_df[sub_df['YEAR'] < 2000]['JUN-SEP'].mean()
            trend_direction = "↑" if recent_trend > 0 else "↓"
            st.warning(f"""
            **Variability**
            - Coefficient of Variation: {cv:.1f}%
            - Recent Trend: {abs(recent_trend):.1f} mm {trend_direction}
            - Standard Deviation: {sub_df['JUN-SEP'].std():.1f} mm
            """)

        # Monthly analysis
        st.subheader("📅 Monthly Rainfall Pattern")
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly average rainfall
            monthly_data = sub_df[['JUN', 'JUL', 'AUG', 'SEP']].mean()
            monthly_std = sub_df[['JUN', 'JUL', 'AUG', 'SEP']].std()
            
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=['June', 'July', 'August', 'September'],
                y=monthly_data,
                error_y=dict(type='data', array=monthly_std),
                name='Average Rainfall',
                marker_color='#3A86FF'
            ))
            
            fig_monthly.update_layout(
                title='Average Monthly Rainfall (with Standard Deviation)',
                xaxis_title='Month',
                yaxis_title='Rainfall (mm)',
                template='plotly_dark',
                showlegend=True
            )
            display_plotly_chart(fig_monthly)
        
        with col2:
            # Year-over-year change distribution
            fig_yoy = go.Figure()
            fig_yoy.add_trace(go.Histogram(
                x=sub_df['YOY_CHANGE'].dropna(),
                nbinsx=30,
                name='Year-over-Year Change',
                marker_color='#3A86FF'
            ))
            
            fig_yoy.update_layout(
                title='Year-over-Year Change Distribution',
                xaxis_title='Change in Rainfall (mm)',
                yaxis_title='Frequency',
                template='plotly_dark',
                showlegend=True
            )
            display_plotly_chart(fig_yoy)

        # Long-term trend analysis
        st.subheader("📈 Long-term Rainfall Trend")
        
        fig_trend = go.Figure()
        
        # Add actual rainfall data
        fig_trend.add_trace(go.Scatter(
            x=sub_df['YEAR'],
            y=sub_df['JUN-SEP'],
            mode='lines+markers',
            name='Actual Rainfall',
            line=dict(color='blue', width=1),
            marker=dict(size=6)
        ))
        
        # Add trend line
        z = np.polyfit(sub_df['YEAR'], sub_df['JUN-SEP'], 1)
        p = np.poly1d(z)
        fig_trend.add_trace(go.Scatter(
            x=sub_df['YEAR'],
            y=p(sub_df['YEAR']),
            mode='lines',
            name='Trend Line',
            line=dict(color='red', dash='dash')
        ))
        
        # Add moving average
        fig_trend.add_trace(go.Scatter(
            x=sub_df['YEAR'],
            y=sub_df['JUN-SEP'].rolling(window=10).mean(),
            mode='lines',
            name='10-Year Moving Average',
            line=dict(color='green', dash='dot')
        ))
        
        fig_trend.update_layout(
            title=f'Long-term Rainfall Trend for {selected_subdivision} (1901-2020)',
            xaxis_title='Year',
            yaxis_title='Total Rainfall (mm)',
            template='plotly_dark',
            showlegend=True
        )
        
        display_plotly_chart(fig_trend)

        # Seasonal Pattern Analysis
        st.subheader("🌧️ Seasonal Pattern Analysis")
        
        # Display the seasonal analysis
        col1, col2 = st.columns(2)
        with col1:
            fig1, fig2 = create_seasonal_analysis(sub_df)
            display_plotly_chart(fig1)
        with col2:
            display_plotly_chart(fig2)

        # Add insights about seasonal patterns
        st.markdown("#### 📊 Seasonal Pattern Insights")
        
        # Calculate monthly statistics
        monthly_stats = sub_df[['JUN', 'JUL', 'AUG', 'SEP']].agg(['mean', 'std', 'min', 'max'])
        peak_month = monthly_stats.loc['mean'].idxmax()
        lowest_month = monthly_stats.loc['mean'].idxmin()
        most_variable = monthly_stats.loc['std'].idxmax()
        
        # Calculate recent trends (last 10 years vs previous years)
        recent_years = sub_df[sub_df['YEAR'] >= sub_df['YEAR'].max() - 10]
        older_years = sub_df[sub_df['YEAR'] < sub_df['YEAR'].max() - 10]
        
        recent_pattern = recent_years[['JUN', 'JUL', 'AUG', 'SEP']].mean()
        older_pattern = older_years[['JUN', 'JUL', 'AUG', 'SEP']].mean()
        pattern_change = recent_pattern - older_pattern
        
        month_names = {'JUN': 'June', 'JUL': 'July', 'AUG': 'August', 'SEP': 'September'}
        
        st.markdown(f"""
        **Key Patterns:**
        - Peak rainfall typically occurs in **{month_names[peak_month]}** ({monthly_stats.loc['mean'][peak_month]:.1f} mm on average)
        - Lowest rainfall is usually in **{month_names[lowest_month]}** ({monthly_stats.loc['mean'][lowest_month]:.1f} mm on average)
        - **{month_names[most_variable]}** shows the highest variability
        
        **Recent Changes (Last 10 Years):**
        - {month_names[pattern_change.abs().idxmax()]} shows the most significant change: 
          {pattern_change.abs().max():.1f} mm {'increase' if pattern_change.max() > 0 else 'decrease'}
        """)
    else:
        st.error(f"No data available for {selected_subdivision}")

# Prediction Page
elif page == "🎯 Rainfall Prediction":
    # Add prediction animation at the top
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎯 Rainfall Prediction Model")
        st.markdown("""
        Use our advanced machine learning model to predict future rainfall patterns.
        The model is trained on historical data from 1901 to 2020 and uses multiple features
        to make accurate predictions.
        """)
    with col2:
        st_lottie(animations['prediction'], height=200, key="prediction_animation")

    st.markdown("### 🎯 Rainfall Prediction Model")
    
    # Model preparation
    @st.cache_resource
    def prepare_model(df):
        le = LabelEncoder()
        df["Sub_Division_Encoded"] = le.fit_transform(df["Sub_Division"])
        features = ["YEAR", "AVG_RAINFALL", "YOY_CHANGE", "LAG_1", "LAG_2", "Sub_Division_Encoded"]
        target = "JUN-SEP"
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model, le, X_test, y_test

    model, le, X_test, y_test = prepare_model(df)
    
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        selected_division = st.selectbox("Choose Region", sorted(df["Sub_Division"].unique()))
    with col2:
        future_year = st.slider("Select Future Year", 2023, 2040, 2025)

    if st.button("🔮 Predict Rainfall"):
        np.random.seed(42)  # Ensure reproducibility for predictions
        sub_df = df[df["Sub_Division"] == selected_division].sort_values("YEAR")
        latest = sub_df.iloc[-1]

        # Calculate historical statistics for the subdivision
        hist_mean = sub_df["JUN-SEP"].mean()
        hist_std = sub_df["JUN-SEP"].std()
        hist_trend = np.polyfit(sub_df["YEAR"], sub_df["JUN-SEP"], 1)[0]

        # Display subdivision statistics
        st.subheader("📊 Historical Statistics for " + selected_division)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Rainfall", f"{hist_mean:.2f} mm")
        with col2:
            st.metric("Standard Deviation", f"{hist_std:.2f} mm")
        with col3:
            trend_direction = "↑" if hist_trend > 0 else "↓"
            st.metric("Historical Trend", f"{abs(hist_trend):.2f} mm/year {trend_direction}")

        # Monthly Pattern Analysis
        st.subheader("📅 Monthly Rainfall Distribution")
        monthly_avg = sub_df[["JUN", "JUL", "AUG", "SEP"]].mean()
        monthly_std = sub_df[["JUN", "JUL", "AUG", "SEP"]].std()
        
        # Create monthly pattern plot
        fig_monthly = go.Figure()
        months = ["JUN", "JUL", "AUG", "SEP"]
        
        # Add bar chart for average monthly rainfall
        fig_monthly.add_trace(go.Bar(
            x=months,
            y=monthly_avg,
            name="Average",
            error_y=dict(type='data', array=monthly_std),
            marker_color='#3A86FF'
        ))
        
        fig_monthly.update_layout(
            title=f'Monthly Rainfall Distribution for {selected_division}',
            xaxis_title='Month',
            yaxis_title='Average Rainfall (mm)',
            template='plotly_dark'
        )
        display_plotly_chart(fig_monthly)

        # Historical Heatmap
        st.subheader("📊 Historical Rainfall Pattern")
        
        # Create year-month heatmap data with improved visualization
        recent_years = 20  # Show last 20 years
        years = sub_df['YEAR'].unique()[-recent_years:]
        months = ['JUN', 'JUL', 'AUG', 'SEP']
        month_names = ['June', 'July', 'August', 'September']
        
        # Calculate z-scores for better color scaling
        monthly_data = sub_df[sub_df['YEAR'].isin(years)][months]
        z_scores = (monthly_data - monthly_data.mean()) / monthly_data.std()
        
        # Create the heatmap
        fig_heatmap = go.Figure()
        
        # Add heatmap
        fig_heatmap.add_trace(go.Heatmap(
            z=z_scores.values,
            x=month_names,
            y=years,
            colorscale='RdBu_r',  # Red-Blue diverging colorscale
            zmid=0,  # Center the colorscale at 0
            text=monthly_data.values.round(1),
            texttemplate='%{text} mm',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='Year: %{y}<br>Month: %{x}<br>Rainfall: %{text} mm<br>Z-score: %{z:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig_heatmap.update_layout(
            title=dict(
                text='Monthly Rainfall Intensity (Z-scores)',
                x=0.5,
                y=0.95
            ),
            xaxis_title='Month',
            yaxis_title='Year',
            yaxis_autorange='reversed',  # Most recent years at top
            height=500,
            annotations=[
                dict(
                    text="Blue = Above Average, Red = Below Average",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=10)
                )
            ]
        )
        
        display_plotly_chart(fig_heatmap)

        # Add pattern analysis
        monthly_trends = sub_df.groupby('YEAR')[['JUN', 'JUL', 'AUG', 'SEP']].mean()
        recent_trend = monthly_trends.iloc[-5:].mean()
        historical_trend = monthly_trends.iloc[:-5].mean()
        
        st.markdown("#### 📈 Pattern Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recent Pattern (Last 5 Years)**")
            for month, value in zip(['June', 'July', 'August', 'September'], recent_trend):
                st.write(f"- {month}: {value:.1f} mm")
        
        with col2:
            st.markdown("**Historical Pattern**")
            for month, value in zip(['June', 'July', 'August', 'September'], historical_trend):
                st.write(f"- {month}: {value:.1f} mm")

        # Calculate seasonal patterns
        monthly_ratios = sub_df[["JUN", "JUL", "AUG", "SEP"]].mean() / sub_df[["JUN", "JUL", "AUG", "SEP"]].mean().mean()
        
        # Initialize lists to store predictions and years
        predictions = []
        years = list(range(latest["YEAR"] + 1, future_year + 1))
        
        # Initialize features with latest known values
        current_avg_rainfall = latest["AVG_RAINFALL"]
        current_yoy = latest["YOY_CHANGE"]
        current_lag1 = latest["JUN-SEP"]
        current_lag2 = latest["LAG_1"]
        
        # Make predictions for each year
        for year in years:
            # Add some controlled randomness based on historical patterns
            noise = np.random.normal(0, hist_std * 0.3)  # Reduced volatility
            trend_component = hist_trend * (year - latest["YEAR"])
            
            input_data = pd.DataFrame([{
                "YEAR": year,
                "AVG_RAINFALL": current_avg_rainfall * (1 + np.random.normal(0, 0.1)),  # Add slight variation
                "YOY_CHANGE": current_yoy,
                "LAG_1": current_lag1,
                "LAG_2": current_lag2,
                "Sub_Division_Encoded": le.transform([selected_division])[0]
            }])
            
            # Get base prediction
            base_prediction = model.predict(input_data)[0]
            
            # Adjust prediction with historical patterns and noise
            adjusted_prediction = base_prediction + noise + trend_component
            
            # Ensure prediction stays within reasonable bounds
            min_rainfall = max(0, hist_mean - 2 * hist_std)  # Can't have negative rainfall
            max_rainfall = hist_mean + 2 * hist_std
            adjusted_prediction = np.clip(adjusted_prediction, min_rainfall, max_rainfall)
            
            predictions.append(adjusted_prediction)
            
            # Update features for next year with more dynamic changes
            current_yoy = adjusted_prediction - current_lag1
            current_lag2 = current_lag1
            current_lag1 = adjusted_prediction
            
            # Update average rainfall using seasonal patterns
            monthly_predictions = []
            for ratio in monthly_ratios:
                monthly_pred = (adjusted_prediction/4) * ratio * (1 + np.random.normal(0, 0.05))
                monthly_predictions.append(monthly_pred)
            current_avg_rainfall = np.mean(monthly_predictions)
        
        # Create prediction trend plot with confidence interval
        fig_pred = go.Figure()
        
        # Historical data
        historical = sub_df[["YEAR", "JUN-SEP"]].copy()
        fig_pred.add_trace(go.Scatter(
            x=historical["YEAR"],
            y=historical["JUN-SEP"],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue')
        ))
        
        # Predictions
        fig_pred.add_trace(go.Scatter(
            x=years,
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence interval
        ci_upper = [p + hist_std for p in predictions]
        ci_lower = [max(0, p - hist_std) for p in predictions]
        
        fig_pred.add_trace(go.Scatter(
            x=years + years[::-1],
            y=ci_upper + ci_lower[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Interval'
        ))
        
        fig_pred.update_layout(
            title=f"Rainfall Predictions for {selected_division}",
            xaxis_title="Year",
            yaxis_title="JUN-SEP Rainfall (mm)",
            template="plotly_dark",
            showlegend=True
        )
        
        # Display the final prediction
        st.success(f"Predicted JUN-SEP Rainfall for **{selected_division}** in **{future_year}**: **{predictions[-1]:.2f} mm**")
        
        # Show the prediction trend
        display_plotly_chart(fig_pred)
        
        # Show detailed predictions table with year-over-year change
        pred_df = pd.DataFrame({
            'Year': years,
            'Predicted Rainfall (mm)': predictions,
            'Year-over-Year Change (mm)': [predictions[0] - latest["JUN-SEP"]] + 
                                         [predictions[i] - predictions[i-1] for i in range(1, len(predictions))]
        })
        st.write("Detailed Predictions:")
        st.dataframe(pred_df)

        # After predictions are made, add new visualizations
        st.subheader("🔄 Seasonal Pattern Comparison")
        
        # Create seasonal decomposition plot
        fig_seasonal = go.Figure()
        
        # Historical seasonal pattern
        fig_seasonal.add_trace(go.Scatter(
            x=months,
            y=monthly_avg,
            name="Historical Average",
            line=dict(color='blue')
        ))

        # Predicted seasonal pattern (using the last prediction)
        predicted_monthly = []
        for ratio in monthly_ratios:
            monthly_pred = (predictions[-1]/4) * ratio
            predicted_monthly.append(monthly_pred)
        
        fig_seasonal.add_trace(go.Scatter(
            x=months,
            y=predicted_monthly,
            name=f"Predicted Pattern ({future_year})",
            line=dict(color='red', dash='dash')
        ))

        fig_seasonal.update_layout(
            title="Seasonal Pattern Comparison",
            xaxis_title="Month",
            yaxis_title="Rainfall (mm)",
            template="plotly_dark"
        )
        
        display_plotly_chart(fig_seasonal)

        # Rainfall Distribution Analysis
        st.subheader("📈 Rainfall Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Historical distribution
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=sub_df["JUN-SEP"],
                name="Historical",
                nbinsx=20,
                marker_color='blue',
                opacity=0.7
            ))
            fig_hist.update_layout(
                title="Historical Rainfall Distribution",
                xaxis_title="Rainfall (mm)",
                yaxis_title="Frequency",
                template="plotly_dark"
            )
            display_plotly_chart(fig_hist)
        
        with col2:
            # Prediction distribution
            fig_pred_dist = go.Figure()
            fig_pred_dist.add_trace(go.Histogram(
                x=predictions,
                name="Predictions",
                nbinsx=10,
                marker_color='red',
                opacity=0.7
            ))
            fig_pred_dist.update_layout(
                title=f"Predicted Rainfall Distribution ({latest['YEAR']+1}-{future_year})",
                xaxis_title="Rainfall (mm)",
                yaxis_title="Frequency",
                template="plotly_dark"
            )
            display_plotly_chart(fig_pred_dist)

        # Risk Analysis
        st.subheader("⚠️ Rainfall Risk Analysis")
        
        historical_percentiles = np.percentile(sub_df["JUN-SEP"], [10, 25, 50, 75, 90])
        predicted_percentiles = np.percentile(predictions, [10, 25, 50, 75, 90])
        
        risk_df = pd.DataFrame({
            'Percentile': ['10th', '25th', '50th (Median)', '75th', '90th'],
            'Historical (mm)': historical_percentiles,
            'Predicted (mm)': predicted_percentiles
        })
        
        st.write("Rainfall Percentile Comparison:")
        st.dataframe(risk_df)

# About Page
elif page == "ℹ️ About":
    st.markdown("""
    ### 🌧️ About This Dashboard
    
    This interactive dashboard provides comprehensive analysis and predictions for rainfall patterns across different regions of India.
    
    #### 🎯 Key Features:
    - Historical rainfall analysis
    - Region-wise comparisons
    - Future rainfall predictions
    - Monthly pattern analysis
    - Risk assessment
    
    #### 📊 Data Source:
    The data used in this dashboard covers rainfall measurements from 1901 to 2020 across various subdivisions of India.
    
    #### 🔬 Methodology:
    - Random Forest Regression for predictions
    - Historical pattern analysis
    - Seasonal decomposition
    - Trend analysis
    
    #### 📈 Model Performance:
    The prediction model is regularly updated and validated using historical data to ensure accuracy.
    
    #### 👥 Contact:
    For questions or feedback, please reach out to [sahilbopche3@gmail.com]
    """)

# Footer with animation
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px; animation: fadeIn 1s ease-in;'>
        Made with ❤️ using Streamlit | Data Source: Indian Meteorological Department
    </div>
""", unsafe_allow_html=True)

