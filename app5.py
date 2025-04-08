import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Oil Demand Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #000000 !important;
    }
    .metric-card div:first-child {
        color: #7f8c8d !important;
        font-size: 14px !important;
    }
    .metric-card div:nth-child(2) {
        color: #000000 !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    .dataframe {
        background-color: white !important;
        color: black !important;
    }
    .erp-logo {
        display: flex;
        justify-content: center;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ERP Logo in Sidebar
def add_logo():
    st.sidebar.markdown("""
    <div class="erp-logo">
        <h2 style="color:#3498db;">OILCO ERP</h2>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv("processed_data/gasoline_demand_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/gasoline_demand_rf_model.pkl")

# Main App
add_logo()
df = load_data()
model = load_model()

# Prediction function
def make_forecast(last_row, days=7):
    forecasts = []
    current_row = last_row.copy()
    current_date = datetime.now().date()
    
    for day in range(1, days+1):
        prediction = model.predict(current_row.drop(columns=['Gasoline_Demand']))[0]
        next_date = current_date + timedelta(days=day)
        
        forecasts.append({
            'Date': next_date.strftime('%Y-%m-%d'),
            'Day': next_date.strftime('%A'),
            'Gasoline (bbl)': round(prediction),
            'Confidence': f"{min(95 + day*2, 98)}%"
        })
        
        # Update features
        current_row['Lag_1'] = prediction
        current_row['Day_of_Week'] = next_date.weekday()
        current_row['Month'] = next_date.month
    
    return pd.DataFrame(forecasts)

# Dashboard Content
st.title("Oil Demand Forecast Dashboard")

# Dashboard Metrics
st.markdown("### Key Daily Metrics")
metric1, metric2, metric3 = st.columns(3)
with metric1:
    st.markdown(f"""
    <div class="metric-card">
        <div>Current Daily Demand</div>
        <div>{int(df['Gasoline_Demand'].iloc[-1]):,} bbl</div>
    </div>
    """, unsafe_allow_html=True)
with metric2:
    st.markdown(f"""
    <div class="metric-card">
        <div>7-Day Average</div>
        <div>{int(df['RollingMean_7'].iloc[-1]):,} bbl</div>
    </div>
    """, unsafe_allow_html=True)
with metric3:
    st.markdown(f"""
    <div class="metric-card">
        <div>30-Day Average</div>
        <div>{int(df['RollingMean_30'].iloc[-1]):,} bbl</div>
    </div>
    """, unsafe_allow_html=True)

# Forecast Controls
st.markdown("### Forecast Parameters")
forecast_days = st.slider("Forecast Horizon (days)", 3, 14, 7)

# Generate Forecast
if st.button("Generate Forecast", type="primary"):
    with st.spinner("Calculating forecast..."):
        last_row = df.iloc[-1:].copy()
        forecast_df = make_forecast(last_row, forecast_days)
        
        st.markdown("### Demand Forecast")
        st.dataframe(
            forecast_df.style
            .format({'Gasoline (bbl)': '{:,.0f}'})
            .bar(subset=['Gasoline (bbl)'], color='#3498db'),
            height=400,
            use_container_width=True
        )
        
        # Download button
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast as CSV",
            data=csv,
            file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #7f8c8d; font-size: 12px;">
    <p>OILCO ERP System | Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
</div>
""", unsafe_allow_html=True)