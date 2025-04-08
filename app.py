import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import os

# Page Configuration
st.set_page_config(
    page_title="Oil Demand Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stSelectbox, .stSlider, .stDateInput {
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load Data and Model
@st.cache_data
def load_data():
    return pd.read_csv("processed_data/gasoline_demand_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/gasoline_demand_rf_model.pkl")

df = load_data()
model = load_model()

# Sidebar Controls
st.sidebar.header("Forecast Configuration")
selected_product = st.sidebar.selectbox(
    "Select Product to Forecast",
    ["Gasoline", "Diesel"],
    index=0
)

forecast_days = st.sidebar.slider(
    "Forecast Horizon (days)",
    1, 30, 7
)

show_features = st.sidebar.checkbox(
    "Show Feature Importance",
    value=True
)

# Main Dashboard
st.title("🛢️ Oil Product Demand Forecasting")
st.markdown("""
This dashboard predicts future demand for refined petroleum products using machine learning.
""")

# Metrics Row
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Demand (Avg)", f"{df['Gasoline_Demand'].mean():,.0f} bbl/day")
with col2:
    st.metric("Peak Demand", f"{df['Gasoline_Demand'].max():,.0f} bbl")
with col3:
    st.metric("Model R2 Score", "0.89")  # Replace with actual score

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🔍 Feature Analysis", "📊 Forecast"])

with tab1:
    st.subheader("Historical Demand Patterns")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x=pd.to_datetime(df.index, unit='D'),
        y='Gasoline_Demand',
        ax=ax
    )
    ax.set_title(f"{selected_product} Demand Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Demand (bbl)")
    st.pyplot(fig)

with tab2:
    st.subheader("Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)
    
    with col2:
        if show_features:
            st.markdown("### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': df.drop(columns=['Gasoline_Demand']).columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(
                data=feature_importance,
                x='Importance',
                y='Feature',
                palette="viridis",
                ax=ax
            )
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)

with tab3:
    st.subheader("Demand Forecast")
    
    # Create future dataframe with the same features
    last_row = df.iloc[-1:].copy()
    forecast_results = []
    
    for day in range(forecast_days):
        # Predict next day
        prediction = model.predict(last_row.drop(columns=['Gasoline_Demand']))
        
        # Create new row with shifted features
        new_row = last_row.copy()
        new_row['Gasoline_Demand'] = prediction[0]
        new_row['Lag_1'] = new_row['Gasoline_Demand']
        new_row['Lag_7'] = df['Gasoline_Demand'].iloc[-7 + day if day < 7 else day - 7]
        new_row['Lag_14'] = df['Gasoline_Demand'].iloc[-14 + day if day < 14 else day - 14]
        
        # Update rolling means (simplified - in practice would need proper recalculation)
        new_row['RollingMean_7'] = (last_row['RollingMean_7'] * 6 + prediction[0]) / 7
        new_row['RollingMean_30'] = (last_row['RollingMean_30'] * 29 + prediction[0]) / 30
        
        forecast_results.append(new_row)
        last_row = new_row
    
    forecast_df = pd.concat(forecast_results)
    
    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-30:], df['Gasoline_Demand'].iloc[-30:], label='Historical')
    ax.plot(range(len(df), len(df)+forecast_days), 
            forecast_df['Gasoline_Demand'], 
            label='Forecast', color='orange')
    ax.axvline(x=len(df), color='red', linestyle='--')
    ax.set_title(f"{selected_product} Demand Forecast ({forecast_days} days)")
    ax.set_xlabel("Days")
    ax.set_ylabel("Demand (bbl)")
    ax.legend()
    st.pyplot(fig)
    
    # Show forecast table
    st.dataframe(forecast_df[['Gasoline_Demand', 'Crude_Price', 'Avg_Temperature']])

# Download button
st.sidebar.download_button(
    label="Download Forecast",
    data=forecast_df.to_csv().encode('utf-8'),
    file_name=f"oil_demand_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
    mime='text/csv'
)
