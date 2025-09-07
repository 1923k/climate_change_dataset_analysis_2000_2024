# =========================
# Climate Change Dataset Analysis Dashboard
# By: Mr. Kefuoe Sole 
# Dataset : Climate Change Dataset
# =========================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px
import seaborn as sns

sns.set(style="whitegrid")

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="Climate Change Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Sidebar - Profile & Filters
# ---------------------------

st.sidebar.header("Filters")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=2000,
    max_value=2024,
    value=(2000, 2024)
)

st.sidebar.header("About")
st.sidebar.markdown(
    "**Name:** Mr. Kefuoe Sole  \n"
    "**Domains:** Data Analyst / Researcher / Programmer / Web Development / Software Engineering / Cybersecurity / Artificial Intelligence / Cloud Computing /   \n"
    "**Affiliation:** Pursuing MSc Information Systems Management / BSc in Computing: General / CCNA / HCIA / OCI / NDE / ALX   "
)

# ---------------------------
# Page Title
# ---------------------------
st.title("Climate Change Dataset Analysis (2000-2024)")
st.markdown(
    "Tracking **Temperature**, **CO2 Emissions**, **Sea Level Rise**, "
    "and **Environmental Trends** for awareness and decision-making."
)

# ---------------------------
# Load Dataset
# ---------------------------
file_path = "climate_change_dataset.csv"
df = pd.read_csv(file_path)

# Ensure Year is integer
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)

# Drop rows with missing critical data
df = df.dropna(subset=[
    'Year', 'Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)',
    'Renewable Energy (%)', 'Forest Area (%)', 'Extreme Weather Events',
    'Sea Level Rise (mm)'
])

# Filter dataset by selected year range
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# ---------------------------
# Check if filtered dataset is empty
# ---------------------------
if df_filtered.empty:
    st.warning("No data available for the selected year range. Please adjust the slider.")
else:

    # ---------------------------
    # Key Metrics
    # ---------------------------
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Temperature (°C)", f"{df_filtered['Avg Temperature (°C)'].mean():.2f}")
    col2.metric("Avg CO2 Emissions (Tons/Capita)", f"{df_filtered['CO2 Emissions (Tons/Capita)'].mean():.2f}")
    col3.metric("Average Extreme Weather Events", f"{df_filtered['Extreme Weather Events'].mean():.2f}")

    # ---------------------------
    # 1. Global Temperature Trend
    # ---------------------------
    st.subheader("Result 1: Global Average Temperature Trend")
    try:
        X = df_filtered[['Year']]
        y = df_filtered['Avg Temperature (°C)']
        model_temp = LinearRegression().fit(X, y)
        warming_rate = model_temp.coef_[0]

        fig_temp = px.scatter(
            df_filtered,
            x='Year',
            y='Avg Temperature (°C)',
            trendline="ols",
            hover_data={'Year': True, 'Avg Temperature (°C)': True},
            title="Global Average Temperature Trend (2000-2024)"
        )
        fig_temp.update_layout(xaxis=dict(range=[2000, 2024]))
        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown(f"**Interpretation:** The global average temperature has been increasing steadily, "
                    f"with an estimated warming rate of **{warming_rate:.4f} °C per year**. "
                    "This trend indicates ongoing global warming.")

    except Exception as e:
        st.error(f"Could not generate Global Temperature Trend plot: {e}")

    # ---------------------------
    # 2. CO2 Emissions vs Temperature
    # ---------------------------
    st.subheader("Result 2: CO2 Emissions vs Avg Temperature")
    try:
        model_co2 = LinearRegression().fit(df_filtered[['CO2 Emissions (Tons/Capita)']], y)
        fig_co2 = px.scatter(
            df_filtered,
            x='CO2 Emissions (Tons/Capita)',
            y='Avg Temperature (°C)',
            trendline="ols",
            hover_data={'CO2 Emissions (Tons/Capita)': True, 'Avg Temperature (°C)': True},
            title="CO2 Emissions vs Avg Temperature"
        )
        st.plotly_chart(fig_co2, use_container_width=True)
        st.markdown(f"**Interpretation:** There is a positive relationship between CO2 emissions and global temperature. "
                    f"Each 1 ton/capita increase in CO2 is associated with an increase of approximately **{model_co2.coef_[0]:.4f} °C** "
                    "in average temperature.")
    except Exception as e:
        st.error(f"Could not generate CO2 vs Temperature plot: {e}")

    # ---------------------------
    # 3. Sea Level Rise Over Time
    # ---------------------------
    st.subheader("Result 3: Sea Level Rise Over Time")
    try:
        fig_sea = px.scatter(
            df_filtered,
            x='Year',
            y='Sea Level Rise (mm)',
            trendline="ols",
            hover_data={'Year': True, 'Sea Level Rise (mm)': True},
            labels={'Sea Level Rise (mm)': 'Sea Level Rise (mm)', 'Year': 'Year'},
            title="Sea Level Rise Over Time"
        )
        fig_sea.update_traces(marker=dict(size=6))
        fig_sea.update_layout(
            xaxis=dict(range=[int(df['Year'].min()), int(df['Year'].max())],
                       showgrid=True, gridwidth=1, gridcolor='lightgrey'),
            yaxis=dict(title='Sea Level Rise (mm)',
                       showgrid=True, gridwidth=1, gridcolor='lightgrey'),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_sea, use_container_width=True)

        # Safely get regression slope
        results = px.get_trendline_results(fig_sea)
        slope = 0
        if not results.empty:
            sea_model = results.iloc[0]["px_fit_results"]
            if len(sea_model.params) > 1:
                slope = sea_model.params[1]

        st.markdown(f"**Interpretation:** Sea levels are rising at approximately **{slope:.4f} mm/year**. "
                    "This increasing trend poses risks for coastal areas and communities.")
    except Exception as e:
        st.error(f"Could not generate Sea Level Rise plot: {e}")

    # ---------------------------
    # 4. Extreme Weather Events vs CO2
    # ---------------------------
    st.subheader("Result 4: Extreme Weather Events vs CO2 Emissions")
    try:
        corr = df_filtered['CO2 Emissions (Tons/Capita)'].corr(df_filtered['Extreme Weather Events'])
        fig_extreme = px.scatter(
            df_filtered,
            x='CO2 Emissions (Tons/Capita)',
            y='Extreme Weather Events',
            hover_data={'CO2 Emissions (Tons/Capita)': True, 'Extreme Weather Events': True},
            title="Extreme Weather Events vs CO2 Emissions"
        )
        st.plotly_chart(fig_extreme, use_container_width=True)
        st.markdown(f"**Interpretation:** There is a **moderate positive correlation ({corr:.2f})** between CO2 emissions "
                    "and extreme weather events, suggesting that higher emissions contribute to more frequent extreme events.")
    except Exception as e:
        st.error(f"Could not generate Extreme Weather Events plot: {e}")

    # ---------------------------
    # 5. Renewable Energy & Forest Area vs CO2 Emissions
    # ---------------------------
    st.subheader("Result 5: Renewable Energy & Forest Area vs CO2 Emissions")
    try:
        coef_renew = LinearRegression().fit(df_filtered[['Renewable Energy (%)']], df_filtered['CO2 Emissions (Tons/Capita)']).coef_[0]
        coef_forest = LinearRegression().fit(df_filtered[['Forest Area (%)']], df_filtered['CO2 Emissions (Tons/Capita)']).coef_[0]

        fig_renew = px.scatter(
            df_filtered,
            x='Renewable Energy (%)',
            y='CO2 Emissions (Tons/Capita)',
            hover_data={'Renewable Energy (%)': True, 'CO2 Emissions (Tons/Capita)': True},
            trendline="ols",
            title="Renewable Energy vs CO2 Emissions"
        )
        fig_forest = px.scatter(
            df_filtered,
            x='Forest Area (%)',
            y='CO2 Emissions (Tons/Capita)',
            hover_data={'Forest Area (%)': True, 'CO2 Emissions (Tons/Capita)': True},
            trendline="ols",
            title="Forest Area vs CO2 Emissions"
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_renew, use_container_width=True)
        col2.plotly_chart(fig_forest, use_container_width=True)

        st.markdown(f"**Interpretation:** Increasing renewable energy usage reduces CO2 emissions by approximately **{abs(coef_renew):.4f} tons/capita per % increase**. "
                    f"Similarly, increasing forest area contributes to a reduction of **{abs(coef_forest):.4f} tons/capita per % increase**, highlighting the importance of green initiatives for climate mitigation.")
    except Exception as e:
        st.error(f"Could not generate Renewable Energy or Forest Area plots: {e}")
