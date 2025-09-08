# =========================================
# Climate Change Dataset Analysis Dashboard
# By: Mr. Kefuoe Sole
# Dataset: Climate Change Dataset
# =========================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import plotly.express as px
import seaborn as sns

sns.set(style="whitegrid")

# Streamlit Page Config
st.set_page_config(
    page_title="By: Kefuoe Sole - Climate Change Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar - Profile & Filters
st.sidebar.header("Filters")
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=2000,
    max_value=2024,
    value=(2000, 2024)
)

st.sidebar.header("Who Am I ?")
st.sidebar.markdown(
    """
    **Names:** Mr. Kefuoe Sole  

    **From:** Botho University  

    **Affiliation:**  
    - MSc Information Systems Management (Pursuing), BSc in Computing (General)  
    - CCNA, HCIA, OCI, NDE, ALX  
    """
)

# Domains dropdown
domains = [
    "Data Analyst", "Researcher", "Programmer", "Web Development",
    "Software Engineering", "Cybersecurity", "Artificial Intelligence", "Cloud Computing"
]
selected_domain = st.sidebar.selectbox("**Domains Interested**", domains)

# Custom Top Navbar
st.markdown("""
<!-- Load Font Awesome -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
.top-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    background-color: #e6f2ff;
    padding: 10px 20px;
    border-radius: 8px;
    font-size: 16px;
}
.navbar-left {
    font-weight: bold;
    color: #003366;
    margin-bottom: 8px;
}
.navbar-right {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
}
.navbar-right a {
    display: flex;
    align-items: center;
    color: black;
    text-decoration: none;
    transition: transform 0.2s, color 0.2s;
}
.navbar-right a:hover {
    transform: scale(1.1);
    color: darkblue;
}
.navbar-right i {
    margin-right: 6px;
}
</style>

<div class="top-navbar">
    <div class="navbar-left">
        🌍 Mr. Kefuoe Sole | Data Analyst / Researcher / Software Developer / Cybersecurity Consultant
    </div>
    <div class="navbar-right">
        <a href="https://github.com/1923k"><i class="fab fa-github"></i> GitHub</a>
        <a href="https://www.linkedin.com/in/kefuoe-sole-0797061ba/"><i class="fab fa-linkedin"></i> LinkedIn</a>
        <a href="mailto:soolekefuoe2@gmail.com"><i class="fas fa-envelope"></i> soolekefuoe2@gmail.com</a>
        <a href="tel:+26650996609"><i class="fas fa-phone"></i> (+266) 50996609 / 58183915</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Page Title
st.title("Climate Change Dataset Analysis (2000–2024)")
st.markdown(
    "Tracking **Temperature**, **CO2 Emissions**, **Sea Level Rise**, "
    "and **Environmental Trends** for awareness and decision-making."
)

file_id = "1iT3zJQHLDVvWE3haqG36E9Fb2La59bHi"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

try:
    df = pd.read_csv(url)
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("File not found. Please check the Google Drive file ID or link.")
    df = pd.DataFrame()  # create empty DataFrame to avoid further errors
except pd.errors.EmptyDataError:
    st.error("The file is empty. Please check your dataset.")
    df = pd.DataFrame()
except pd.errors.ParserError:
    st.error("Error parsing the CSV file. Ensure it is properly formatted.")
    df = pd.DataFrame()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the dataset: {e}")
    df = pd.DataFrame()

# Ensure Year is integer
df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)

# Drop rows with missing data
df = df.dropna(subset=[
    'Year', 'Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)',
    'Renewable Energy (%)', 'Forest Area (%)',
    'Extreme Weather Events', 'Sea Level Rise (mm)'
])

# Filter dataset by year range
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# If no data, show warning
if df_filtered.empty:
    st.warning("No data available for the selected year range. Please adjust the slider.")
else:
    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Temperature (°C)", f"{df_filtered['Avg Temperature (°C)'].mean():.2f}")
    col2.metric("Avg CO2 Emissions (Tons/Capita)", f"{df_filtered['CO2 Emissions (Tons/Capita)'].mean():.2f}")
    col3.metric("Avg Extreme Weather Events", f"{df_filtered['Extreme Weather Events'].mean():.2f}")

    # Result 1: Global Temperature Trend
    st.subheader("Result 1: Global Average Temperature Trend")
    try:
        fig_temp = px.scatter(
            df_filtered, x='Year', y='Avg Temperature (°C)',
            trendline="ols",
            hover_data={'Year': True, 'Avg Temperature (°C)': True},
            labels={'Year': 'Year (2000–2024)', 'Avg Temperature (°C)': 'Avg Temp (°C)'},
            title="Global Average Temperature Trend"
        )

        # Get regression results
        results_temp = px.get_trendline_results(fig_temp)
        slope_temp = results_temp.iloc[0]["px_fit_results"].params[1]
        intercept_temp = results_temp.iloc[0]["px_fit_results"].params[0]

        # Update regression line hover
        for trace in fig_temp.data:
            if trace.name == "ols":
                trace.hovertemplate = (
                    "Year: %{x}<br>"
                    "Predicted Temp: %{y:.2f} °C<br>"
                    f"Slope: {slope_temp:.4f} °C/year<br>"
                    f"Intercept: {intercept_temp:.2f} °C"
                )

        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown(
            f"**Interpretation:** The global average temperature has increased steadily, "
            f"with a warming rate of ~**{slope_temp:.4f} °C per year**."
        )
    except Exception as e:
        st.error(f"Could not generate Global Temperature Trend: {e}")

    # Result 2: CO2 vs Temperature
    st.subheader("Result 2: CO2 Emissions vs Avg Temperature")
    try:
        y = df_filtered['Avg Temperature (°C)']
        X = df_filtered[['CO2 Emissions (Tons/Capita)']]
        model_co2 = LinearRegression().fit(X, y)

        fig_co2 = px.scatter(
            df_filtered, x='CO2 Emissions (Tons/Capita)', y='Avg Temperature (°C)',
            trendline="ols",
            hover_data={'CO2 Emissions (Tons/Capita)': True, 'Avg Temperature (°C)': True},
            title="CO2 Emissions vs Avg Temperature"
        )
        st.plotly_chart(fig_co2, use_container_width=True)

        st.markdown(
            f"**Interpretation:** Higher CO2 emissions are strongly linked to higher temperatures. "
            f"Each additional 1 ton/capita CO2 relates to ~**{model_co2.coef_[0]:.4f} °C** increase."
        )
    except Exception as e:
        st.error(f"Could not generate CO2 vs Temperature: {e}")

    # Result 3: Sea Level Rise
    st.subheader("Result 3: Sea Level Rise Over Time")
    try:
        fig_sea = px.scatter(
            df_filtered, x='Year', y='Sea Level Rise (mm)',
            trendline="ols",
            hover_data={'Year': True, 'Sea Level Rise (mm)': True},
            labels={'Year': 'Year (2000–2024)', 'Sea Level Rise (mm)': 'Sea Level (mm)'},
            title="Sea Level Rise Over Time"
        )

        # Get regression results
        results_sea = px.get_trendline_results(fig_sea)
        slope_sea = results_sea.iloc[0]["px_fit_results"].params[1]
        intercept_sea = results_sea.iloc[0]["px_fit_results"].params[0]

        # Update regression line hover
        for trace in fig_sea.data:
            if trace.name == "ols":
                trace.hovertemplate = (
                    "Year: %{x}<br>"
                    "Predicted Sea Level: %{y:.2f} mm<br>"
                    f"Slope: {slope_sea:.4f} mm/year<br>"
                    f"Intercept: {intercept_sea:.2f} mm"
                )

        st.plotly_chart(fig_sea, use_container_width=True)
        st.markdown(
            f"**Interpretation:** Sea levels are rising by ~**{slope_sea:.4f} mm/year**, "
            "posing threats to coastal areas."
        )
    except Exception as e:
        st.error(f"Could not generate Sea Level Rise: {e}")

    # Result 4: Extreme Weather vs CO2
    st.subheader("Result 4: Extreme Weather Events vs CO2 Emissions")
    try:
        fig_extreme = px.scatter(
            df_filtered, x='CO2 Emissions (Tons/Capita)', y='Extreme Weather Events',
            trendline="ols",
            hover_data={'CO2 Emissions (Tons/Capita)': True, 'Extreme Weather Events': True},
            title="Extreme Weather Events vs CO2 Emissions"
        )
        st.plotly_chart(fig_extreme, use_container_width=True)

        corr = df_filtered['CO2 Emissions (Tons/Capita)'].corr(df_filtered['Extreme Weather Events'])
        st.markdown(
            f"**Interpretation:** Extreme weather events increase with CO2 emissions. "
            f"The correlation is **{corr:.2f}**, showing a strong positive relationship."
        )
    except Exception as e:
        st.error(f"Could not generate Extreme Weather Events plot: {e}")

    # Result 5: Renewable Energy & Forest Area
    st.subheader("Result 5: Renewable Energy & Forest Area vs CO2 Emissions")
    try:
        coef_renew = LinearRegression().fit(df_filtered[['Renewable Energy (%)']], df_filtered['CO2 Emissions (Tons/Capita)']).coef_[0]
        coef_forest = LinearRegression().fit(df_filtered[['Forest Area (%)']], df_filtered['CO2 Emissions (Tons/Capita)']).coef_[0]

        fig_renew = px.scatter(
            df_filtered, x='Renewable Energy (%)', y='CO2 Emissions (Tons/Capita)',
            trendline="ols",
            hover_data={'Renewable Energy (%)': True, 'CO2 Emissions (Tons/Capita)': True},
            title="Renewable Energy vs CO2 Emissions"
        )
        fig_forest = px.scatter(
            df_filtered, x='Forest Area (%)', y='CO2 Emissions (Tons/Capita)',
            trendline="ols",
            hover_data={'Forest Area (%)': True, 'CO2 Emissions (Tons/Capita)': True},
            title="Forest Area vs CO2 Emissions"
        )

        col1, col2 = st.columns(2)
        col1.plotly_chart(fig_renew, use_container_width=True)
        col2.plotly_chart(fig_forest, use_container_width=True)

        st.markdown(
            f"**Interpretation:** More renewable energy use reduces CO2 by ~**{abs(coef_renew):.4f} tons/capita per % increase**. "
            f"Expanding forest area cuts emissions by ~**{abs(coef_forest):.4f} tons/capita per % increase**."
        )
    except Exception as e:
        st.error(f"Could not generate Renewable Energy & Forest Area plots: {e}")
