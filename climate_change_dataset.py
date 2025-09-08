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

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(
    page_title="By: Kefuoe Sole - Climate Change Dashboard",
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

st.sidebar.header("Who Am I ?")
st.sidebar.markdown("""
**Names:** Mr. Kefuoe Sole  

**From:** Botho University  

**Affiliation:**  
- MSc Information Systems Management (Pursuing), BSc in Computing (General)  
- CCNA, HCIA, OCI, NDE, ALX  
""")

domains = [
    "Data Analyst", "Researcher", "Programmer", "Web Development",
    "Software Engineering", "Cybersecurity", "Artificial Intelligence", "Cloud Computing"
]
selected_domain = st.sidebar.selectbox("**Domains Interested**", domains)

# ---------------------------
# Custom Top Navbar
# ---------------------------
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
.top-navbar {
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    background-color: #e6f2ff; padding: 10px 20px; border-radius: 8px; font-size: 16px;
}
.navbar-left { font-weight: bold; color: #003366; margin-bottom: 8px; }
.navbar-right { display: flex; flex-wrap: wrap; gap: 15px; }
.navbar-right a { display: flex; align-items: center; color: black; text-decoration: none; transition: transform 0.2s, color 0.2s; }
.navbar-right a:hover { transform: scale(1.1); color: darkblue; }
.navbar-right i { margin-right: 6px; }
.result-box {
    background-color: #f0f8ff; padding: 15px; border-radius: 8px; border: 1px solid #a6c8ff;
    font-size: 14px; line-height: 1.6;
}
.result-title { font-weight: bold; font-size: 16px; color: #003366; margin-bottom: 8px; }
.result-year { color: #ff4500; font-weight: bold; }
</style>
<div class="top-navbar">
    <div class="navbar-left">
        <i class="fa fa-user"></i> Mr. Kefuoe Sole | Data Analyst / Researcher / Software Developer / Cybersecurity Consultant
    </div>
    <div class="navbar-right">
        <a href="https://github.com/1923k"><i class="fab fa-github"></i> GitHub</a>
        <a href="https://www.linkedin.com/in/kefuoe-sole-0797061ba/"><i class="fab fa-linkedin"></i> LinkedIn</a>
        <a href="mailto:soolekefuoe2@gmail.com"><i class="fas fa-envelope"></i> soolekefuoe2@gmail.com</a>
        <a href="tel:+26650996609"><i class="fas fa-phone"></i> (+266) 50996609 / 58183915</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Page Title
# ---------------------------
st.title("Climate Change Dataset Analysis (2000–2024)")
st.markdown(
    "Tracking **Temperature**, **CO2 Emissions**, **Sea Level Rise**, "
    "and **Environmental Trends** for awareness and decision-making."
)

# ---------------------------
# Load Dataset from Google Drive
# ---------------------------
file_id = "1iT3zJQHLDVvWE3haqG36E9Fb2La59bHi"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

try:
    df = pd.read_csv(url)
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("File not found. Please check the Google Drive file ID or link.")
    df = pd.DataFrame()
except pd.errors.EmptyDataError:
    st.error("The file is empty. Please check your dataset.")
    df = pd.DataFrame()
except pd.errors.ParserError:
    st.error("Error parsing the CSV file. Ensure it is properly formatted.")
    df = pd.DataFrame()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the dataset: {e}")
    df = pd.DataFrame()

# ---------------------------
# Clean and Prepare Dataset
# ---------------------------
if not df.empty:
    try:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        essential_cols = [
            'Year', 'Avg Temperature (°C)', 'CO2 Emissions (Tons/Capita)',
            'Renewable Energy (%)', 'Forest Area (%)',
            'Extreme Weather Events', 'Sea Level Rise (mm)'
        ]
        df = df.dropna(subset=essential_cols)
        df['Year'] = df['Year'].astype(int)
        df = df.sort_values(by='Year').reset_index(drop=True)
        st.success("Dataset cleaned and sorted by Year successfully!")
    except KeyError as ke:
        st.error(f"Dataset is missing expected column: {ke}")
        df = pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during dataset cleaning: {e}")
        df = pd.DataFrame()
else:
    st.warning("No dataset available to clean or sort.")

# ---------------------------
# Filter by Year Range
# ---------------------------
df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

if df_filtered.empty:
    st.warning("No data available for the selected year range. Please adjust the slider.")
else:
    # ---------------------------
    # Key Metrics
    # ---------------------------
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Temperature (°C)", f"{df_filtered['Avg Temperature (°C)'].mean():.2f}")
    col2.metric("Avg CO2 Emissions (Tons/Capita)", f"{df_filtered['CO2 Emissions (Tons/Capita)'].mean():.2f}")
    col3.metric("Avg Extreme Weather Events", f"{df_filtered['Extreme Weather Events'].mean():.2f}")

    # ---------------------------
    # Function to Display Chart with Trendline & Results
    # ---------------------------
    def display_chart_with_trendline(x_col, y_col, chart_title, y_label=None):
        y_label = y_label or y_col
        fig = px.scatter(df_filtered, x=x_col, y=y_col, trendline="ols", hover_data={x_col: True, y_col: True})
        results = px.get_trendline_results(fig)
        slope = results.iloc[0]["px_fit_results"].params[1]
        intercept = results.iloc[0]["px_fit_results"].params[0]
        min_val = df_filtered[y_col].min()
        max_val = df_filtered[y_col].max()
        mean_val = df_filtered[y_col].mean()
        fig.update_layout(title=chart_title)

        # Display chart and trendline info
        col_chart, col_info = st.columns([3, 1])
        col_chart.plotly_chart(fig, use_container_width=True)
        col_info.markdown(f"""
        <div class="result-box">
        <div class="result-title">{chart_title}</div>
        <b>Year Range:</b> <span class="result-year">{year_range[0]} – {year_range[1]}</span><br>
        <b>Trendline:</b> y = {slope:.4f}x + {intercept:.2f}<br>
        <b>Slope:</b> {slope:.4f}<br>
        <b>Intercept:</b> {intercept:.2f}<br>
        <b>Min {y_label}:</b> {min_val:.2f}<br>
        <b>Max {y_label}:</b> {max_val:.2f}<br>
        <b>Mean {y_label}:</b> {mean_val:.2f}
        </div>
        """, unsafe_allow_html=True)

        return slope

    # ---------------------------
    # Result 1: Global Temperature Trend
    # ---------------------------
    st.subheader("Result 1: Global Average Temperature Trend")
    slope_temp = display_chart_with_trendline('Year', 'Avg Temperature (°C)', 'Global Average Temperature Trend', 'Temperature (°C)')

    # ---------------------------
    # Result 2: CO2 vs Temperature
    # ---------------------------
    st.subheader("Result 2: CO2 Emissions vs Avg Temperature")
    slope_co2 = display_chart_with_trendline('CO2 Emissions (Tons/Capita)', 'Avg Temperature (°C)', 'CO2 Emissions vs Avg Temperature')

    # ---------------------------
    # Result 3: Sea Level Rise
    # ---------------------------
    st.subheader("Result 3: Sea Level Rise Over Time")
    slope_sea = display_chart_with_trendline('Year', 'Sea Level Rise (mm)', 'Sea Level Rise Over Time', 'Sea Level (mm)')

    # ---------------------------
    # Result 4: Extreme Weather vs CO2
    # ---------------------------
    st.subheader("Result 4: Extreme Weather Events vs CO2 Emissions")
    slope_extreme = display_chart_with_trendline('CO2 Emissions (Tons/Capita)', 'Extreme Weather Events', 'Extreme Weather Events vs CO2 Emissions')

    # ---------------------------
    # Result 5: Renewable Energy & Forest Area vs CO2
    # ---------------------------
    st.subheader("Result 5: Renewable Energy & Forest Area vs CO2 Emissions")
    slope_renew = display_chart_with_trendline('Renewable Energy (%)', 'CO2 Emissions (Tons/Capita)', 'Renewable Energy vs CO2 Emissions')
    slope_forest = display_chart_with_trendline('Forest Area (%)', 'CO2 Emissions (Tons/Capita)', 'Forest Area vs CO2 Emissions')
