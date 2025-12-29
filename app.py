import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import requests
from datetime import datetime

st.set_page_config(page_title="Crop Yield Prediction System", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    :root {
        --primary: #00d4ff;
        --secondary: #00ff88;
        --dark-bg: #0a0e27;
        --card-bg: #1a1f3a;
        --accent: #6366f1;
    }
    
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f2847 100%);
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stHeader"] {
        background: transparent;
        border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    }
    
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 3.5em !important;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        margin-bottom: 0.5em !important;
        letter-spacing: 2px;
    }
    
    h2 {
        color: #00d4ff;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
        padding-bottom: 10px;
        font-weight: 700;
        font-size: 1.8em;
        letter-spacing: 1px;
    }
    
    [data-testid="stColumn"] {
        background: rgba(26, 31, 58, 0.5);
        border-radius: 15px;
        border: 1px solid rgba(0, 212, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stColumn"]:hover {
        border-color: rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        transform: translateY(-5px);
    }
    
    [data-testid="stNumberInput"] input, [data-testid="stSlider"] input, [data-testid="stSelectbox"] select {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        color: #00d4ff !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    [data-testid="stNumberInput"] input:focus, [data-testid="stSlider"] input:focus, [data-testid="stSelectbox"] select:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }
    
    body, p, span, div {
        color: #e0e0ff;
    }
    
    [data-testid="stDataFrame"] {
        background: rgba(26, 31, 58, 0.7) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 136, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetricLabel"] {
        color: #00d4ff !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2.5em !important;
    }
    
    [data-testid="stAlert"] {
        background: rgba(255, 100, 100, 0.1) !important;
        border: 1px solid rgba(255, 100, 100, 0.3) !important;
        border-radius: 10px !important;
        color: #ff6464 !important;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
        margin: 30px 0;
    }
    
    .section-header {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 136, 0.05));
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .result-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 136, 0.05));
        border: 2px solid #00d4ff;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
    }
    
    .crop-name {
        color: #00ff88;
        font-size: 1.8em;
        font-weight: 900;
        letter-spacing: 2px;
    }
    
    .confidence-score {
        color: #00d4ff;
        font-size: 1.5em;
        font-weight: 700;
    }
    
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        border: 2px solid rgba(0, 212, 255, 0.3);
        margin: 15px 0;
    }
    
    .about-horizontal {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }
    
    .about-item {
        flex: 1;
        min-width: 250px;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 136, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .about-item h4 {
        color: #00d4ff;
        margin-bottom: 10px;
        font-size: 1.1em;
    }
    
    .about-item p {
        color: #e0e0ff;
        font-size: 0.95em;
        line-height: 1.6;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_encoders():
    """Load trained models and encoders from .pkl files"""
    try:
        yield_model = joblib.load('yield_model.pkl')
        crop_model = joblib.load('crop_model.pkl')
        le_crop = joblib.load('label_encoder_crop.pkl')
        le_season = joblib.load('label_encoder_season.pkl')
        le_district = joblib.load('label_encoder_district.pkl')
        
        return {
            'yield': yield_model,
            'crop': crop_model,
            'crop_encoder': le_crop,
            'season_encoder': le_season,
            'district_encoder': le_district
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def get_weather_data(latitude, longitude):
    """Fetch complete weather data from Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            current = data.get('current', {})
            daily = data.get('daily', {})
            
            weather = {
                'avg_temp': current.get('temperature_2m'),
                'min_temp': daily.get('temperature_2m_min', [None])[0] if daily.get('temperature_2m_min') else None,
                'max_temp': daily.get('temperature_2m_max', [None])[0] if daily.get('temperature_2m_max') else None,
                'avg_humidity': current.get('relative_humidity_2m'),
                'min_humidity': daily.get('relative_humidity_2m_min', [None])[0] if daily.get('relative_humidity_2m_min') else None,
                'max_humidity': daily.get('relative_humidity_2m_max', [None])[0] if daily.get('relative_humidity_2m_max') else None
            }
            return weather
    except:
        pass
    
    return None

def get_location_name(latitude, longitude):
    """Get city and country name from coordinates"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            city = data.get('address', {}).get('city') or data.get('address', {}).get('town') or data.get('address', {}).get('village') or 'Unknown'
            country = data.get('address', {}).get('country', 'Unknown')
            return city, country
    except:
        pass
    return 'Unknown', 'Unknown'

st.title("🌾 CROP YIELD PREDICTION SYSTEM")

st.markdown("""
<div class="image-container">
    <img src="https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=1200&q=80" style="width: 100%; height: 450px; object-fit: cover; border-radius: 15px;" alt="Agriculture">
</div>
""", unsafe_allow_html=True)

st.markdown("---")

with st.expander("About System"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="about-item">
            <h4>Yield Model</h4>
            <p><strong>Type:</strong> Decision Tree</p>
            <p><strong>R² Score:</strong> 0.8687</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="about-item">
            <h4>Crop Model</h4>
            <p><strong>Type:</strong> KNN Classifier</p>
            <p><strong>Accuracy:</strong> 0.8714</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="about-item">
            <h4>Training Data</h4>
            <p><strong>Records:</strong> 181,385</p>
            <p><strong>Location:</strong> Bangladesh</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 255, 136, 0.05)); border: 1px solid rgba(0, 212, 255, 0.2); border-radius: 12px; padding: 20px; margin: 15px 0;">
        <h4 style="color: #00d4ff; margin-bottom: 15px;">About This System</h4>
        <p style="color: #e0e0ff; line-height: 1.8; margin-bottom: 15px;">This advanced agricultural prediction system combines machine learning with real-time weather data to provide accurate crop yield forecasts and crop recommendations.</p>
        <p style="color: #e0e0ff; line-height: 1.8; margin-bottom: 15px;">The system is trained on comprehensive agricultural data from Bangladesh and uses proven machine learning algorithms to deliver reliable predictions for farmers and agricultural planners.</p>
        <p style="text-align: center; color: #00ff88; font-weight: bold; margin-top: 20px;">Built for Agricultural Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

models = load_models_and_encoders()
if models is None:
    st.error("Could not load models. Please ensure all .pkl files are in the same folder as app.py")
    st.stop()

st.markdown("""
<div class="section-header">
    How do you want to enter data?
</div>
""", unsafe_allow_html=True)

data_entry_mode = st.radio("Select your preference", options=["Auto-Detect from GPS Coordinates", "Manual Entry"], key="entry_mode")

st.markdown("---")

if data_entry_mode == "Auto-Detect from GPS Coordinates":
    st.markdown("""
    <div class="section-header">
        Auto-Detect Mode - Enter GPS Coordinates
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 GPS Coordinates")
        latitude = st.number_input("Latitude", value=23.8103, min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f", help="Enter your latitude (e.g., 23.8103 for Dhaka)")
    
    with col2:
        st.subheader("📍 GPS Coordinates")
        longitude = st.number_input("Longitude", value=90.4125, min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f", help="Enter your longitude (e.g., 90.4125 for Dhaka)")
    
    # Get location name and weather
    city, country = get_location_name(latitude, longitude)
    weather_data = get_weather_data(latitude, longitude)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Detected Location")
        st.write(f"**City:** {city}")
        st.write(f"**Country:** {country}")
        st.write(f"**Latitude:** {latitude:.4f}")
        st.write(f"**Longitude:** {longitude:.4f}")
        
        st.subheader("Weather Data")
        if weather_data:
            st.write(f"**Avg Temp:** {weather_data['avg_temp']:.1f}°C")
            st.write(f"**Min Temp:** {weather_data['min_temp']:.1f}°C")
            st.write(f"**Max Temp:** {weather_data['max_temp']:.1f}°C")
            st.write(f"**Avg Humidity:** {weather_data['avg_humidity']:.0f}%")
        else:
            st.warning("Could not fetch weather data")
    
    with col2:
        st.subheader("Manual Inputs Required")
        available_districts = list(models['district_encoder'].classes_)
        selected_district = st.selectbox("Select District", options=available_districts, key="district_auto")
        
        area_hectares = st.number_input("Cultivated Area (hectares)", value=100000.0, min_value=0.5, max_value=10000000.0, step=1000.0)
        
        available_seasons = list(models['season_encoder'].classes_)
        selected_season = st.selectbox("Select Season", options=available_seasons, key="season_auto")
    
    if weather_data:
        min_temp = weather_data['min_temp']
        avg_temp = weather_data['avg_temp']
        max_temp = weather_data['max_temp']
        min_humidity = weather_data['min_humidity']
        avg_humidity = weather_data['avg_humidity']
        max_humidity = weather_data['max_humidity']
    else:
        min_temp = 18.0
        avg_temp = 25.0
        max_temp = 32.0
        min_humidity = 40
        avg_humidity = 70
        max_humidity = 95

else:
    st.markdown("""
    <div class="section-header">
        Manual Entry Mode - Enter Your Farm Details
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        available_districts = list(models['district_encoder'].classes_)
        selected_district = st.selectbox("District", options=available_districts, key="district_manual")
        
        area_hectares = st.number_input("Cultivated Area (hectares)", value=100000.0, min_value=0.5, max_value=10000000.0, step=1000.0, key="area_manual")
        
        available_seasons = list(models['season_encoder'].classes_)
        selected_season = st.selectbox("Season", options=available_seasons, key="season_manual")
    
    with col2:
        st.subheader("Temperature Conditions (Celsius)")
        col_min, col_avg, col_max = st.columns(3)
        with col_min:
            min_temp = st.number_input("Min Temp", value=18.0, key="temp_min_manual")
        with col_avg:
            avg_temp = st.number_input("Avg Temp", value=25.0, key="temp_avg_manual")
        with col_max:
            max_temp = st.number_input("Max Temp", value=32.0, key="temp_max_manual")
        
        st.subheader("Humidity Conditions (Percent)")
        col_min_hum, col_avg_hum, col_max_hum = st.columns(3)
        with col_min_hum:
            min_humidity = st.number_input("Min Humidity", value=40, min_value=0, max_value=100, key="hum_min_manual")
        with col_avg_hum:
            avg_humidity = st.number_input("Avg Humidity", value=70, min_value=0, max_value=100, key="hum_avg_manual")
        with col_max_hum:
            max_humidity = st.number_input("Max Humidity", value=95, min_value=0, max_value=100, key="hum_max_manual")

st.markdown("---")

st.markdown("""
<div class="section-header">
    Get Predictions
</div>
""", unsafe_allow_html=True)

predict_button = st.button("Predict Crop and Yield", use_container_width=True, key="predict_btn")

if predict_button:
    try:
        season_encoded = models['season_encoder'].transform([selected_season])[0]
        district_encoded = models['district_encoder'].transform([selected_district])[0]
        
        yieldfeatures = np.array([area_hectares, avg_temp, avg_humidity, max_temp, min_temp, max_humidity, min_humidity])
        
        cropfeatures = np.array([avg_temp, avg_humidity, max_temp, min_temp, max_humidity, min_humidity, season_encoded, district_encoded])
        
        predicted_yield = models['yield'].predict(yieldfeatures.reshape(1, -1))[0]
        total_production = predicted_yield * area_hectares
        
        crop_pred_idx = models['crop'].predict(cropfeatures.reshape(1, -1))[0]
        recommended_crop = models['crop_encoder'].classes_[int(crop_pred_idx)]
        
        if hasattr(models['crop'], 'predict_proba'):
            probas = models['crop'].predict_proba(cropfeatures.reshape(1, -1))[0]
            confidence = f"{probas.max():.1%}"
        else:
            confidence = "NA"
        
        st.markdown("---")
        
        st.markdown("""
        <div class="result-box">
            <h3 style="color: #00ff88; margin-bottom: 10px;">Prediction Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recommended Crop", recommended_crop.upper(), f"Confidence: {confidence}")
        with col2:
            st.metric("Predicted Yield", f"{predicted_yield:.2f}", "tons/hectare")
        
        st.markdown("<div></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-box">
            <h3 style="color: #00d4ff; margin-bottom: 20px;">Production Forecast</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col_prod1, col_prod2, col_prod3 = st.columns(3)
        with col_prod1:
            st.metric("Total Production", f"{total_production:.0f}", "tons")
        with col_prod2:
            st.metric("Area", f"{area_hectares:.0f}", "hectares")
        with col_prod3:
            st.metric("Season", selected_season)
        
        st.markdown("<div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("Farm Summary")
        summary_df = pd.DataFrame(
            {
                "Parameter": [
                    "District",
                    "Season",
                    "Area (hectares)",
                    "Min Temp (°C)",
                    "Avg Temp (°C)",
                    "Max Temp (°C)",
                    "Min Humidity (%)",
                    "Avg Humidity (%)",
                    "Max Humidity (%)"
                ],
                "Value": [
                    selected_district,
                    selected_season,
                    f"{area_hectares:.0f}",
                    f"{min_temp:.1f}",
                    f"{avg_temp:.1f}",
                    f"{max_temp:.1f}",
                    f"{min_humidity}",
                    f"{avg_humidity}",
                    f"{max_humidity}"
                ]
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
