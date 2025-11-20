import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import requests
from datetime import datetime

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - MODERN FUTURISTIC DESIGN
# ============================================================================
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
    
    /* Title Styling */
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
    
    /* Cards and Containers */
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
    
    /* Input Elements */
    [data-testid="stNumberInput"] input,
    [data-testid="stSlider"] input,
    [data-testid="stSelectbox"] select {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        color: #00d4ff !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stSlider"] input:focus,
    [data-testid="stSelectbox"] select:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }
    
    /* Checkboxes */
    [data-testid="stCheckbox"] {
        color: #00ff88;
    }
    
    /* Text */
    body, p, span, div {
        color: #e0e0ff;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(26, 31, 58, 0.7) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Metric Cards */
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
    
    /* Error Messages */
    [data-testid="stAlert"] {
        background: rgba(255, 100, 100, 0.1) !important;
        border: 1px solid rgba(255, 100, 100, 0.3) !important;
        border-radius: 10px !important;
        color: #ff6464 !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
        margin: 30px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.1);
    }
    
    /* Custom Section Headers */
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
    
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS & ENCODERS
# ============================================================================
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

# ============================================================================
# AUTO-DETECTION FUNCTIONS - FROM YOUR NOTEBOOK
# ============================================================================

def get_user_location_data():
    """Try multiple APIs for location detection - EXACTLY FROM YOUR NOTEBOOK"""
    apis = [
        {
            'name': 'ipapi.co',
            'url': 'https://ipapi.co/json',
            'parser': lambda d: {'city': d.get('city'), 'latitude': d.get('latitude'), 'longitude': d.get('longitude')}
        },
        {
            'name': 'ip-api.com',
            'url': 'http://ip-api.com/json',
            'parser': lambda d: {'city': d.get('city'), 'lat': d.get('lat'), 'lon': d.get('lon')}
        }
    ]
    
    for api in apis:
        try:
            response = requests.get(api['url'], timeout=5)
            if response.status_code == 200:
                data = response.json()
                location = api['parser'](data)
                if location['city']:
                    return location
        except:
            continue
    return None

def get_weather_data(latitude, longitude):
    """Fetch complete weather data from Open-Meteo API - EXACTLY FROM YOUR NOTEBOOK"""
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

# ============================================================================
# HEADER
# ============================================================================
st.title("CROP YIELD PREDICTION SYSTEM")

st.markdown("""
<div class="section-header">
    NEXT-GENERATION AGRICULTURAL AI INTELLIGENCE
</div>
""", unsafe_allow_html=True)

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Yield Model", "Decision Tree", "R² = 0.8687")

with col_info2:
    st.metric("Crop Model", "KNN Classifier", "Accuracy = 0.8714")

with col_info3:
    st.metric("Training Data", "181,385 Records", "Bangladesh")

st.markdown("---")

# ============================================================================
# MAIN CONTENT
# ============================================================================
models = load_models_and_encoders()

if models is None:
    st.error("Could not load models. Please ensure all .pkl files are in the same folder as app.py")
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="section-header">
        INPUT FARM PARAMETERS
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-detect location & weather
    location_data = get_user_location_data()
    weather_data = None
    
    if location_data:
        weather_data = get_weather_data(location_data.get('latitude') or location_data.get('lat'), 
                                       location_data.get('longitude') or location_data.get('lon'))
    
    # District Selection
    st.subheader(" DISTRICT")
    available_districts = list(models['district_encoder'].classes_)
    
    auto_district = location_data['city'] if location_data else None
    if auto_district and auto_district in available_districts:
        use_auto = st.checkbox(f"Use auto-detected: {auto_district}?", value=True, key="district_auto")
        selected_district = auto_district if use_auto else st.selectbox("Select District", available_districts)
    else:
        selected_district = st.selectbox("Select District", available_districts)
    
    # Area - Manual Input Only
    st.subheader(" CULTIVATED AREA (hectares)")
    area_hectares = st.number_input("Area", value=5.0, min_value=0.5, max_value=100.0, step=0.5)
    
    # Temperature - EXACTLY LIKE YOUR NOTEBOOK
    st.subheader(" TEMPERATURE CONDITIONS")
    
    if weather_data and all([weather_data.get('min_temp'), weather_data.get('avg_temp'), weather_data.get('max_temp')]):
        col_min, col_avg, col_max = st.columns(3)
        
        with col_min:
            use_min = st.checkbox(f"Auto: {weather_data['min_temp']:.1f}°C?", value=True, key="temp_min")
            min_temp = weather_data['min_temp'] if use_min else st.number_input("Min Temp (°C)", value=18.0, key="temp_min_manual")
        
        with col_avg:
            use_avg = st.checkbox(f"Auto: {weather_data['avg_temp']:.1f}°C?", value=True, key="temp_avg")
            avg_temp = weather_data['avg_temp'] if use_avg else st.number_input("Avg Temp (°C)", value=25.0, key="temp_avg_manual")
        
        with col_max:
            use_max = st.checkbox(f"Auto: {weather_data['max_temp']:.1f}°C?", value=True, key="temp_max")
            max_temp = weather_data['max_temp'] if use_max else st.number_input("Max Temp (°C)", value=32.0, key="temp_max_manual")
    else:
        col_min, col_avg, col_max = st.columns(3)
        with col_min:
            min_temp = st.number_input("Min Temp (°C)", value=18.0)
        with col_avg:
            avg_temp = st.number_input("Avg Temp (°C)", value=25.0)
        with col_max:
            max_temp = st.number_input("Max Temp (°C)", value=32.0)
    
    # Humidity - EXACTLY LIKE YOUR NOTEBOOK
    st.subheader(" HUMIDITY CONDITIONS (%)")
    
    if weather_data and all([weather_data.get('min_humidity'), weather_data.get('avg_humidity'), weather_data.get('max_humidity')]):
        col_min_hum, col_avg_hum, col_max_hum = st.columns(3)
        
        with col_min_hum:
            use_min_hum = st.checkbox(f"Auto: {weather_data['min_humidity']:.0f}%?", value=True, key="hum_min")
            min_humidity = weather_data['min_humidity'] if use_min_hum else st.number_input("Min Humidity (%)", value=40, min_value=0, max_value=100, key="hum_min_manual")
        
        with col_avg_hum:
            use_avg_hum = st.checkbox(f"Auto: {weather_data['avg_humidity']:.0f}%?", value=True, key="hum_avg")
            avg_humidity = weather_data['avg_humidity'] if use_avg_hum else st.number_input("Avg Humidity (%)", value=70, min_value=0, max_value=100, key="hum_avg_manual")
        
        with col_max_hum:
            use_max_hum = st.checkbox(f"Auto: {weather_data['max_humidity']:.0f}%?", value=True, key="hum_max")
            max_humidity = weather_data['max_humidity'] if use_max_hum else st.number_input("Max Humidity (%)", value=95, min_value=0, max_value=100, key="hum_max_manual")
    else:
        col_min_hum, col_avg_hum, col_max_hum = st.columns(3)
        with col_min_hum:
            min_humidity = st.number_input("Min Humidity (%)", value=40, min_value=0, max_value=100)
        with col_avg_hum:
            avg_humidity = st.number_input("Avg Humidity (%)", value=70, min_value=0, max_value=100)
        with col_max_hum:
            max_humidity = st.number_input("Max Humidity (%)", value=95, min_value=0, max_value=100)
    
    # Season
    st.subheader(" SEASON")
    available_seasons = list(models['season_encoder'].classes_)
    selected_season = st.selectbox("Select Season", available_seasons)

with col2:
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        AI PREDICTIONS
    </div>
    """, unsafe_allow_html=True)
    
    try:
        season_encoded = models['season_encoder'].transform([selected_season])[0]
        district_encoded = models['district_encoder'].transform([selected_district])[0]
        
        yield_features = np.array([[area_hectares, avg_temp, avg_humidity, max_temp, min_temp, max_humidity, min_humidity]])
        crop_features = np.array([[avg_temp, avg_humidity, max_temp, min_temp, max_humidity, min_humidity, season_encoded, district_encoded]])
        
        predicted_yield = models['yield'].predict(yield_features)[0]
        total_production = predicted_yield * area_hectares
        
        st.markdown("""
        <div class="result-box">
        <h3 style="color: #00ff88; margin-bottom: 10px;">RECOMMENDED CROP VARIETY</h3>
        """, unsafe_allow_html=True)
        
        if hasattr(models['crop'], 'predict_proba'):
            probas = models['crop'].predict_proba(crop_features)[0]
            top_1_index = np.argmax(probas)
            recommended_crop = models['crop_encoder'].classes_[top_1_index]
            confidence = probas[top_1_index] * 100
        else:
            pred_idx = models['crop'].predict(crop_features)[0]
            recommended_crop = models['crop_encoder'].classes_[pred_idx]
            confidence = 90.0
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: rgba(0, 255, 136, 0.05); border-radius: 10px; margin: 15px 0;">
            <div class="crop-name">{recommended_crop}</div>
            <div class="confidence-score">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="result-box">
        <h3 style="color: #00d4ff; margin-bottom: 20px;">YIELD FORECAST</h3>
        """, unsafe_allow_html=True)
        
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            st.metric("Yield", f"{predicted_yield:.2f}", "tons/hectare")
        with col_y2:
            st.metric("Total Production", f"{total_production:.0f}", "tons")
        
        st.write("")
        st.write("FARM SUMMARY")
        st.write(f"District: {selected_district}")
        st.write(f"Season: {selected_season}")
        st.write(f"Area: {area_hectares:.1f} hectares")
        st.write(f"Temperature: {min_temp:.1f}°C - {max_temp:.1f}°C (Avg: {avg_temp:.1f}°C)")
        st.write(f"Humidity: {min_humidity:.0f}% - {max_humidity:.0f}% (Avg: {avg_humidity:.0f}%)")
        
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")

st.markdown("---")
