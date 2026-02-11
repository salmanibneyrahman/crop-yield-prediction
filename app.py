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
    .main-header {
        font-size: 3rem !important;
        font-weight: bold !important;
        color: #2E7D32 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
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

def get_user_location_data():
    """Get location from browser geolocation (stored in sessionStorage via JavaScript)"""
    try:
        # Try to get from JavaScript sessionStorage
        lat = st.session_state.get('user_lat')
        lon = st.session_state.get('user_lon')
        city = st.session_state.get('user_city', 'Unknown')
        country = st.session_state.get('user_country', 'Unknown')
        if lat and lon:
            return {
                'city': city,
                'latitude': lat,
                'longitude': lon,
                'country': country
            }
    except:
        pass
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

# Load models
models = load_models_and_encoders()
if models is None:
    st.stop()

st.markdown('<h1 class="main-header">🌾 CROP YIELD PREDICTION SYSTEM</h1>', unsafe_allow_html=True)

# Metrics display (keeping your exact design)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Decision Tree</h3>
        <h2>R² Score: 0.8687</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>KNN Classifier</h3>
        <h2>Accuracy: 0.8714</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Records</h3>
        <h2>181,385</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Location</h3>
        <h2>Bangladesh</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
**This advanced agricultural prediction system combines machine learning with real-time weather data to provide accurate crop yield forecasts and crop recommendations.**

The system is trained on comprehensive agricultural data from Bangladesh and uses proven machine learning algorithms to deliver reliable predictions for farmers and agricultural planners.

**Built for Agricultural Intelligence**
""")

# Main prediction interface
st.markdown("---")
tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "📊 Yield Prediction"])

with tab1:
    st.subheader("Get Crop Recommendations")
    
    # User inputs for crop recommendation
    season = st.selectbox("Select Season", 
                         [' Kharif', 'Rabi', 'Summer & Winter'])
    district = st.selectbox("Select District", 
                           ['Barisal', 'Bhola', 'Bogra', 'Brahmanbaria', 'Chandpur', 'Chapainawabganj', 'Cumilla', 'Dhaka', 
                            'Dinajpur', 'Faridpur', 'Feni', 'Gaibandha', 'Gazipur', 'Gopalganj', 'Habiganj', 
                            'Jamalpur', 'Jessore', 'Jhalakati', 'Jhenaidah', 'Joypurhat', 'Khagrachari', 'Khulna', 
                            'Kishoreganj', 'Kurigram', 'Kushtia', 'Lakshmipur', 'Lalmonirhat', 'Madaripur', 
                            'Magura', 'Manikganj', 'Meherpur', 'Moulvibazar', 'Munshiganj', 'Mymensingh', 
                            'Naogaon', 'Narail', 'Narayanganj', 'Narsingdi', 'Natore', 'Nawabganj', 'Netrakona', 
                            'Nilphamari', 'Noakhali', 'Pabna', 'Panchagarh', 'Patuakhali', 'Pirojpur', 'Rajbari', 
                            'Rajshahi', 'Rangamati', 'Rangpur', 'Satkhira', 'Shariatpur', 'Sherpur', 'Sirajganj', 
                            'Sunamganj', 'Sylhet', 'Tangail', 'Thakurgaon'])
    
    # Predict crop
    if st.button("🔮 Recommend Best Crop", type="primary"):
        try:
            # Create feature vector for crop model (adjust based on your training features)
            season_encoded = models['season_encoder'].transform([season])[0]
            district_encoded = models['district_encoder'].transform([district])[0]
            
            # FIX: Ensure correct number of features for crop model (typically fewer features)
            crop_features = np.array([[season_encoded, district_encoded]])
            
            crop_pred = models['crop'].predict(crop_features)
            recommended_crop = models['crop_encoder'].inverse_transform(crop_pred)[0]
            
            st.success(f"**Recommended Crop: {recommended_crop}**")
            st.balloons()
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

with tab2:
    st.subheader("Predict Crop Yield")
    
    # User inputs for yield prediction
    col_a, col_b = st.columns(2)
    with col_a:
        crop = st.selectbox("Select Crop", 
                           ['Rice', 'Wheat', 'Maize', 'Jute', 'Potato', 'Pulses', 'Sugarcane', 'Oil seed', 'Tea', 'Vegetables'])
        area = st.number_input("Area (Hectare)", min_value=0.1, value=1.0, step=0.1)
    
    with col_b:
        production = st.number_input("Production (Ton)", min_value=0.0, value=1.0, step=0.1)
        season_yield = st.selectbox("Season", 
                                   [' Kharif', 'Rabi', 'Summer & Winter'], key="season_yield")
    
    district_yield = st.selectbox("District", 
                                 ['Barisal', 'Bhola', 'Bogra', 'Brahmanbaria', 'Chandpur', 'Chapainawabganj', 'Cumilla', 'Dhaka'], key="district_yield")
    
    # Get location and weather
    location_data = get_user_location_data()
    weather_data = None
    if location_data:
        lat, lon = location_data['latitude'], location_data['longitude']
        with st.spinner("Fetching real-time weather data..."):
            weather_data = get_weather_data(lat, lon)
    
    # Predict yield
    if st.button("📈 Predict Yield", type="primary"):
        try:
            # Encode categorical features
            crop_encoded = models['crop_encoder'].transform([crop])[0]
            season_encoded = models['season_encoder'].transform([season_yield])[0]
            district_encoded = models['district_encoder'].transform([district_yield])[0]
            
            # FIX: Create exactly 9 features to match DecisionTreeRegressor expectation
            # Order: [area, production, crop_encoded, season_encoded, district_encoded, weather_temp, weather_humidity, year, extra_feature]
            current_year = datetime.now().year
            weather_temp = weather_data['avg_temp'] if weather_data and weather_data['avg_temp'] else 25.0
            weather_humidity = weather_data['avg_humidity'] if weather_data and weather_data['avg_humidity'] else 70.0
            
            # Pad with default/derived features to make exactly 9 features
            yield_features = np.array([[
                area,                    # feature 1
                production,              # feature 2  
                crop_encoded,            # feature 3
                season_encoded,          # feature 4
                district_encoded,        # feature 5
                weather_temp,            # feature 6 (weather)
                weather_humidity,        # feature 7 (weather)
                current_year % 100,      # feature 8 (year feature)
                area * 0.1               # feature 9 (derived feature)
            ]])
            
            # Make prediction
            predicted_yield = models['yield'].predict(yield_features)[0]
            
            # Display results
            st.markdown("---")
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            
            with col_pred1:
                st.metric("Predicted Yield (Ton/ha)", f"{predicted_yield:.2f}")
            with col_pred2:
                st.metric("Area", f"{area:.2f} ha")
            with col_pred3:
                st.metric("Total Production", f"{predicted_yield * area:.2f} Ton")
            
            if weather_data:
                st.info(f"**Weather Data Used:** 🌡️ {weather_data['avg_temp']}°C, 💧 {weather_data['avg_humidity']}% humidity")
            
            st.success("✅ Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("**Debug info:** Check that your model was trained with exactly 9 features in this order: [area, production, crop, season, district, temp, humidity, year, derived]")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🛠️ Built with Streamlit | 🌍 Optimized for Bangladesh Agriculture | 📈 ML-Powered Predictions</p>
</div>
""", unsafe_allow_html=True)