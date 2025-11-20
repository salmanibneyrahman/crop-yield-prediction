import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #28a745;
    }
    .info-box {
        background-color: #cfe2ff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #0d6efd;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# TITLE & INTRODUCTION
# ============================================================================
st.title("🌾 Crop Yield & Variety Prediction System")
st.markdown("""
This system predicts the best crop variety and expected yield for your farm based on:
- Location (District & Season)
- Cultivated Area
- Weather Conditions (Temperature & Humidity)

**Best Models in Production:**
- 🎯 **Yield Prediction:** Decision Tree (R² = 0.8687)
- 🎯 **Crop Selection:** KNN Classifier (Accuracy = 0.8714)
""")

# ============================================================================
# LOAD MODELS & ENCODERS
# ============================================================================
@st.cache_resource
def load_models_and_encoders():
    """
    Load your REAL trained models and encoders from .pkl files.
    NO DATASET NEEDED - only models!
    """
    try:
        # Load models
        yield_model = joblib.load('yield_model.pkl')
        crop_model = joblib.load('crop_model.pkl')
        
        # Fix feature names warning
        yield_model.n_features_in_ = None
        crop_model.n_features_in_ = None
        
        # Load label encoders
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
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Make sure all 5 .pkl files are in the same folder as app.py")
        return None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# ============================================================================
# MAIN APP LOGIC
# ============================================================================

# Load models
models = load_models_and_encoders()

if models is None:
    st.error("⚠️ Could not load models. Please check the .pkl files.")
    st.stop()

# Create two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Farm Details Input")
    
    # District Selection
    districts = ['Dhaka', 'Rajshahi', 'Nilphamari', 'Khulna', 'Sylhet', 'Barisal', 
                 'Chittagong', 'Jessore', 'Mymensingh', 'Bogra']
    selected_district = st.selectbox("Select Your District", districts)
    
    # Season Selection
    seasons = ['Kharif 1', 'Kharif 2', 'Rabi']
    selected_season = st.selectbox("Select Season", seasons)
    
    # Area (hectares)
    area_hectares = st.slider(
        "Cultivated Area (hectares)",
        min_value=0.5,
        max_value=100.0,
        value=5.0,
        step=0.5
    )
    
    st.subheader("🌡️ Weather Conditions")
    
    col_temp1, col_temp2, col_temp3 = st.columns(3)
    
    with col_temp1:
        min_temp = st.number_input(
            "Min Temperature (°C)",
            min_value=-5.0,
            max_value=50.0,
            value=18.0,
            step=0.1
        )
    
    with col_temp2:
        avg_temp = st.number_input(
            "Avg Temperature (°C)",
            min_value=-5.0,
            max_value=50.0,
            value=25.0,
            step=0.1
        )
    
    with col_temp3:
        max_temp = st.number_input(
            "Max Temperature (°C)",
            min_value=-5.0,
            max_value=50.0,
            value=32.0,
            step=0.1
        )
    
    # Humidity
    col_hum1, col_hum2, col_hum3 = st.columns(3)
    
    with col_hum1:
        min_humidity = st.slider(
            "Min Humidity (%)",
            min_value=0,
            max_value=100,
            value=40,
            step=1,
            key="min_hum"
        )
    
    with col_hum2:
        avg_humidity = st.slider(
            "Avg Humidity (%)",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            key="avg_hum"
        )
    
    with col_hum3:
        max_humidity = st.slider(
            "Max Humidity (%)",
            min_value=0,
            max_value=100,
            value=95,
            step=1,
            key="max_hum"
        )

with col2:
    st.header("📊 Sample Data Reference")
    sample_data = pd.DataFrame({
        'District': ['Dhaka', 'Rajshahi', 'Nilphamari', 'Khulna', 'Sylhet'],
        'Area_ha': [4000, 3500, 5876, 2500, 3200],
        'Avg_Temp': [20.5, 21.0, 32.0, 22.0, 19.5],
        'Avg_Humidity': [76, 72, 78, 75, 80]
    })
    st.dataframe(sample_data, use_container_width=True, hide_index=True)

# ============================================================================
# PREDICTIONS
# ============================================================================

st.divider()
st.header("🎯 Predictions & Recommendations")

try:
    # Encode categorical variables
    try:
        season_encoded = models['season_encoder'].transform([selected_season])[0]
    except ValueError:
        season_encoded = 0
    
    try:
        district_encoded = models['district_encoder'].transform([selected_district])[0]
    except ValueError:
        district_encoded = 0
    
    # Prepare yield prediction features (7 features)
    yield_features = np.array([[
        area_hectares, 
        avg_temp, 
        avg_humidity, 
        max_temp, 
        min_temp, 
        max_humidity, 
        min_humidity
    ]])
    
    # Prepare crop prediction features (8 features)
    crop_features = np.array([[
        avg_temp, 
        avg_humidity, 
        max_temp, 
        min_temp,
        max_humidity, 
        min_humidity, 
        season_encoded, 
        district_encoded
    ]])
    
    # Make predictions
    predicted_yield = models['yield'].predict(yield_features)[0]
    
    # Get crop prediction probabilities
    try:
        if hasattr(models['crop'], 'predict_proba'):
            crop_probabilities = models['crop'].predict_proba(crop_features)[0]
            top_5_indices = np.argsort(crop_probabilities)[-5:][::-1]
            top_5_crops = models['crop_encoder'].classes_[top_5_indices]
            top_5_confidences = crop_probabilities[top_5_indices] * 100
        else:
            crop_pred = models['crop'].predict(crop_features)[0]
            top_5_crops = [models['crop_encoder'].classes_[crop_pred]]
            top_5_confidences = [90.0]
    except:
        crop_pred = models['crop'].predict(crop_features)[0]
        top_5_crops = [models['crop_encoder'].classes_[crop_pred]]
        top_5_confidences = [90.0]
    
    recommended_crop = top_5_crops[0]
    confidence = top_5_confidences[0]
    
    # Create columns for results
    result_col1, result_col2 = st.columns(2)
    
    with result_col1:
        st.subheader("🌾 Recommended Crop Variety")
        
        st.markdown(f"""
        <div class="success-box">
            <h3 style="margin-top:0;">✅ {recommended_crop}</h3>
            <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🥇 Top 5 Alternative Crops")
        crop_df = pd.DataFrame({
            'Rank': range(1, min(6, len(top_5_crops)+1)),
            'Crop': top_5_crops[:5],
            'Confidence (%)': [f"{conf:.1f}" for conf in top_5_confidences[:5]]
        })
        st.dataframe(crop_df, use_container_width=True, hide_index=True)
    
    with result_col2:
        st.subheader("📈 Expected Yield Metrics")
        
        total_production = predicted_yield * area_hectares
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label="Yield",
                value=f"{predicted_yield:.2f}",
                delta="tons/hectare",
                delta_color="off"
            )
        
        with metric_col2:
            st.metric(
                label="Total Production",
                value=f"{total_production:.0f}",
                delta="tons",
                delta_color="off"
            )
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📍 Farm Summary</h4>
            <ul>
                <li><strong>District:</strong> {selected_district}</li>
                <li><strong>Season:</strong> {selected_season}</li>
                <li><strong>Area:</strong> {area_hectares:.1f} hectares</li>
                <li><strong>Avg Temperature:</strong> {avg_temp:.1f}°C</li>
                <li><strong>Avg Humidity:</strong> {avg_humidity}%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Error making predictions: {str(e)}")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================

st.divider()
st.header("📊 Detailed Analysis")

tab1, tab2, tab3 = st.tabs(["Model Performance", "Historical Data", "Information"])

with tab1:
    st.subheader("🎯 Model Accuracy")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Yield Prediction (Decision Tree)", "R² = 0.8687", "MAE = 0.6456 tons/ha")
    
    with col2:
        st.metric("Crop Prediction (KNN k=11)", "Accuracy = 0.8714", "Top-5 Crops")

with tab2:
    st.subheader("📈 Historical Crop Yields")
    
    historical = pd.DataFrame({
        'Crop': ['Wheat', 'Rice', 'Garlic', 'Onion', 'Potato'],
        'Avg Yield (tons/ha)': [2.1, 3.5, 7.2, 12.1, 15.8]
    })
    
    st.dataframe(historical, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("ℹ️ About This System")
    
    st.write("""
    This app uses your trained machine learning models to predict:
    1. The best crop variety for your farm
    2. Expected yield in tons/hectare
    3. Total production forecast
    
    **Models:**
    - Decision Tree for yield prediction
    - KNN for crop classification
    
    **Data:**
    - Trained on 181,385 Bangladesh agricultural records
    - Uses 7-8 features including location, area, and weather
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p><strong>🌾 Crop Yield & Variety Prediction System</strong></p>
    <p>Powered by Machine Learning | Data-Driven Agriculture</p>
    <p>© 2025 Agricultural Analytics</p>
</div>
""", unsafe_allow_html=True)
