import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Bangladesh Crop & Yield Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# LOAD MODELS & ENCODERS (FIXED - 3 files only)
# ============================================================================
@st.cache_resource
def load_models():
    try:
        yield_model = joblib.load("yield_model.pkl")
        crop_model = joblib.load("crop_model.pkl")
        le_crop = joblib.load("label_encoder_crop.pkl")
        
        return {
            "yield": yield_model,
            "crop": crop_model,
            "crop_encoder": le_crop
        }
    except Exception as e:
        st.error(f"Error loading models/encoders: {e}")
        st.stop()

models = load_models()

# ============================================================================
# SAFE LOCATION + WEATHER HELPERS
# ============================================================================
@st.cache_data(ttl=600)
def get_user_location():
    """Try multiple IP geolocation APIs."""
    apis = [
        {"name": "ipapi.co", "url": "https://ipapi.co/json", 
         "parser": lambda d: {"city": d.get("city"), "country": d.get("country_name"), 
                              "lat": d.get("latitude"), "lon": d.get("longitude")}},
        {"name": "ipinfo.io", "url": "https://ipinfo.io/json",
         "parser": lambda d: {"city": d.get("city"), "country": d.get("country"),
                              "lat": float(d.get("loc", "0,0").split(",")[0]) if d.get("loc") else None,
                              "lon": float(d.get("loc", "0,0").split(",")[1]) if d.get("loc") else None}}
    ]
    
    for api in apis:
        try:
            r = requests.get(api["url"], timeout=5)
            if r.status_code == 200:
                data = r.json()
                loc = api["parser"](data)
                if loc.get("lat") is not None and loc.get("lon") is not None:
                    return loc
        except Exception:
            continue
    return None

@st.cache_data(ttl=600)
def get_weather(lat: float, lon: float):
    """Fetch basic weather from Open-Meteo."""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min"
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        
        data = r.json()
        current = data.get("current", {})
        daily = data.get("daily", {})
        
        return {
            "avg_temp": current.get("temperature_2m"),
            "avg_humidity": current.get("relative_humidity_2m"),
            "min_temp": (daily.get("temperature_2m_min") or [None])[0],
            "max_temp": (daily.get("temperature_2m_max") or [None])[0],
            "min_humidity": (daily.get("relative_humidity_2m_min") or [None])[0],
            "max_humidity": (daily.get("relative_humidity_2m_max") or [None])[0],
        }
    except Exception:
        return None

# ============================================================================
# HEADER
# ============================================================================
st.title("🌾 Bangladesh Crop & Yield Intelligence System")

st.markdown("""
This app uses machine learning models trained on **Bangladesh SPAS data** 
to recommend suitable crops and estimate yield (tons/hectare).

**Weather-based predictions** - no district/season encoders needed!
""")

st.markdown("---")

# ============================================================================
# INPUT MODE SELECTION
# ============================================================================
mode = st.radio(
    "How do you want to provide weather data?",
    ["Auto (approximate from IP)", "Manual"],
    horizontal=True,
)

st.markdown(
    "> ℹ️ On public hosting, auto-location shows **server** location. Use **Manual** for accuracy."
)

st.markdown("---")

# ============================================================================
# COLLECT INPUTS (FIXED - no district/season)
# ============================================================================
col_left, col_right = st.columns(2)

# Area input only (left column)
with col_left:
    st.subheader("🌾 Farm Details")
    area_ha = st.number_input(
        "Cultivated Area (hectares)",
        min_value=0.1, max_value=1_000_000.0, value=10_000.0, step=100.0
    )

# Weather inputs (right column)
with col_right:
    st.subheader("🌤️ Weather Conditions (°C, %)")
    if mode == "Auto (approximate from IP)":
        loc = get_user_location()
        if loc is not None:
            st.info(f"📍 Detected: **{loc.get('city', 'Unknown')}, {loc.get('country', 'Unknown')}**")
            weather = get_weather(loc["lat"], loc["lon"])
        else:
            st.warning("Could not detect location.")
            weather = None

        if weather is not None:
            st.success(f"Detected: **{weather['avg_temp']:.1f}°C**, **{weather['avg_humidity']:.0f}%** humidity")
            
            # Pre-fill with detected weather
            min_temp = st.number_input("Min Temp (°C)", value=float(weather["min_temp"] or 20.0))
            avg_temp = st.number_input("Avg Temp (°C)", value=float(weather["avg_temp"] or 25.0))
            max_temp = st.number_input("Max Temp (°C)", value=float(weather["max_temp"] or 32.0))
            min_humidity = st.number_input("Min Humidity (%)", 0, 100, int(weather["min_humidity"] or 40))
            avg_humidity = st.number_input("Avg Humidity (%)", 0, 100, int(weather["avg_humidity"] or 70))
            max_humidity = st.number_input("Max Humidity (%)", 0, 100, int(weather["max_humidity"] or 95))
        else:
            # Fallback manual
            min_temp = st.number_input("Min Temp (°C)", value=20.0)
            avg_temp = st.number_input("Avg Temp (°C)", value=26.0)
            max_temp = st.number_input("Max Temp (°C)", value=32.0)
            min_humidity = st.number_input("Min Humidity (%)", 0, 100, 40)
            avg_humidity = st.number_input("Avg Humidity (%)", 0, 100, 70)
            max_humidity = st.number_input("Max Humidity (%)", 0, 100, 95)
    else:  # Manual mode
        min_temp = st.number_input("Min Temp (°C)", value=20.0)
        avg_temp = st.number_input("Avg Temp (°C)", value=26.0)
        max_temp = st.number_input("Max Temp (°C)", value=32.0)
        min_humidity = st.number_input("Min Humidity (%)", 0, 100, 40)
        avg_humidity = st.number_input("Avg Humidity (%)", 0, 100, 70)
        max_humidity = st.number_input("Max Humidity (%)", 0, 100, 95)

st.markdown("---")

# ============================================================================
# PREDICT BUTTON (FIXED - weather only)
# ============================================================================
if st.button("🔮 Predict Recommended Crop & Yield", use_container_width=True):
    try:
        # Yield prediction (7 features: area + 6 weather)
        yield_features = np.array([[
            area_ha, avg_temp, avg_humidity, max_temp, 
            min_temp, max_humidity, min_humidity
        ]])
        
        # Crop prediction (6 weather features ONLY)
        crop_features = np.array([[
            avg_temp, avg_humidity, max_temp, 
            min_temp, max_humidity, min_humidity
        ]])

        # Predictions
        yield_per_ha = float(models["yield"].predict(yield_features)[0])
        total_production = yield_per_ha * area_ha

        crop_idx = int(models["crop"].predict(crop_features)[0])
        crop_name = models["crop_encoder"].inverse_transform([crop_idx])[0]

        # Confidence
        conf_str = "N/A"
        if hasattr(models["crop"], "predict_proba"):
            proba = models["crop"].predict_proba(crop_features)[0]
            conf_str = f"{proba.max()*100:.1f}%"

        # Results
        st.subheader("✅ Prediction Results")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("🥬 Recommended Crop", crop_name.upper(), f"Conf: {conf_str}")
        with c2:
            st.metric("📈 Yield (tons/ha)", f"{yield_per_ha:.2f}")
        with c3:
            st.metric("🚛 Total Production", f"{total_production:,.0f} tons")

        # Input summary
        st.markdown("### 📋 Input Summary")
        summary_df = pd.DataFrame({
            "Parameter": ["Area (ha)", "Min Temp (°C)", "Avg Temp (°C)", "Max Temp (°C)", 
                         "Min Humidity (%)", "Avg Humidity (%)", "Max Humidity (%)"],
            "Value": [f"{area_ha:,.1f}", f"{min_temp:.1f}", f"{avg_temp:.1f}", 
                     f"{max_temp:.1f}", f"{min_humidity}", f"{avg_humidity}", f"{max_humidity}"]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

st.markdown("---")
st.markdown("*Powered by Bangladesh SPAS agricultural data & ML models*")
