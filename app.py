# ============================================================================
# QUICK MODEL TRAINING & DEPLOYMENT CODE
# ============================================================================
# Skip all hyperparameter tuning - Use the BEST parameters already found
# Train final models and save as .pkl files for Streamlit deployment
# Time: ~30 seconds total
# ============================================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

# ============================================================================
# STEP 1: LOAD & PREPARE DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA...")
print("=" * 80)

# Load your dataset

print(f"Dataset shape: {df.shape}")

# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================

print("\nCleaning data...")

# Calculate yield if not exists
if 'Yield_tons_hectare' not in df.columns:
    df['Yield_tons_hectare'] = df['Production'] / df['Area']

# Remove invalid records
df = df[df['Yield_tons_hectare'] > 0]
df = df[df['Area'] > 0]
df = df.dropna()

print(f"After cleaning: {df.shape}")
print(f"Yield range: {df['Yield_tons_hectare'].min():.2f} - {df['Yield_tons_hectare'].max():.2f}")
print(f"Unique crops: {df['Crop Name'].nunique()}")
print(f"Unique districts: {df['District'].nunique()}")
print(f"Unique seasons: {df['Season'].nunique()}")

# ============================================================================
# STEP 3: FEATURE ENGINEERING & ENCODING
# ============================================================================

print("\nEngineering features...")

# Features for yield prediction (7 features)
feature_columns_yield = [
    'Area', 'Avg Temp', 'Avg Humidity', 
    'Max Temp', 'Min Temp', 'Max Relative Humidity', 'Min Relative Humidity'
]

X_yield = df[feature_columns_yield].copy()
y_yield = df['Yield_tons_hectare'].copy()

# Features for crop prediction (without encoding yet)
feature_columns_crop = [
    'Avg Temp', 'Avg Humidity', 'Max Temp', 'Min Temp',
    'Max Relative Humidity', 'Min Relative Humidity'
]

X_crop_raw = df[feature_columns_crop].copy()
y_crop_raw = df['Crop Name'].copy()

# Create label encoders
print("Creating label encoders...")
lecrop = LabelEncoder()
leseason = LabelEncoder()
ledistrict = LabelEncoder()

# Fit encoders
df['Season_Code'] = leseason.fit_transform(df['Season'])
df['District_Code'] = ledistrict.fit_transform(df['District'])
y_crop = lecrop.fit_transform(y_crop_raw)

print(f"✅ Encoded {len(lecrop.classes_)} crops")
print(f"✅ Encoded {len(leseason.classes_)} seasons")
print(f"✅ Encoded {len(ledistrict.classes_)} districts")

# Add encoded features to crop features
X_crop = X_crop_raw.copy()
X_crop['SeasonCode'] = df['Season_Code']
X_crop['DistrictCode'] = df['District_Code']

print(f"Yield features shape: {X_yield.shape}")
print(f"Crop features shape: {X_crop.shape}")

# ============================================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================================

print("\nSplitting data...")

# 80-20 split
X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(
    X_yield, y_yield, test_size=0.2, random_state=42
)

X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

print(f"Yield - Train: {X_train_yield.shape}, Test: {X_test_yield.shape}")
print(f"Crop - Train: {X_train_crop.shape}, Test: {X_test_crop.shape}")

# ============================================================================
# STEP 5: TRAIN FINAL YIELD MODEL (Decision Tree with BEST parameters)
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING FINAL YIELD MODEL")
print("=" * 80)

# Best parameters found from hyperparameter tuning
# Decision Tree with depth=25 was the best (R² = 0.8687)
finalmodelyield = DecisionTreeRegressor(
    max_depth=25,
    random_state=42
)

# Train
finalmodelyield.fit(X_train_yield, y_train_yield)

# Evaluate
yield_pred = finalmodelyield.predict(X_test_yield)
finalyield_r2 = r2_score(y_test_yield, yield_pred)
finalyield_mae = mean_absolute_error(y_test_yield, yield_pred)

print(f"\n✅ Final Yield Model: Decision Tree")
print(f"   R² Score: {finalyield_r2:.4f}")
print(f"   MAE: {finalyield_mae:.4f} tons/hectare")

# ============================================================================
# STEP 6: TRAIN FINAL CROP MODEL (KNN with BEST parameters)
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING FINAL CROP MODEL")
print("=" * 80)

# Best parameters found from hyperparameter tuning
# KNN with k=11 was the best (Accuracy = 0.8714)
finalmodelcrop = KNeighborsClassifier(
    n_neighbors=11,
    n_jobs=-1
)

# Train
finalmodelcrop.fit(X_train_crop, y_train_crop)

# Evaluate
crop_pred = finalmodelcrop.predict(X_test_crop)
finalcrop_accuracy = accuracy_score(y_test_crop, crop_pred)

print(f"\n✅ Final Crop Model: KNN (k=11)")
print(f"   Accuracy: {finalcrop_accuracy:.4f}")

# ============================================================================
# STEP 7: SAVE MODELS AS .pkl FILES
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODELS AS .pkl FILES")
print("=" * 80)

try:
    # Save models
    joblib.dump(finalmodelyield, 'yield_model.pkl')
    print("✅ Saved: yield_model.pkl")
    
    joblib.dump(finalmodelcrop, 'crop_model.pkl')
    print("✅ Saved: crop_model.pkl")
    
    # Save label encoders
    joblib.dump(lecrop, 'label_encoder_crop.pkl')
    print("✅ Saved: label_encoder_crop.pkl")
    
    joblib.dump(leseason, 'label_encoder_season.pkl')
    print("✅ Saved: label_encoder_season.pkl")
    
    joblib.dump(ledistrict, 'label_encoder_district.pkl')
    print("✅ Saved: label_encoder_district.pkl")
    
    print("\n" + "=" * 80)
    print("✅ ALL MODELS SAVED SUCCESSFULLY!")
    print("=" * 80)
    print("\nFiles created:")
    print("  1. yield_model.pkl         (Decision Tree - Yield prediction)")
    print("  2. crop_model.pkl          (KNN - Crop classification)")
    print("  3. label_encoder_crop.pkl  (73 crops)")
    print("  4. label_encoder_season.pkl (3 seasons)")
    print("  5. label_encoder_district.pkl (64 districts)")
    print("\n➡️  Copy these files to your Streamlit app folder!")
    print("➡️  Then deploy with app_with_real_models.py renamed to app.py")
    
except Exception as e:
    print(f"❌ Error saving models: {e}")

# ============================================================================
# STEP 8: VERIFY MODELS CAN MAKE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("VERIFICATION: Test a prediction")
print("=" * 80)

try:
    # Create a sample input
    sample_yield_input = np.array([[5.0, 22.0, 70.0, 32.0, 18.0, 95.0, 44.0]])
    sample_crop_input = np.array([[22.0, 70.0, 32.0, 18.0, 95.0, 44.0, 0, 0]])
    
    # Make predictions
    sample_yield = finalmodelyield.predict(sample_yield_input)[0]
    sample_crop_pred = finalmodelcrop.predict(sample_crop_input)[0]
    sample_crop_name = lecrop.inverse_transform([sample_crop_pred])[0]
    
    print(f"\n✅ Sample Prediction Test:")
    print(f"   Input: Area=5 ha, Temp=22°C, Humidity=70%")
    print(f"   Predicted Yield: {sample_yield:.2f} tons/hectare")
    print(f"   Predicted Crop: {sample_crop_name}")
    print(f"   Models are working! ✨")
    
except Exception as e:
    print(f"❌ Error in prediction: {e}")

print("\n" + "=" * 80)
print("🎉 READY FOR DEPLOYMENT!")
print("=" * 80)
