import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page Config
st.set_page_config(page_title="Car Price Predictor", layout="wide")

def load_models():
    # Loading the scaler and model provided in your files
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('ridge_model.pkl')
    return scaler, model

try:
    scaler, model = load_models()
    
    st.title("🚗 Car Price Prediction Dashboard")
    st.markdown("Enter the vehicle specifications below to estimate the market value.")
    st.divider()

    # Create Three Main Columns for Input
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📊 Physical Specs")
        symboling = st.slider("Symboling (Risk Rating)", -3, 3, 0)
        wheelbase = st.number_input("Wheelbase", value=98.0)
        carlength = st.number_input("Car Length", value=174.0)
        carwidth = st.number_input("Car Width", value=65.0)
        curbweight = st.number_input("Curb Weight", value=2500)

    with col2:
        st.subheader("⚙️ Engine & Performance")
        enginesize = st.number_input("Engine Size", value=120)
        stroke = st.number_input("Stroke", value=3.25)
        horsepower = st.number_input("Horsepower", value=100)
        citympg = st.number_input("City MPG", value=24)

    with col3:
        st.subheader("🛠️ Configuration")
        # Handling One-Hot Encoded Categorical Features from your scaler list
        body_type = st.selectbox("Body Style", ["Sedan", "Hatchback", "Wagon", "Hardtop"])
        drive_wheel = st.radio("Drive Wheel", ["fwd", "rwd"])
        cylinders = st.selectbox("Cylinders", ["four", "six", "five", "three", "twelve", "two"])

    # Expandable section for rare/technical features
    with st.expander("Advanced Engine & Fuel Settings"):
        ae1, ae2 = st.columns(2)
        fuel_sys = ae1.selectbox("Fuel System", ["mpfi", "2bbl", "4bbl", "idi", "mfi", "spdi", "spfi"])
        engine_type = ae2.selectbox("Engine Type", ["ohc", "ohcf", "ohcv", "dohcv", "l", "rotor"])

    # --- PREPROCESSING LOGIC ---
    
    if st.button("Estimate Price", type="primary", use_container_width=True):
        # 1. Initialize a dictionary with all 34 features at 0
        input_data = {feat: 0.0 for feat in scaler.feature_names_in_}
        
        # 2. Fill in Numerical Values
        input_data.update({
            'symboling': symboling, 'wheelbase': wheelbase, 'carlength': carlength,
            'carwidth': carwidth, 'curbweight': curbweight, 'enginesize': enginesize,
            'horsepower': horsepower, 'citympg': citympg, 'stroke': stroke
        })
        
        # 3. Fill in One-Hot Encoded Values
        if f"carbody_{body_type.lower()}" in input_data:
            input_data[f"carbody_{body_type.lower()}"] = 1
        if f"drivewheel_{drive_wheel}" in input_data:
            input_data[f"drivewheel_{drive_wheel}"] = 1
        if f"fuelsystem_{fuel_sys}" in input_data:
            input_data[f"fuelsystem_{fuel_sys}"] = 1
        if f"enginetype_{engine_type}" in input_data:
            input_data[f"enginetype_{engine_type}"] = 1
        if f"cylindernumber_{cylinders}" in input_data:
            input_data[f"cylindernumber_{cylinders}"] = 1

        # Convert to Array
        features_array = np.array([list(input_data.values())])
        
        # Scale and Predict
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)

        # Output UI
        st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
        
        # Visualizing weights
        with st.expander("View Feature Analysis"):
            importance_df = pd.DataFrame({
                'Feature': scaler.feature_names_in_,
                'Value': features_array[0]
            }).set_index('Feature')
            st.bar_chart(importance_df)

except FileNotFoundError:
    st.error("Please ensure 'scaler.pkl' and 'ridge_model.pkl' are in the same directory.")
