import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import os

# Load the machine learning model and encoders
classification_model = joblib.load('classification_model.pkl')
colour_description_encoder = joblib.load('colour_description_encoder.pkl')
colour_encoder = joblib.load('colour_encoder.pkl')
colour_shade_encoder = joblib.load('colour_shade_encoder.pkl')
denier_encoder = joblib.load('denier_encoder.pkl')
dyeing_method_encoder = joblib.load('dyeing_method_encoder.pkl')
first_colour_encoder = joblib.load('first_colour_encoder.pkl')
lab_dip_encoder = joblib.load('lab_dip_encoder.pkl')
machine_code_encoder = joblib.load('machine_code_encoder.pkl')
nylon_type_encoder = joblib.load("nylon_type_encoder.pkl")
scaler = joblib.load('scaler.pkl')
X_train_columns = joblib.load("X_train.pkl")

# Function to set background
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background(image_path):
    base64_str = get_base64_image(image_path)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background image (ensure the file exists)
set_background("background.jpg")

st.title('Nylon Dyeing Recipe Status Predictor')

# Function to reset prediction
def reset_prediction():
    st.session_state.prediction_class = None

# User inputs
recipe_quantity = st.number_input('Enter Recipe Quantity (kg):', min_value=0.001, step=0.001, format="%.3f", key="recipe_quantity", on_change=reset_prediction)
colour_shade = st.selectbox('Select the Colour Shade:', ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark'], key="colour_shade", on_change=reset_prediction)
first_colour = st.radio('Is the Color Being Created for the First Time in the Dye Lab?', ['Yes', 'No'], key="first_colour", on_change=reset_prediction)
colour_description = st.selectbox('Select the Colour Description:', ['Normal', 'Softner', 'Special Colour'], key="colour_description", on_change=reset_prediction)
lab_dip = st.radio('Is the Swatch Being Created in the Dye Lab?', ['Yes', 'No'], key="lab_dip", on_change=reset_prediction)
nylon_type = st.selectbox('Select the Nylon Type:', ['Stretch Nylon', 'Micro Fiber Streach Nylon', 'Other'], key="nylon_type", on_change=reset_prediction)
denier = st.selectbox('Select the Denier Count:', [44, 70, 78, 100], key="denier", on_change=reset_prediction)
dyeing_method = st.selectbox('Select the Dyeing Method:', ['Bullet', 'Hank', 'Package'], key="dyeing_method", on_change=reset_prediction)
colour = st.selectbox('Select the Colour:', ['Black', 'White', 'Grey', 'Blue', 'Navy Blue', 'Green', 'Pink', 'Red', 'Orange', 'Yellow', 'Beige', 'Brown', 'Purple', 'Cream', 'Lylac', 'Other'], key="colour", on_change=reset_prediction)
machine_code = st.selectbox('Select the Machine Code:', ['D-001O', 'D-002A', 'D-002B', 'D-002C'], key="machine_capacity", on_change=reset_prediction)

# Prediction state
if 'prediction_class' not in st.session_state:
    st.session_state.prediction_class = None

if st.button('Predict Status'):
    input_data = pd.DataFrame({
        'RecipeQty': [recipe_quantity],
        'ColourShade': [colour_shade],
        'IsFirstColour': [first_colour],
        'ColourDescription': [colour_description],
        'IsLabDip': [lab_dip],
        'NylonType': [nylon_type],
        'Denier': [denier],
        'DyeingMethod': [dyeing_method],
        'Colour': [colour],
        'MachineCode': [machine_code]
    })

    # Encoding and scaling
    input_data['ColourShade'] = colour_shade_encoder.transform(input_data['ColourShade'])
    input_data['IsFirstColour'] = first_colour_encoder.transform(input_data['IsFirstColour'])
    input_data['ColourDescription'] = colour_description_encoder.transform(input_data['ColourDescription'])
    input_data['IsLabDip'] = lab_dip_encoder.transform(input_data['IsLabDip'])
    input_data['NylonType'] = nylon_type_encoder.transform(input_data['NylonType'])
    input_data['Denier'] = denier_encoder.transform(input_data['Denier'])
    input_data['DyeingMethod'] = dyeing_method_encoder.transform(input_data['DyeingMethod'])
    input_data['Colour'] = colour_encoder.transform(input_data['Colour'])
    input_data['MachineCode'] = machine_code_encoder.transform(input_data['MachineCode'])
    input_data['RecipeQty'] = scaler.transform(input_data[['RecipeQty']])

    input_data = input_data[X_train_columns]

    # Predict
    prediction = classification_model.predict(input_data)
    st.session_state.prediction_class = prediction[0]

if st.session_state.prediction_class is not None:
    prediction_label = "RFT" if st.session_state.prediction_class == 1 else "WFT. Please proceed with necessary steps."
    st.write(f"Prediction: {prediction_label}")