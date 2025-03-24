import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import base64
import openai

# Set up the OpenAI API key using Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

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

set_background("background.jpg")

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

st.title('Nylon Dyeing Recipe Status Predictor')

def reset_prediction():
    st.session_state.prediction_class = None

recipe_quantity = st.number_input('Enter Recipe Quantity (kg):', min_value=0.001, step=0.001, format="%.3f", key="recipe_quantity", on_change=reset_prediction)
colour_shade = st.selectbox('Select the Colour Shade:', ['Very Light', 'Light', 'Medium', 'Dark', 'Very Dark'], key="colour_shade", on_change=reset_prediction)
first_colour = st.radio('Is the Color Being Created for the First Time in the Dye Lab?', ['Yes', 'No'], key="first_colour", on_change=reset_prediction)
colour_description = st.selectbox('Select the Colour Description:', ['Normal', 'Softner', 'Special Colour'], key="colour_description", on_change=reset_prediction)
lab_dip = st.radio('Is the Swatch Being Created in the Dye Lab?', ['Yes', 'No'], key="lab_dip", on_change=reset_prediction)
nylon_type = st.selectbox('Select the Nylon Type:', ['Stretch Nylon', 'Micro Fiber Streach Nylon', 'Other'], key="nylon_type", on_change=reset_prediction)
denier = st.selectbox('Select the Denier Count:', [44, 70, 78, 100], key="denier", on_change=reset_prediction)
dyeing_method = st.selectbox('Select the Dyeing Method:', ['Bullet', 'Hank', 'Package'], key="dyeing_method", on_change=reset_prediction)
colour = st.selectbox('Select the Colour:', ['Black', 'White', 'Grey', 'Blue', 'Navy Blue', 'Green', 'Pink', 'Red', 'Orange', 'Yellow', 'Beige', 'Brown', 'Purple', 'Cream', 'Lylac', 'Other'], key="colour", on_change=reset_prediction)
machine_code = st.selectbox('Select the Machine Code:', ['D-001O', 'D-002A', 'D-002B', 'D-002C', 'D-003A', 'D-003B', 'D-003C ', 'D-003D', 'D-004B', 'D-004C', 'D-005A', 'D-006A', 'D-006B', 'D-006C', 'D-008A', 'D-010A', 'D-012A', 'D-012B', 'D-012C', 'D-015A', 'D-016A', 'D-024A', 'D-028A', 'D-028B', 'D-030A', 'D-030B', 'D-036A', 'D-042A', 'D-045A', 'D-048A', 'D-048B', 'D-054A', 'D-056A', 'D-075A', 'D-090A ', 'D-090B', 'D-104A', 'D-104B', 'D-108A', 'D-132A', 'D-132B', 'D-216A', 'D-216B', 'D-264A', 'D-264B', 'D-432A', 'D-558A', 'D-981A', 'D-981B'],key="machine_capacity", on_change=reset_prediction)

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
    
    rft_data = pd.DataFrame(input_data, index=[0])

    rft_data['ColourShade'] = colour_shade_encoder.transform(rft_data['ColourShade'])
    rft_data['IsFirstColour'] = first_colour_encoder.transform(rft_data['IsFirstColour'])
    rft_data['ColourDescription'] = colour_description_encoder.transform(rft_data['ColourDescription'])
    rft_data['IsLabDip'] = lab_dip_encoder.transform(rft_data['IsLabDip'])
    rft_data['NylonType'] = nylon_type_encoder.transform(rft_data['NylonType'])
    rft_data['Denier'] = denier_encoder.transform(rft_data['Denier'])
    rft_data['DyeingMethod'] = dyeing_method_encoder.transform(rft_data['DyeingMethod'])
    rft_data['Colour'] = colour_encoder.transform(rft_data['Colour'])
    rft_data['MachineCode'] = machine_code_encoder.transform(rft_data['MachineCode'])
    rft_data['RecipeQty'] = scaler.transform(rft_data[['RecipeQty']])

    rft_data = rft_data[X_train_columns]
    
    prediction_class = classification_model.predict(rft_data)
    st.session_state.prediction_class = prediction_class[0]
    
if st.session_state.prediction_class is not None:
    if st.session_state.prediction_class == 1:
        prediction_label = "RFT"
    else:
        prediction_label = "WFT. Please proceed with necessary steps."
    
    st.write(f"Prediction: {prediction_label}")

# AI Assistant Page
st.sidebar.title("AI Assistant")
selected_page = st.sidebar.radio("Choose Page", ["Nylon Dyeing Recipe Status Predictor", "AI Assistant"])

if selected_page == "AI Assistant":
    st.title("AI Assistant")
    user_input = st.text_input("Ask me anything about nylon dyeing or recipes:", "")

    if user_input:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=user_input,
                max_tokens=150
            )
            answer = response.choices[0].text.strip()
            st.write(f"AI Assistant: {answer}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
