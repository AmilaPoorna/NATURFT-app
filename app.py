import streamlit as st

st.set_page_config(
    page_title="Nylon Dyeing App",
    page_icon="🎨",
    layout="wide"
)

st.sidebar.title("Navigation")
st.sidebar.page_link("Home", "🏠 Home", "home.py")
st.sidebar.page_link("AI Assistant", "🤖 AI Assistant", "AI_assistant.py")

st.write("### Welcome to the Nylon Dyeing Recipe Status Predictor!")
st.write("Use the sidebar to navigate between pages.")
