import streamlit as st

st.set_page_config(
    page_title="Nylon Dyeing App",
    page_icon="🎨",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a Page", ["🏠 Home", "🤖 AI Assistant"])

if page == "🏠 Home":
    import home
elif page == "🤖 AI Assistant":
    import AI_assistant

st.write("### Welcome to the Nylon Dyeing Recipe Status Predictor!")
st.write("Use the sidebar to navigate between pages.")
