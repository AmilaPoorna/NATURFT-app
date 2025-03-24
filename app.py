import streamlit as st

st.set_page_config(
    page_title="Nylon Dyeing App",
    page_icon="🎨",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 AI Assistant"])

# Ensure only one page loads at a time
if page == "🏠 Home":
    st.session_state.current_page = "home"
    from home import *
elif page == "🤖 AI Assistant":
    st.session_state.current_page = "ai_assistant"
    from AI_assistant import *
