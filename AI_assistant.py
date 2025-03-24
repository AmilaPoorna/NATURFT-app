import streamlit as st
import os
from openai import OpenAI

st.title("AI Assistant 🤖")
st.write("Ask me anything!")

# Retrieve API key from environment variable
api_key = os.getenv("OPENAI_RFT_KEY")

if not api_key:
    st.error("API key not found. Please set OPENAI_RFT_KEY as an environment variable.")
else:
    client = OpenAI(api_key=api_key)

    query = st.text_input("Enter your question:")
    if st.button("Ask AI"):
        if query:
            try:
                response = client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": query}]
                )
                st.write(response["choices"][0]["message"]["content"])
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question!")
