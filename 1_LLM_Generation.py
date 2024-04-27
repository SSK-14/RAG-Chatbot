import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load configs from .env file
load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')

if "messages" not in st.session_state:
    st.session_state.messages = []

def generate(prompt):
    response = model.generate_content(prompt)
    return response.text

def prompt_template(user_input):
    return f"""Your are one of the best creative chef in the world. You have to give an amazing recipe to a beginner based on 
    the ingredients given. 
    Here is the ingredients: ```{user_input}```.
    Give the recipe containing the following details:
    1. Name of the dish
    2. Ingredients with quantity as table
    3. Cooking instructions

    Your recipe in Markdown format:"""

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter the ingredients you have..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        prompt = prompt_template(user_input)
        response = generate(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(page_title="Recipe Builder", page_icon="🍽️")
    st.title(':gray[Recipe Builder] 🍽️')
    display_chat()

if __name__ == "__main__":
    main()