import os
import streamlit as st
import google.generativeai as genai
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load configs from .env file
load_dotenv()

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')
qdrant = QdrantClient("http://localhost:6333") 

if "messages" not in st.session_state:
    st.session_state.messages = []

def generate(prompt):
    response = model.generate_content(prompt)
    return response.text

def create_embedding(text):
    return genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query",
    )['embedding']

def search_database(embedding):
    response = qdrant.search(
        collection_name="tech_radar", # Name of the collection
        query_vector=embedding,
        with_payload=True,
        limit=4, # Number of search results to fetch
    )
    print("Search Results: ", response)
    return [item.payload.get("page_content") for item in response]

def search_document(user_input):
    # Create embeddings of the user input
    embedding = create_embedding(user_input)
    # Search the Qdrant database
    return search_database(embedding)

def prompt_template(question, search_results):
    return f"""You are an Intelligent Chatbot, a helpful assistant that assists users with the context provided. 
    Use the following pieces of context to answer the user's question:
    {search_results}

    USER QUESTION: ```{question}```
    Answer in markdown:"""


def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Here is RAG flow
        search_results = search_document(user_input)
        prompt = prompt_template(user_input, search_results)
        response = generate(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🦾")
    st.title(':gray[Tech Radar AI] 🦾')
    display_chat()

if __name__ == "__main__":
    main()