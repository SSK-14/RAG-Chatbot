import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
from dotenv import load_dotenv

# Load configs from .env file
load_dotenv()

set_debug(True)

llm = ChatGoogleGenerativeAI(model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

qdrant_client = QdrantClient("http://localhost:6333") 
qdrant = Qdrant(client=qdrant_client, collection_name="tech_radar", embeddings=embeddings)
retriever = qdrant.as_retriever()
output_parser = StrOutputParser()

PROMPT_TEMPLATE = """You are an helpful assistant that assists users with their questions. 
Use the following pieces of context to answer the user's question:
CONTEXT INFORMATION is below.
---------------------
{context}
---------------------
RULES:
1. Only Answer the USER QUESTION using the CONTEXT text above.
2. Keep your answer grounded in the facts of the CONTEXT. 
3. If you don't know the answer, just say that you don't know.
4. Should not answer any out-of-context USER QUESTION.

USER QUESTION: ```{question}```
Answer in markdown:"""

rag_prompt_template = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt_template
    | llm
    | StrOutputParser()
)

if "messages" not in st.session_state:
    st.session_state.messages = []

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = rag_chain.invoke(user_input)    

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🦾")
    st.title(':gray[Tech Radar AI] 🦾')
    display_chat()

if __name__ == "__main__":
    main()