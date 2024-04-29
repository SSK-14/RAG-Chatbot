import json, os
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load configs from .env file
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# qdrant_client = QdrantClient("http://localhost:6333") 
qdrant_client = QdrantClient(os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'])
qdrant = Qdrant(client=qdrant_client, collection_name="tech_radar", embeddings=embeddings)

retriever = qdrant.as_retriever(search_type="mmr")
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)


if "messages" not in st.session_state:
    st.session_state.messages = []

def prompt_template(question, search_results):
    return f"""You are an Intelligent Chatbot, a helpful assistant that assists users with the context provided. 
    Use the following pieces of context to answer the user's question:
    {search_results}

    USER QUESTION: ```{question}```
    Answer in markdown:"""

def intent_prompt(user_query):
    return f"""Role: Intent Classifier for Virtual Assistant Interactions.
    Task: Identify and classify user intents from their inputs with a virtual assistant, aligning each with a specific intent from the list provided.
    Context: This dialogue involves a virtual assistant and a customer discussing about technology etc.
    Predefined Intents:
    Greeting: User greets the assistant or initiates a conversation.
    Output: {{"intent": "greeting", "message": "<Reply to user's greeting>"}}
    Query not clear: User's query is not clear or ambiguous.
    Output: {{"intent": "query_not_clear", "message": "<Ask user to provide more details or clarification>"}}
    Valid query: User's query is clear and valid.
    [ NO NEED TO ANSWER QUERY ]
    Output: {{"intent": "valid_query", "message": "NONE"}}
    Out of scope or context: Applies when the user's request doesn't fit any listed intents or falls beyond the assistant's scope, or the intent is ambiguous.
    Output: {{"intent": "out_of_scope", "message": "<Inform user that the request is out of scope>"}}

    Your objective is to accurately map each user responses to the relevant intent and provide the assistant's response.
    User Query: {user_query}
    Your Output JSON response:"""

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Find the intent of the user input
        intent_response = llm.invoke(
            intent_prompt(user_input)
        ).content
        print(intent_response)
        
        if json.loads(intent_response)["intent"] != "valid_query":
            response = json.loads(intent_response)["message"]
        else:
            # Search for relevant documents with mmr and re-ranking
            result_docs = compression_retriever.get_relevant_documents(user_input)
            # Generate response
            response = llm.invoke(
                prompt_template(user_input, result_docs)
            ).content

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🦾")
    st.title(':gray[Tech Radar AI] 🦾')
    display_chat()

if __name__ == "__main__":
    main()