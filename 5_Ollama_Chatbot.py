import streamlit as st
from pypdf import PdfReader
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are an helpful assistant to help the Requiter to understand and analyze the candidate's resume.
Candidate's resume is given below:
---------------------
{context}
---------------------
QUESTION: ```{question}```
Answer in markdown:"""

# Download https://ollama.com/ 
# Run the model ```ollama run tinyllama```` 
# Feel free to use any other model : https://ollama.com/library
llm = ChatOllama(model="tinyllama")
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
chain = prompt | llm | StrOutputParser()

if "messages" not in st.session_state:
    st.session_state.messages = []

def display_chat(resume_text):
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Enter your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response = chain.invoke({"question": user_input, "context": resume_text})    

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(page_title="Ollama Chatbot", page_icon="💼")
    st.title(':gray[Requiter AI] 💼')
    uploaded_file = st.file_uploader("Upload Candidate Resume PDF", type="pdf")
    if uploaded_file:
        resume_text = ""
        reader = PdfReader(uploaded_file)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            resume_text += page.extract_text()
        display_chat(resume_text)

if __name__ == "__main__":
    main()