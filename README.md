# Build your RAG Chatbot

Welcome to our hands-on workshop where you'll dive into the world of building RAG-based applications! In this workshop, you'll embark on a journey through below modules, each designed to equip you with the knowledge and skills to create your very own RAG chatbot application.

| Module                       | Description                                                                                     | File              |
|-----------------------------|-------------------------------------------------------------------------------------------------|------------------------|
| 🔮 LLM Generation           | Using LLM with prompt engineering to solve a specific use case.                                 | [1_LLM_Generation.py](1_LLM_Generation.py) |
| 📚 Vector Database          | Creating a vector database from our knowledge base (PDFs) and the process of data ingestion.     | [2_Vector_DB[qdrant].py](2_Vector_DB[qdrant].py) |
| 🤖 RAG Chatbot              | Implementing a chatbot using RAG with the vector database and LLM for response generation.       | [3_RAG_Chatbot.py](3_RAG_Chatbot.py) |
| 🔗 RAG & LangChain  | Integrating LangChain library to enhance the RAG chatbot application.                             | [4_RAG_Chatbot_Langchain.py](4_RAG_Chatbot_Langchain.py) |
| 🦙 Ollama Chatbot           | Utilizing an open-source LLM running on our machine for generative AI tasks.                     | [5_Ollama_Chatbot.py](5_Ollama_Chatbot.py) |
| 📈 Advanced RAG           | Optimizing RAG with intent recognition, re-ranking, mmr.                     | [6_Advanced_RAG.py](6_Advanced_RAG.py) |

## Requirements ✅
- Python 3.7 or above
- [Get an Gemini API key](https://makersuite.google.com/app/apikey) 

## Run The Application ⚙️

### 1. Clone the repo
```
git clone https://github.com/SSK-14/RAG-Chatbot.git
```

### 2. Install packages
If running for the first time,

- Create virtual environment
```
pip3 install virtualenv
python3 -m venv {your-venvname}
source {your-venvname}/bin/activate
```

- Install required libraries
```
pip3 install -r requirements.txt
```

- Activate your virtual environment
```
source {your-venvname}/bin/activate
```

### 3. Set up your API key
```
export GOOGLE_API_KEY="YOUR API KEY"
```

### 4. Running
```
streamlit run 1_LLM_Generation.py 
```

## Setting up Vector database 🗃️

We will be using [qdrant](https://qdrant.tech/documentation/overview/) vector database

### Use the Cloud 

[Qdrant Cloud Sign Up](https://cloud.qdrant.io/login) 

1. Create a Cluster 
2. Get API key and cluster URL

### Setup in Local
First, download the latest Qdrant image from Dockerhub:
```
docker pull qdrant/qdrant
```
Then, run the service:
```
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
