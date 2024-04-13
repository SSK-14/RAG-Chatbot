# RAG Chatbot

## Requirements ✅
- Python 3.7 or above
- [Get an Gemini API key](https://makersuite.google.com/app/apikey) 
- Streamlit library

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
streamlit run chatbot.py 
```

## Setting up Vector database 🗃️

We will be using [qdrant](https://qdrant.tech/documentation/overview/) vector database

### Download and run
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

### Ingesting data

1. Add documents needed (PDF's) to knowledge_base folder
2. export GOOGLE_API_KEY="YOUR API KEY"
3. Run file ```create_vectordb.py```
