import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
import google.generativeai as genai

PATH_TO_KNOWLEDGE_BASE = "knowledge_base" # Path where the PDFs are stored
COLLECTION_NAME = "tech_radar" # Name of the collection

# Set the API key by exporting it as an environment variable
genai.configure(api_key=os.environ['GOOGLE_API_KEY']) 

# Make sure qdrant docker container is running
# Connect to the Qdrant server. 
# qdrant = QdrantClient("http://localhost:6333") # If running locally
qdrant = QdrantClient(
    "xyz-example.eu-central.aws.cloud.qdrant.io",
    api_key="<paste-your-api-key-here>",
)

# Function to create embeddings of the text
def create_embedding(text):
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document",
            title="Embedding of single string")
        return result['embedding']
    except Exception as error:
        print(f"Error: {error}")

# Function to create a collection in Qdrant
def create_collection():
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )

# Function to ingest the documents into the collection
def ingest_document():
    documents = []

    # Load the PDFs from the knowledge base
    for file in os.listdir(PATH_TO_KNOWLEDGE_BASE):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(PATH_TO_KNOWLEDGE_BASE, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Total number of chunks: {len(chunked_documents)}")

    # Ingest the chunks as vector embedding into the collection with metadata
    for i, chunk in enumerate(chunked_documents):
        embedding = create_embedding(chunk.page_content)
        try:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=i+1,
                        vector=embedding,
                        payload={
                            "metadata": chunk.metadata,
                            "page_content": chunk.page_content,
                        },
                    )
                ],
            )
            print(f"Chunk ingested: {i+1}")
        except Exception as error:
            print(f"Error: {error}")
            print(f"Chunk not ingested: {i+1}")

if __name__ == "__main__":
    create_collection()
    ingest_document()