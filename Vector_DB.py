from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 1. Load and split the document (ensuring data is in memory)
loader = PyPDFLoader("./RAG project/handbook.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 2. Define the Embedding Model (using a free open-source model from Hugging Face)
# This model will run locally on your M4 Mac to convert text into vectors
print("🚀 Loading the Embedding model (this may take a moment to download initially)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create and save the vector database to the local drive
persist_directory = "./chroma_db" # The database will be stored in this directory
print(f"📦 Creating the vector database and saving it to {persist_directory}...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("✅ Vector database created successfully!")