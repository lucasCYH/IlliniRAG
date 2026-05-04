from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 1. 重新執行載入與切割 (確保資料在記憶體中)
loader = PyPDFLoader("./RAG project/handbook.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 2. 定義 Embedding 模型 (使用 Hugging Face 的免費開源模型)
# 這個模型會在你的 M4 Mac 上跑，將文字轉換成向量
print("🚀 正在加載 Embedding 模型 (這可能需要一點時間下載)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. 建立並儲存向量資料庫到硬碟
persist_directory = "./chroma_db" # 資料庫會存在這個資料夾
print(f"📦 正在建立向量資料庫並存至 {persist_directory}...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("✅ 向量資料庫建立完成！")