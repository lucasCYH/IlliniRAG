from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 載入 PDF (請確保檔名與你資料夾中的 PDF 一致)
pdf_path = "./RAG project/handbook.pdf" 
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"✅ 成功載入文件，共 {len(docs)} 頁")

# 2. 將文件切割成小區塊 (Chunks)
# chunk_size=500 代表每個區塊大約 500 個字元
# chunk_overlap=50 代表區塊之間會有 50 個字元的重疊，避免上下文斷句斷在奇怪的地方
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)

print(f"✅ 文件已成功切割成 {len(chunks)} 個區塊")
print("-" * 30)
print("🔍 預覽第一個區塊的內容：")
print(chunks[0].page_content)