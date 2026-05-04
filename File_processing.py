from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load PDF (Ensure the filename matches the PDF in your project folder)
pdf_path = "./RAG project/handbook.pdf" 
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"✅ Successfully loaded document: {len(docs)} pages total")

# 2. Split the document into small chunks
# chunk_size=500: Each chunk is approximately 500 characters
# chunk_overlap=50: Overlap ensures context is preserved between chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)

print(f"✅ Document successfully split into {len(chunks)} chunks")
print("-" * 30)
print("🔍 Previewing the content of the first chunk:")
print(chunks[0].page_content)