import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import PDF2MD
from backend import db, config

CHROMA_PERSIST_DIR = config.CHROMA_PERSIST_DIR

def init_embeddings():
    return config.get_embeddings()

import hashlib

def get_file_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ingest_document(file_path, progress_callback=None):
    print(f"Ingesting document: {file_path}")
    
    file_md5 = get_file_md5(file_path)
    existing_doc = db.get_document_by_md5(file_md5)
    
    if existing_doc:
        print(f"Document already ingested (MD5 match: {file_md5}). Skipping parsing.")
        if progress_callback:
            progress_callback(100, f"Cache Hit: '{existing_doc['filename']}' already exists.")
        
        conn = db.sqlite3.connect(db.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM parent_chunks WHERE document_id = ?", (existing_doc["id"],))
        parent_count = cursor.fetchone()[0]
        conn.close()
        return parent_count, parent_count * 5

    if progress_callback:
        progress_callback(5, "Converting PDF to Markdown format...")
        
    # 1. Process PDF to Markdown
    docs = PDF2MD.process_pdf_to_markdown(file_path)
    
    if progress_callback:
        progress_callback(15, "Registering document in SQLite store...")
        
    # 2. Add document to SQLite
    filename = os.path.basename(file_path)
    doc_id = db.add_document(filename, md5_hash=file_md5)
    
    if progress_callback:
        progress_callback(25, "Splitting document into parent and child chunks...")
        
    # 3. Setup Splitters
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        add_start_index=True
    )
    
    # 4. Split and process
    parent_docs = parent_splitter.split_documents(docs)
    child_chunks = []
    
    # Use doc_id to ensure parent_ids are unique across different documents
    for p_idx, p_doc in enumerate(parent_docs):
        parent_id = f"doc_{doc_id}_parent_{p_idx}"
        p_doc.metadata["parent_id"] = parent_id
        
        # Save to SQLite
        db.add_parent_chunk(doc_id, parent_id, p_doc.page_content, p_doc.metadata)
        
        # Create child chunks
        sub_docs = child_splitter.split_documents([p_doc])
        for s_doc in sub_docs:
            s_doc.metadata["parent_id"] = parent_id
            s_doc.metadata["source"] = p_doc.metadata.get("source", filename)
            s_doc.metadata["page"] = p_doc.metadata.get("page", "N/A")
            child_chunks.append(s_doc)
            
    print(f"Adding {len(child_chunks)} child chunks to ChromaDB...")
    if progress_callback:
        progress_callback(40, f"Indexing {len(child_chunks)} child chunks to Chroma database...")
        
    # 5. Add to ChromaDB
    embeddings = init_embeddings()
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    vector_db.add_documents(child_chunks)
    vector_db.persist()
    print(f"✅ Persisted {len(child_chunks)} child chunks to ChromaDB")
    
    if progress_callback:
        progress_callback(55, "Analyzing document structure to generate chapter/section outlines...")
        
    # ---- Hierarchical Document Summary Index ----
    if config.ENABLE_SUMMARY_INDEX:
        from backend import summary_hierarchical
        
        def sub_progress(percent, text):
            if progress_callback:
                # Scale sub-progress from 55% to 95%
                scaled = 55 + int(percent * 0.40)
                progress_callback(scaled, text)
                
        summary_hierarchical.generate_hierarchical_summary(
            docs, doc_id, filename, progress_callback=sub_progress
        )
        print(f"📝 Hierarchical summaries generated and stored for document ID {doc_id}")
    
    if progress_callback:
        progress_callback(100, "Successfully completed document ingestion!")
        
    print("Ingestion complete!")
    return len(parent_docs), len(child_chunks)

def migrate_existing():
    """Migrate the existing parent_store.json to the new DB if needed"""
    import json
    if os.path.exists("parent_store.json"):
        print("Migrating parent_store.json to SQLite...")
        with open("parent_store.json", "r", encoding="utf-8") as f:
            store = json.load(f)
            
        doc_id = db.add_document("handbook_migrated.pdf")
        for parent_id, data in store.items():
            db.add_parent_chunk(doc_id, parent_id, data["page_content"], data["metadata"])
        
        os.rename("parent_store.json", "parent_store.json.bak")
        print("Migration complete.")

if __name__ == "__main__":
    # If run directly, ingest any PDFs in RAG_files
    migrate_existing()
    data_dir = "./RAG_files/"
    if os.path.exists(data_dir):
        pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
        for path in pdf_paths:
            ingest_document(path)
