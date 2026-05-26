import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PDF2MD
import glob
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json

# 1. Define the directory path
# [span_2](start_span)[span_3](start_span)This allows the script to ingest the Handbook and any other MCS-related PDFs[span_2](end_span)[span_3](end_span)
data_dir = "./RAG_files/"

if not os.path.exists(data_dir):
    print(f"⚠️ Directory not found: {data_dir}. Creating directory...")
    os.makedirs(data_dir)
    print("📂 Please place your PDF files (like handbook.pdf) into the 'data' folder and run again.")
else:
    # 2. Use DirectoryLoader to traverse the entire folder
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))

    print(f"Loading all documents from {data_dir}...")
    docs = []

    for path in pdf_paths:
        docs.extend(PDF2MD.process_pdf_to_markdown(path))


    print("✅ All PDFs processed into Markdown format.")



        # 3. Refined Splitting Logic (Parent-Document Retrieval 形式)
    
    # 建立 Parent（大）切分器：保留完整上下文
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    # 建立 Child（小）切分器：用於精準語義檢索
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        add_start_index=True
    )

    # 先切出大塊的 Parent Documents
    parent_docs = parent_splitter.split_documents(docs)
    
    # 用來存放所有小塊的 Child Chunks，並綁定與 Parent 的關係
    child_chunks = []
    
    # 遍歷每一個 Parent，再切成數個 Child
    for parent_idx, p_doc in enumerate(parent_docs):
        # 替 Parent 加上一個獨一無二的 ID
        parent_id = f"parent_{parent_idx}"
        p_doc.metadata["parent_id"] = parent_id
        
        # 將這個 Parent 切成小的子區塊
        sub_docs = child_splitter.split_documents([p_doc])
        
        # 讓每個子區塊都記住自己屬於哪一個 Parent ID
        for s_doc in sub_docs:
            s_doc.metadata["parent_id"] = parent_id
            # 保留原本 Parent 的 source 和 page 資訊
            s_doc.metadata["source"] = p_doc.metadata.get("source", "Unknown")
            s_doc.metadata["page"] = p_doc.metadata.get("page", "N/A")
            child_chunks.append(s_doc)

    print(f"✅ Parent Documents (大) 總共切出 {len(parent_docs)} 塊。")
    print(f"✅ Child Chunks (小) 總共切出 {len(child_chunks)} 塊 (將用於 Vector DB)。")
    print("-" * 30)

    # 4. Detailed Preview (對比 Parent 與 Child)
    if child_chunks:
        print("\n" + "="*10 + " Parent-Child Retrieval Preview " + "="*10)
        
        # 取得第一個小塊（Child）和它對應的大塊（Parent）
        sample_child = child_chunks[0]
        target_parent_id = sample_child.metadata["parent_id"]
        
        # 找出對應的 Parent 內容
        sample_parent = next(p for p in parent_docs if p.metadata["parent_id"] == target_parent_id)
        
        source_path = sample_child.metadata.get('source', 'Unknown')
        file_name = os.path.basename(source_path) if source_path != 'Unknown' else 'Unknown'
        page_info = sample_child.metadata.get('page', 'N/A')
        page_display = page_info + 1 if isinstance(page_info, int) else page_info

        print(f"📄 Source File : {file_name}")
        print(f"📖 Page        : {page_display}")
        print(f"🆔 Parent ID   : {target_parent_id}")
        print("-" * 48)
        # 呈現小塊（用於搜尋）與大塊（最後餵給 LLM 的範圍）的差別
        print(f"🔍 [Child Chunk 內容 - 用於向量檢索] (字數: {len(sample_child.page_content)}):\n{sample_child.page_content[:200]}...")
        print("-" * 48)
        print(f"📚 [Parent Document 內容 - 最後餵給 LLM] (字數: {len(sample_parent.page_content)}):\n{sample_parent.page_content[:400]}...")
        print("=" * 48)

    # --- 5. Hand-rolled Ingestion (自刻寫入邏輯) ---

    print("1. 初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("2. 正在將 Child Chunks 寫入 ChromaDB...")
    # 如果資料夾已經存在，Chroma 會自動把資料 append 進去
    vector_db = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ ChromaDB 向量資料庫建置完成！")

    print("3. 正在手刻 Docstore (將 Parent Docs 寫入 JSON)...")
    parent_store = {}

    for p_doc in parent_docs:
        parent_id = p_doc.metadata["parent_id"]
        # 以 parent_id 為 Key，將內容與 Metadata 存成 Value
        parent_store[parent_id] = {
            "page_content": p_doc.page_content,
            "metadata": p_doc.metadata
        }

    # 將字典存為 JSON 檔案
    with open("parent_store.json", "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False, indent=2)

    print("✅ 本地端 Lookup Table (parent_store.json) 儲存完成！")
    print("🎉 整個 RAG 前處理與資料庫建置大功告成！")