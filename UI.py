import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from backend import db, ingestion, retriever, studio, podcast, router, config

# --- 1. Web UI Basic Settings ---
st.set_page_config(page_title="Local NotebookLM", layout="wide")

# Load Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📚 Local NotebookLM")

# --- 2. Resource Loading Area ---
@st.cache_resource(show_spinner=False)
def load_heavy_models():
    embeddings = config.get_embeddings()
    llm = Ollama(model="llama3.1", temperature=0)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return embeddings, llm, reranker

@st.cache_resource(show_spinner=False)
def load_data_connections(_embeddings, _reranker):
    vector_db = Chroma(persist_directory=config.CHROMA_PERSIST_DIR, embedding_function=_embeddings)
    hybrid_retriever = retriever.init_hybrid_retriever(vector_db, reranker=_reranker)
    # Use router retriever to handle global vs fine‑grained queries
    custom_retriever = router.RouterRetriever(vector_db, hybrid_retriever)
    return vector_db, custom_retriever

with st.spinner("🚀 Initializing AI Models and Loading Database..."):
    embeddings, llm, reranker = load_heavy_models()
    vector_db, custom_retriever = load_data_connections(embeddings, reranker)

# --- 3. Core RAG Logic ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, custom_retriever, contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert AI assistant. Answer the question based ONLY on the provided context.\n\n"
        "Instructions:\n"
        "- Answer the user's question completely. If the question has multiple parts, address each part step-by-step.\n"
        "- Do not skip any nuance, different experimental settings, or comparisons mentioned in the context (e.g. classification vs. detection settings).\n"
        "- IMPORTANT: You MUST answer in the exact same language as the user's input. If the user asks in Traditional Chinese (繁體中文), you must reply in Traditional Chinese.\n"
        "- Do not make up answers. If the information is not in the context, state that it is not available.\n\n"
        "Context: {context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. Sidebar: Document Management ---
with st.sidebar:
    st.header("📂 Data Sources")
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")
    if uploaded_file:
        # Fullscreen loading overlay to dim the screen and block all interactions
        st.markdown("""
            <div class="upload-overlay">
                <div class="upload-loader"></div>
                <div class="upload-text">📂 檔案導入中，請稍候... (請勿點擊或重新整理網頁)</div>
            </div>
            <style>
            .upload-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: rgba(0, 0, 0, 0.7);
                z-index: 999999;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                pointer-events: all;
            }
            .upload-loader {
                border: 6px solid #444;
                border-top: 6px solid #3498db;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1.2s linear infinite;
                margin-bottom: 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .upload-text {
                color: #ffffff;
                font-size: 22px;
                font-weight: 600;
                font-family: sans-serif;
            }
            </style>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0, text="Preparing file...")
        # Save temp file
        temp_path = f"/tmp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        def update_progress(percent, text):
            progress_bar.progress(percent, text=text)
            
        # Ingest with progress callback
        enable_summary = st.session_state.get("enable_summary_index", True)
        ingestion.ingest_document(temp_path, progress_callback=update_progress, enable_summary=enable_summary)
        st.success(f"Successfully ingested {uploaded_file.name}")
        # Reset uploader and reload retriever
        st.session_state.uploader_key += 1
        load_data_connections.clear()
        import time
        time.sleep(1)
        st.rerun()
            
    st.divider()
    st.subheader("⚙️ Settings")
    st.toggle("全域摘要模式 (Global Mode)", key="global_mode", value=False, help="開啟後將強制使用全域大綱/摘要檢索模式；關閉時則依問題自動進行語意路由。")
    st.toggle("啟用文件摘要索引 (Generate Summaries)", key="enable_summary_index", value=True, help="開啟後將在導入時自動生成章節與段落摘要以支援全域大綱檢索；關閉可大幅加快上傳速度。")

    st.divider()
    st.subheader("Uploaded Documents")
    docs = db.get_all_documents()
    for d in docs:
        # 每筆文件外層包一層 .doc-item，供 CSS 控制 hover 顯示
        st.markdown(f'<div class="doc-item">', unsafe_allow_html=True)
        col_text, col_btn = st.columns([4, 1])
        col_text.markdown(f"📄 {d['filename']}")
        # 刪除按鈕僅在 hover 時顯示，樣式會在 CSS 中定義
        if col_btn.button("✖", key=f"del_{d['id']}"):
            db.delete_document(d['id'])
            try:
                vector_db._collection.delete(where={"source": d['filename']})
            except Exception:
                pass
            
            # Also delete summaries
            try:
                from backend.summary_hierarchical import _get_chapter_store, _get_section_store
                _get_chapter_store()._collection.delete(where={"doc_id": d['id']})
                _get_section_store()._collection.delete(where={"doc_id": d['id']})
            except Exception:
                pass

            st.toast(f"🗑️ 檔案 **{d['filename']}** 已被刪除", icon="🗑️")
            load_data_connections.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# --- 5. Main Layout: 2 Columns ---
col1, col2 = st.columns([1, 1])

# Left Column: Workspace
with col1:
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Notes", "🎙️ Studio", "🔍 Document Viewer", "📊 Summary Viewer"])
    
    with tab1:
        st.subheader("Your Notes")
        new_note = st.text_area("Write a new note...", height=100)
        if st.button("Save Note") and new_note:
            db.add_note(new_note)
            st.success("Note saved!")
            st.rerun()
            
        st.divider()
        st.subheader("Saved Notes")
        
        if "editing_note_id" not in st.session_state:
            st.session_state.editing_note_id = None
            
        notes = db.get_all_notes()
        if not notes:
            st.caption("No notes saved yet.")
        else:
            for note in notes:
                note_id = note['id']
                if st.session_state.editing_note_id == note_id:
                    edited_content = st.text_area(
                        "Edit Note Content:",
                        value=note['content'],
                        key=f"edit_content_{note_id}",
                        height=100
                    )
                    col_save, col_cancel = st.columns([1, 6])
                    if col_save.button("💾 儲存", key=f"save_note_{note_id}"):
                        db.update_note(note_id, edited_content)
                        st.session_state.editing_note_id = None
                        st.toast("📝 筆記已成功更新", icon="✅")
                        st.rerun()
                    if col_cancel.button("❌ 取消", key=f"cancel_note_{note_id}"):
                        st.session_state.editing_note_id = None
                        st.rerun()
                else:
                    st.info(note['content'])
                    col_edit, col_del, _ = st.columns([1, 1, 5])
                    if col_edit.button("📝 編輯", key=f"btn_edit_{note_id}"):
                        st.session_state.editing_note_id = note_id
                        st.rerun()
                    if col_del.button("🗑️ 刪除", key=f"btn_del_note_{note_id}"):
                        db.delete_note(note_id)
                        st.toast("🗑️ 筆記已刪除", icon="🗑️")
                        st.rerun()
                st.divider()
            
    with tab2:
        st.subheader("Notebook Studio")
        documents = db.get_all_documents()
        if not documents:
            st.info("No documents ingested yet. Upload a PDF to start using the Studio!")
        else:
            selected_studio_docs = st.multiselect(
                "選擇要納入生成範圍的文件 (留空則預設為全部文件)：",
                options=documents,
                format_func=lambda d: d["filename"],
                key="studio_select_docs"
            )
            selected_ids = [d["id"] for d in selected_studio_docs] if selected_studio_docs else None
            
            if st.button("📄 Generate Study Guide"):
                with st.spinner("Analyzing selected documents..."):
                    guide = studio.generate_study_guide(selected_ids)
                    st.markdown(guide)
                    
            if st.button("🎧 Generate Audio Podcast"):
                with st.spinner("Writing script and synthesizing voices..."):
                    res = podcast.generate_podcast_audio(selected_ids)
                    warning_msg = None
                    if res and len(res) == 3:
                        audio_file, script, warning_msg = res
                    else:
                        audio_file, script = res
                        
                    if warning_msg:
                        st.warning(warning_msg)
                        
                    if audio_file:
                        st.audio(audio_file)
                    st.markdown("**Podcast Script:**")
                    st.text(script)
                
    with tab3:
        st.subheader("🔍 Document Viewer")
        st.caption("Rendered preview of your ingested documents.")
        documents = db.get_all_documents()
        if not documents:
            st.info("No documents ingested yet. Upload a PDF from the sidebar to get started!")
        else:
            selected_viewer_doc = st.selectbox(
                "選擇要檢視的檔案：",
                options=documents,
                format_func=lambda d: d["filename"],
                key="viewer_select_doc"
            )
            if selected_viewer_doc:
                chunks = db.get_parent_chunks_by_document(selected_viewer_doc["id"])
                if chunks:
                    with st.container(height=500, border=True):
                        full_text = "\n\n---\n\n".join(chunks)
                        st.markdown(full_text)
                else:
                    st.warning("No content chunks found for this document.")

    with tab4:
        st.subheader("📊 Summary Viewer")
        st.caption("Browse chapter & section outlines generated for your documents.")
        documents = db.get_all_documents()
        if not documents:
            st.info("No documents ingested yet. Upload a PDF from the sidebar to generate summaries!")
        else:
            selected_doc = st.selectbox(
                "Select Document to View Summary:",
                options=documents,
                format_func=lambda d: d["filename"],
                key="summary_select_doc"
            )
            if selected_doc:
                from backend import summary_hierarchical
                with st.spinner("Loading outline hierarchy..."):
                    hierarchy_data = summary_hierarchical.get_document_hierarchy(selected_doc["id"])
                    hierarchy = hierarchy_data["hierarchy"]
                
                if not hierarchy:
                    st.warning("No hierarchical summaries found for this document. Try re-uploading to generate them.")
                else:
                    st.markdown(f"### Outline of `{hierarchy_data['filename']}`")
                    for ch_title, ch_info in hierarchy.items():
                        with st.expander(f"📖 Chapter: {ch_title}", expanded=True):
                            st.markdown(f"**Chapter Summary:**\n{ch_info['summary']}")
                            
                            if ch_info["sections"]:
                                st.markdown("---")
                                st.markdown("**Sections in this chapter:**")
                                for sec_title, sec_summary in ch_info["sections"].items():
                                    st.markdown(f"##### 🔗 {sec_title}")
                                    st.info(sec_summary)

# Right Column: Chat Interface
def render_messages_reversed(messages_list):
    """
    純粹、乾淨的歷史紀錄倒序渲染器（不包含任何 Spinner 與模型推理邏輯）
    """
    turns = []
    i = 0
    while i < len(messages_list):
        if i + 1 < len(messages_list) and messages_list[i]["role"] == "user" and messages_list[i+1]["role"] == "assistant":
            turns.append([messages_list[i], messages_list[i+1]])
            i += 2
        else:
            turns.append([messages_list[i]])
            i += 1
            
    for idx, turn in enumerate(reversed(turns)):
        if idx > 0:
            st.divider()
        for msg in turn:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

with col2:
    st.subheader("💬 Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    user_input = st.chat_input("Ask about your documents...")
    
    # ---------------------------------------------------------
    # 【核心破局點 1】物理劃分地盤
    # 無論有沒有輸入、模型有沒有在跑，這兩個容器的相對位置在網頁上一開始就定死！
    # ---------------------------------------------------------
    active_turn_container = st.container()  # 最上方：永遠只放當前這一輪 (Q2 + Thinking)
    history_container = st.container()      # 下方：老老實實放過去所有的歷史對話 (Q1 + A1)
    
    # ---------------------------------------------------------
    # 【核心破局點 2】在腳本一啟動，立刻在下方渲染現有的所有歷史紀錄
    # 這樣當你輸入 Q2 的瞬間，Q1 和 A1 就已經穩穩地躺在下方了，絕對不會因為 Spinner 轉圈而被凍結隱藏！
    # ---------------------------------------------------------
    if st.session_state.messages:
        with history_container:
            st.divider()
            st.caption("⌛ 歷史對話 (History)")
            render_messages_reversed(st.session_state.messages)
            
    # ---------------------------------------------------------
    # 【核心破局點 3】處理當前新輸入的 Q2 與推理
    # ---------------------------------------------------------
    if user_input:
        # 清空路由快取
        st.session_state["last_routing_decision"] = ""
        st.session_state["last_agent_name"] = ""
        
        # 回到最上方的活動對話框，單獨渲染當前這一輪
        with active_turn_container:
            # 1. 渲染新問題 Q2
            with st.chat_message("user"):
                st.markdown(user_input)
                
            # 2. 渲染專屬於 Q2 的助理對話框與 Spinner
            # 此時，上方的對話框內只有 Spinner 在轉，而下方的 history_container 早已渲染完畢，兩者完美並存！
            with st.chat_message("assistant"):
                with st.spinner("Thinking... (The first global query may take a few seconds to load the routing model)"):
                    try:
                        response = rag_chain.invoke({
                            "input": user_input,
                            "chat_history": st.session_state.chat_history
                        })
                        answer = response["answer"]
                        
                        # 處理與格式化引用來源
                        citation_texts = []
                        seen_citations = set()
                        for doc in response["context"]:
                            source = doc.metadata.get("source", "Unknown")
                            chapter_info = doc.metadata.get("chapter_info")
                            if not chapter_info:
                                h1 = doc.metadata.get("Header 1")
                                h2 = doc.metadata.get("Header 2")
                                if h1 and h2:
                                    chapter_info = f"{h1} > {h2}"
                                elif h1:
                                    chapter_info = h1
                                else:
                                    chapter_info = "Section details"
                                
                                citation = f"`{source}` ({chapter_info})"
                                if citation not in seen_citations:
                                    seen_citations.add(citation)
                                    citation_texts.append(citation)

                        if citation_texts:
                            answer += "\n\n**Sources:**\n" + "\n".join([f"- {c}" for c in citation_texts])
                        
                        decision_reason = st.session_state.get("last_routing_decision", "")
                        agent_name = st.session_state.get("last_agent_name", "")
                        if agent_name:
                            answer += f"\n\n*(Routed via **{agent_name}** - Reason: {decision_reason})*"
                            
                        # 3. 轉圈圈結束，原地把答案 A2 渲染出來
                        st.markdown(answer)
                        
                        with st.expander("🔍 View Raw Context"):
                            for i, doc in enumerate(response["context"]):
                                source = doc.metadata.get('source', 'Unknown')
                                st.write(f"**Source {i+1}: {source}**")
                                w_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                st.write(w_content)
                                st.divider()
                                
                        # 4. 【高光時刻】當這一輪完美落幕後，我們才把 Q2 和 A2 寫入 session_state
                        # 這樣一來，下一次你再輸入 Q3 時，這一輪就會自動被上面的 history_container 收納進歷史紀錄中
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.chat_history.extend([
                            HumanMessage(content=user_input),
                            AIMessage(content=answer)
                        ])
                        
                        # 5. 強制重繪，讓這一輪生成的 A2 順利融入下方的歷史紀錄大軍中，完美翻轉
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {e}")