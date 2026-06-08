import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from backend import db, ingestion, retriever, studio, podcast, router

# --- 1. Web UI Basic Settings ---
st.set_page_config(page_title="Local NotebookLM", layout="wide")

# Load Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📚 Local NotebookLM")

# --- 2. Resource Loading Area ---
@st.cache_resource(show_spinner=False)
def load_heavy_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = Ollama(model="llama3.1", temperature=0)
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return embeddings, llm, reranker

@st.cache_resource(show_spinner=False)
def load_data_connections(_embeddings, _reranker):
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=_embeddings)
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
    ("system", "You are an expert AI assistant. Answer the question based ONLY on the provided context. IMPORTANT: You MUST answer in the exact same language as the user's input. If the user asks in Traditional Chinese (繁體中文), you must reply in Traditional Chinese. Do not make up answers. Context: {context}"),
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
        with st.spinner("Ingesting document..."):
            # Save temp file
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Ingest
            ingestion.ingest_document(temp_path)
            st.success(f"Ingested {uploaded_file.name}")
            # Reset uploader and reload retriever
            st.session_state.uploader_key += 1
            load_data_connections.clear()
            st.rerun()
            
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
            st.toast(f"🗑️ 檔案 **{d['filename']}** 已被刪除", icon="🗑️")
            load_data_connections.clear()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# --- 5. Main Layout: 2 Columns ---
col1, col2 = st.columns([1, 1])

# Left Column: Workspace
with col1:
    tab1, tab2, tab3 = st.tabs(["📝 Notes", "🎙️ Studio", "🔍 Document Viewer"])
    
    with tab1:
        st.subheader("Your Notes")
        new_note = st.text_area("Write a new note...")
        if st.button("Save Note") and new_note:
            db.add_note(new_note)
            st.success("Note saved!")
            st.rerun()
            
        notes = db.get_all_notes()
        for note in notes:
            st.info(note['content'])
            
    with tab2:
        st.subheader("Notebook Studio")
        if st.button("📄 Generate Study Guide"):
            with st.spinner("Analyzing all documents..."):
                guide = studio.generate_study_guide()
                st.markdown(guide)
                
        if st.button("🎧 Generate Audio Podcast"):
            with st.spinner("Writing script and synthesizing voices..."):
                audio_file, script = podcast.generate_podcast_audio()
                if audio_file:
                    st.audio(audio_file)
                st.markdown("**Podcast Script:**")
                st.text(script)
                
    with tab3:
        st.subheader("🔍 Document Viewer")
        st.caption("Rendered preview of your ingested documents.")
        chunks = db.get_all_parent_chunks_text()
        if chunks:
            with st.container(height=500, border=True):
                # Join chunks and render as markdown
                full_text = "\n\n---\n\n".join(chunks)
                st.markdown(full_text)
        else:
            st.info("No documents ingested yet. Upload a PDF from the sidebar to get started!")

# Right Column: Chat Interface
with col2:
    st.subheader("💬 Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if user_input := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke({
                        "input": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Source Documents"):
                        for i, doc in enumerate(response["context"]):
                            source = doc.metadata.get('source', 'Unknown')
                            st.write(f"**Source {i+1}: {source}**")
                            st.write(doc.page_content[:300] + "...")
                            st.divider()
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.session_state.chat_history.extend([
                        HumanMessage(content=user_input),
                        AIMessage(content=answer)
                    ])
                except Exception as e:
                    st.error(f"An error occurred: {e}")