import json
import streamlit as st
from typing import List
from pydantic import Field
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# --- 1. Web UI Basic Settings ---
st.set_page_config(page_title="UIUC MCS Assistant", layout="centered")
st.title("🎓 UIUC MCS Knowledge Assistant")

# --- 2. Resource Loading Area (with Caching) ---
@st.cache_resource
def load_resources():
    # 載入 Embedding 模型與 ChromaDB (負責 Child Chunks)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # 載入 LLM
    llm = Ollama(model="llama3", temperature=0)
    
    # 載入我們手刻的 Docstore (負責 Parent Documents)
    with open("parent_store.json", "r", encoding="utf-8") as f:
        parent_store = json.load(f)
        
    return vector_db, llm, parent_store

vector_db, llm, parent_store = load_resources()

# --- 3. 定義自刻的雙層檢索器 (Custom Parent Retriever) ---
class CustomParentRetriever(BaseRetriever):
    # 使用 Pydantic 的 Field 來定義類別屬性，這是 LangChain BaseRetriever 的標準寫法
    vector_db: Chroma = Field(description="The underlying vector store for child chunks")
    parent_store: dict = Field(description="The dictionary mapping parent_ids to parent content")
    search_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        核心檢索邏輯：向量搜尋小塊 -> 提取 ID -> 去 JSON 組裝大塊 -> 排除重複
        """
        # 1. 搜尋小塊 (Child Chunks)
        child_docs = self.vector_db.similarity_search(query, k=self.search_k)
        
        unique_parent_ids = set()
        final_parent_docs = []
        
        # 2. 透過 parent_id 組裝大塊 (Parent Documents)
        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if pid and pid not in unique_parent_ids:
                unique_parent_ids.add(pid)
                
                # 從 JSON 查找表提取原始完整內容
                if pid in self.parent_store:
                    parent_data = self.parent_store[pid]
                    doc = Document(
                        page_content=parent_data["page_content"],
                        metadata=parent_data["metadata"]
                    )
                    final_parent_docs.append(doc)
                    
        return final_parent_docs

# 初始化我們剛剛做好的自訂檢索器
# search_k=10 代表先撈 10 個小塊，最後還原出來的大塊通常會少於 10 個 (因為會去重)
custom_retriever = CustomParentRetriever(
    vector_db=vector_db, 
    parent_store=parent_store, 
    search_k=10
)

# --- 4. Core RAG Logic (with Conversational Memory) ---

# Step 4.1: Create a "History-Aware Retriever"
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 🔥 關鍵修改：將原本的 vector_db.as_retriever 換成我們的 custom_retriever
history_aware_retriever = create_history_aware_retriever(
    llm, custom_retriever, contextualize_q_prompt
)

# Step 4.2: Create the final QA Chain
qa_system_prompt = (
    "You are an expert assistant for UIUC Master of Computer Science students. "
    "Use the following retrieved context to answer the user's question accurately. "
    "If you cannot find the answer in the context, just say 'I do not know based on the provided documents'. "
    "Keep your answers clear, concise, and professional."
    "\n\n"
    "Context: {context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Step 4.3: Combine into the complete RAG Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 5. Chat UI Implementation ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything about the UIUC MCS program..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching the files..."):
            try:
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.markdown(answer)
                
                # 🔥 優化 UI 的來源顯示：展現 Markdown 章節層級與 Parent 內容
                with st.expander("🔍 View Source Documents (Parent Context)"):
                    for i, doc in enumerate(response["context"]):
                        # 擷取 Markdown 結構標題，如果沒有則顯示 Unknown
                        h1 = doc.metadata.get('Header 1', '')
                        h2 = doc.metadata.get('Header 2', '')
                        h3 = doc.metadata.get('Header 3', '')
                        page_num = doc.metadata.get('page', 'N/A')
                        
                        # 把有值的標題串起來 (例如: "POLICIES & PROCEDURES > Registration")
                        chapter_path = " > ".join(filter(None, [h1, h2, h3]))
                        if not chapter_path:
                            chapter_path = f"Page {page_num}"
                            
                        st.write(f"**Source {i+1}: {chapter_path}**")
                        # 顯示完整的 Parent Context，讓使用者知道 LLM 參考了多完整的段落
                        st.write(doc.page_content)
                        st.divider()
                        
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=answer)
                ])
                
            except Exception as e:
                st.error(f"An error occurred: {e}")