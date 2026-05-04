import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. 網頁介面基本設定 ---
st.set_page_config(page_title="UIUC MCS 助理", layout="centered")
st.title("🎓 UIUC MCS 知識小助手")

# --- 2. 資源載入區 (加入快取機制) ---
@st.cache_resource
def load_resources():
    # 載入詞向量模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # 讀取本地已建好的 ChromaDB
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # 連結本地模型
    llm = Ollama(model="llama3", temperature=0)
    return vector_db, llm

vector_db, llm = load_resources()

# --- 3. 核心 RAG 處理邏輯 (加入對話記憶) ---

# 步驟 3.1: 建立「具備歷史意識的檢索器」
# 這個 Prompt 的作用是教導 AI 看著上下文，把代名詞還原成完整的搜尋關鍵字
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

# 讓檢索器具備改寫問題的能力
history_aware_retriever = create_history_aware_retriever(
    llm, vector_db.as_retriever(search_kwargs={"k": 10}), contextualize_q_prompt
)

# 步驟 3.2: 建立最終問答的 Chain
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

# 步驟 3.3: 組合完整的 RAG 鍊
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. 聊天對話 UI 實作 ---

# 紀錄 Streamlit UI 顯示用的訊息
if "messages" not in st.session_state:
    st.session_state.messages = []

# 紀錄 LangChain 底層運算用的對話歷史 (HumanMessage 與 AIMessage 物件)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 渲染歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 處理使用者新輸入的問題
if user_input := st.chat_input("想問關於 UIUC MCS 的事嗎？"):
    # 顯示並儲存使用者的問題 (UI 用)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成並顯示 AI 的回答
    with st.chat_message("assistant"):
        with st.spinner("正在思考與查閱手冊資料..."):
            try:
                # 將輸入問題與對話歷史一起餵給模型
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                
                # 輸出文字結果
                st.markdown(answer)
                
                # 建立可展開的參考資料區塊
                with st.expander("🔍 檢視參考資料來源"):
                    for i, doc in enumerate(response["context"]):
                        page_num = doc.metadata.get('page', 'N/A')
                        st.write(f"**來源 {i+1} (第 {page_num} 頁):**")
                        st.write(doc.page_content)
                        st.divider()
                        
                # 更新 UI 訊息紀錄
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # 更新 LangChain 對話歷史紀錄
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=answer)
                ])
                
            except Exception as e:
                st.error(f"執行時發生錯誤: {e}")