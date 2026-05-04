import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

# --- 3. 核心 RAG 處理邏輯 ---
# 定義系統角色與提示詞
system_prompt = (
    "You are an expert assistant for UIUC Master of Computer Science students. "
    "Use the following retrieved context to answer the user's question accurately. "
    "If you cannot find the answer in the context, just say 'I do not know based on the provided documents'. "
    "Keep your answers clear, concise, and professional."
    "\n\n"
    "Context: {context}"
)

# 建立提示詞模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 組合檢索與生成鍊
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain)

# --- 4. 聊天對話 UI 實作 ---
# 初始化對話紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 處理使用者新輸入的問題
if user_input := st.chat_input("想問關於 UIUC MCS 的事嗎？"):
    # 顯示並儲存使用者的問題
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 生成並顯示 AI 的回答
    with st.chat_message("assistant"):
        with st.spinner("正在查閱手冊資料..."):
            try:
                # 執行推論
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                
                # 輸出文字結果
                st.markdown(answer)
                
                # 建立可展開的參考資料區塊
                with st.expander("🔍 檢視參考資料來源"):
                    for i, doc in enumerate(response["context"]):
                        # 嘗試抓取頁碼，若無則顯示 N/A
                        page_num = doc.metadata.get('page', 'N/A')
                        st.write(f"**來源 {i+1} (第 {page_num} 頁):**")
                        st.write(doc.page_content)
                        st.divider()
                        
                # 將回答存入歷史紀錄
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"執行時發生錯誤: {e}")