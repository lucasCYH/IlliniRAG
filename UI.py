import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Web UI Basic Settings ---
st.set_page_config(page_title="UIUC MCS Assistant", layout="centered")
st.title("🎓 UIUC MCS Knowledge Assistant")

# --- 2. Resource Loading Area (with Caching) ---
@st.cache_resource
def load_resources():
    # Load embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load the local ChromaDB
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # Connect to local LLM
    llm = Ollama(model="llama3", temperature=0)
    return vector_db, llm

vector_db, llm = load_resources()

# --- 3. Core RAG Logic (with Conversational Memory) ---

# Step 3.1: Create a "History-Aware Retriever"
# This prompt teaches the AI to use context to resolve pronouns into standalone search queries
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

# Enable the retriever to reformulate questions
history_aware_retriever = create_history_aware_retriever(
    llm, vector_db.as_retriever(search_kwargs={"k": 10}), contextualize_q_prompt
)

# Step 3.2: Create the final QA Chain
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

# Step 3.3: Combine into the complete RAG Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 4. Chat UI Implementation ---

# Record messages for Streamlit UI display
if "messages" not in st.session_state:
    st.session_state.messages = []

# Record chat history for LangChain backend (HumanMessage and AIMessage objects)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process new user input
if user_input := st.chat_input("Ask me anything about the UIUC MCS program..."):
    # Display and store user's question (for UI)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display AI's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching the handbook..."):
            try:
                # Feed the input question and chat history to the model
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                
                # Output text result
                st.markdown(answer)
                
                # Create an expandable reference section
                with st.expander("🔍 View Source Documents"):
                    for i, doc in enumerate(response["context"]):
                        page_num = doc.metadata.get('page', 'N/A')
                        st.write(f"**Source {i+1} (Page {page_num}):**")
                        st.write(doc.page_content)
                        st.divider()
                        
                # Update UI message record
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                # Update LangChain chat history record
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=answer)
                ])
                
            except Exception as e:
                st.error(f"An error occurred: {e}")