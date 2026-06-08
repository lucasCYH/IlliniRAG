from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from backend import db

def generate_study_guide(doc_ids=None):
    llm = Ollama(model="llama3.1", temperature=0.3)
    
    # Get selected documents text. If doc_ids is None, falls back to all docs.
    texts = db.get_parent_chunks_text_by_docs(doc_ids)
    full_text = "\n\n".join(texts)[:50000] 
    
    if not full_text:
        return "No documents found to generate a guide."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert academic assistant. Generate a structured Study Guide and FAQ based on the following document context. Use Markdown formatting."),
        ("human", "Context:\n{context}\n\nPlease generate a Study Guide with key concepts, followed by a FAQ section.")
    ])
    
    chain = prompt | llm
    return chain.invoke({"context": full_text})
