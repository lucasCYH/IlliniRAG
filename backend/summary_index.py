# backend/summary_index.py

"""Document Summary Index utilities.

Provides functions to generate concise summaries for whole documents using an LLM
and store/retrieve them in a dedicated Chroma collection.
"""

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.llms import Ollama

from . import config

# Initialize embeddings (shared with other vector stores)
def _init_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize or retrieve the summary Chroma collection
def _get_summary_store():
    embeddings = _init_embeddings()
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name=config.SUMMARY_COLLECTION,
    )
    return vector_db

def generate_summary(text: str) -> str:
    """Generate a concise summary of *text* using the configured LLM.

    The function uses the Ollama model defined in ``config.SUMMARY_MODEL``.
    It returns a short paragraph (≈150 words).
    """
    llm = Ollama(model=config.SUMMARY_MODEL, temperature=0)
    prompt = (
        "Summarize the following document in a concise paragraph (max 150 words).\n\n"
        + text
    )
    # ``invoke`` returns a string response for Ollama LLMs.
    summary = llm.invoke(prompt)
    return summary.strip()

def add_summary(doc_id: int, summary_text: str, source: str) -> None:
    """Add a summary document to the summary collection.

    ``doc_id`` is the identifier from the main SQLite document table.
    ``source`` is usually the original filename.
    """
    store = _get_summary_store()
    doc = Document(
        page_content=summary_text,
        metadata={"doc_id": doc_id, "source": source},
    )
    store.add_documents([doc])
    store.persist()

def search_summary(query: str, k: int = 5) -> List[Document]:
    """Retrieve the most relevant document summaries for *query*.
    """
    store = _get_summary_store()
    return store.similarity_search(query, k=k)
