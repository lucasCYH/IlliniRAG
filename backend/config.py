# backend/config.py

"""Configuration constants for IlliniRAG project.

These can be imported by other modules to keep settings centralized.
"""

# Enable or disable the Document Summary Index
ENABLE_SUMMARY_INDEX = True

# Model name for summary generation (used with Ollama or OpenAI)
SUMMARY_MODEL = "llama3.2"  # change to desired model name

SUMMARY_COLLECTION_CHAPTER = "doc_summary_chapter"
SUMMARY_COLLECTION_SECTION = "doc_summary_section"

# Model for classifying global vs needle queries
GLOBAL_CLASSIFIER_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GLOBAL_CLASSIFIER_MARGIN = 0.1  # confidence margin for routing

# Token count threshold (approx) to decide global vs needle queries
GLOBAL_QUERY_THRESHOLD = 150

# Keywords that indicate a global summarization request
GLOBAL_KEYWORDS = ["overview", "summary", "global", "全貌", "摘要"]

import os
# Paths for database persistence (can be overridden by environment variables for container deployment)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DB_PATH = os.getenv("DB_PATH", "notebook_store.db")

_embeddings_instance = None

def get_embeddings():
    """Get the shared HuggingFaceEmbeddings instance to prevent memory redundancy and PyTorch device conflicts."""
    global _embeddings_instance
    if _embeddings_instance is None:
        print("[Embeddings] Loading shared HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_instance

