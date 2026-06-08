# backend/config.py

"""Configuration constants for IlliniRAG project.

These can be imported by other modules to keep settings centralized.
"""

# Enable or disable the Document Summary Index
ENABLE_SUMMARY_INDEX = True

# Model name for summary generation (used with Ollama or OpenAI)
SUMMARY_MODEL = "llama3.1"  # change to desired model name

# Chroma collection name for storing document summaries
SUMMARY_COLLECTION = "doc_summaries"

# Token count threshold (approx) to decide global vs needle queries
GLOBAL_QUERY_THRESHOLD = 150

# Keywords that indicate a global summarization request
GLOBAL_KEYWORDS = ["overview", "summary", "global", "全貌", "摘要"]
