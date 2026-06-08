# backend/router.py

"""Router module for selecting between global summary retrieval and fine‑grained retrieval.

The router inspects the user's query and decides whether to use the Document Summary
Index (global view) or the existing hybrid parent/child retriever (needle view).
"""

from typing import List, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from . import config, summary_index


class RouterRetriever(BaseRetriever):
    """A retriever that routes queries to either the summary index or the hybrid retriever.

    Parameters
    ----------
    vector_db: Any
        The Chroma vector store for child chunks (passed through to the hybrid retriever).
    hybrid_retriever: BaseRetriever
        The existing hybrid retriever handling fine‑grained searches.
    """

    vector_db: Any = None
    hybrid_retriever: BaseRetriever = None
    def __init__(self, vector_db: Any, hybrid_retriever: BaseRetriever):
        super().__init__()
        object.__setattr__(self, "vector_db", vector_db)
        object.__setattr__(self, "hybrid_retriever", hybrid_retriever)

    class Config:
        arbitrary_types_allowed = True
    def _is_global_query(self, query: str) -> bool:
        """Determine if *query* should be handled by the summary index.

        Heuristics:
        * Token count exceeds ``config.GLOBAL_QUERY_THRESHOLD``.
        * Presence of any keyword in ``config.GLOBAL_KEYWORDS`` (case‑insensitive).
        """
        # Token count heuristic (simple whitespace split)
        token_count = len(query.split())
        if token_count >= config.GLOBAL_QUERY_THRESHOLD:
            return True
        # Keyword heuristic
        lowered = query.lower()
        for kw in config.GLOBAL_KEYWORDS:
            if kw.lower() in lowered:
                return True
        return False

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Route the query and return relevant documents.
        """
        if config.ENABLE_SUMMARY_INDEX and self._is_global_query(query):
            # Use the summary collection
            return summary_index.search_summary(query, k=5)
        else:
            # Defer to the hybrid retriever for fine‑grained results
            # ``HybridParentRetriever`` implements ``_get_relevant_documents``
            return self.hybrid_retriever._get_relevant_documents(query, run_manager=run_manager)
