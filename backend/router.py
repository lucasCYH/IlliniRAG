# backend/router.py

"""Router module for selecting between global summary retrieval and fine‑grained retrieval.

The router inspects the user's query and decides whether to use the Document Summary
Index (global view) or the existing hybrid parent/child retriever (needle view).
"""

from typing import List, Any
import streamlit as st
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from backend.agents import AgentRouter
from . import config


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
    router: AgentRouter = None

    def __init__(self, vector_db: Any, hybrid_retriever: BaseRetriever):
        super().__init__()
        object.__setattr__(self, "vector_db", vector_db)
        object.__setattr__(self, "hybrid_retriever", hybrid_retriever)
        object.__setattr__(self, "router", AgentRouter(hybrid_retriever))

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        """Route the query and return relevant documents.
        """
        # Read user's UI toggle preference if in Streamlit context
        force_mode = None
        try:
            if st.session_state.get("global_mode", False):
                force_mode = "global"
        except Exception:
            pass

        # Perform routing
        agent, decision_reason = self.router.route(query, force_mode=force_mode)
        
        # Save decision for UI visualization
        try:
            st.session_state["last_routing_decision"] = decision_reason
            st.session_state["last_agent_name"] = agent.name
        except Exception:
            pass
            
        print(f"[RouterRetriever] Routing query '{query}' to agent '{agent.name}' (Reason: {decision_reason})")
        
        # Retrieve documents
        return agent.retrieve(query, k=5)

