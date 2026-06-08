# backend/agents/__init__.py

"""Agent framework for IlliniRAG.

Defines BaseAgent, GlobalAgent, NeedleAgent, and AgentRouter to orchestrate
routing of queries between global-level summaries and fine-grained chunks.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np
import streamlit as st
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend import config, db
from backend.summary_hierarchical import _get_chapter_store, _get_section_store

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve the most relevant documents for the query."""
        pass

class GlobalAgent(BaseAgent):
    def __init__(self):
        super().__init__("GlobalAgent")

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Retrieves chapter/section summaries that match the global query."""
        chapter_store = _get_chapter_store()
        section_store = _get_section_store()

        try:
            ch_docs = chapter_store.similarity_search(query, k=max(1, k // 2))
        except Exception as e:
            print(f"Error searching chapter summaries: {e}")
            ch_docs = []

        try:
            sec_docs = section_store.similarity_search(query, k=max(1, k // 2))
        except Exception as e:
            print(f"Error searching section summaries: {e}")
            sec_docs = []

        all_docs = ch_docs + sec_docs
        
        # Populate custom metadata field for citation in UI
        for doc in all_docs:
            level = doc.metadata.get("level", "summary")
            title = doc.metadata.get("title", "")
            doc.metadata["chapter_info"] = f"{level.capitalize()}: {title}"
            
        return all_docs[:k]

class NeedleAgent(BaseAgent):
    def __init__(self, hybrid_retriever):
        super().__init__("NeedleAgent")
        self.hybrid_retriever = hybrid_retriever

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Delegates to the fine-grained HybridParentRetriever."""
        self.hybrid_retriever.final_k = k
        return self.hybrid_retriever.invoke(query)

class EmbeddingClassifier:
    def __init__(self):
        print("Loading SentenceTransformer model for router classification...")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.GLOBAL_CLASSIFIER_MODEL)
        self.global_examples = [
            "Summarize the entire document",
            "What is the overall structure of this book",
            "Give me a global summary of the chapters",
            "What are the main topics discussed in this handbook",
            "提供這份文件的章節大綱與摘要",
            "整篇手冊的重點是什麼",
            "這本書主要在講什麼",
            "請給我全域的概述",
            "Who is this document for overall?",
            "What is a summary of the chapters?",
            "Can you write a study guide of this document?",
            "說明這份手冊的整體架構",
            "What is the handbook outline?"
        ]
        self.needle_examples = [
            "What is the exact stipend for PhD students?",
            "Where is the department office located?",
            "Who is the director of graduate studies?",
            "How many credit hours are required for graduation?",
            "What are the prerequisite courses for CS 598?",
            "如何申請差旅補助與相關規定",
            "辦公室的電話號碼是多少",
            "Fellowship的申請截止日期是哪天",
            "How do I submit an thesis advisor request?",
            "Tell me about the specific fellowships offered.",
            "Who handles graduate admissions?",
            "What is the email for academic advisor?"
        ]
        
        # Pre-compute centroids
        self.global_embeds = np.array(self.embeddings.embed_documents(self.global_examples))
        self.needle_embeds = np.array(self.embeddings.embed_documents(self.needle_examples))
        self.global_centroid = np.mean(self.global_embeds, axis=0)
        self.needle_centroid = np.mean(self.needle_embeds, axis=0)
        
    def classify(self, query: str) -> str:
        query_embed = np.array(self.embeddings.embed_query(query))
        
        # Cosine similarity
        def cosine_sim(v1, v2):
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(v1, v2) / (norm1 * norm2)
        
        sim_to_global = cosine_sim(query_embed, self.global_centroid)
        sim_to_needle = cosine_sim(query_embed, self.needle_centroid)
        
        print(f"[Router Classifier] Scores: Global={sim_to_global:.4f}, Needle={sim_to_needle:.4f}")
        
        if sim_to_global > sim_to_needle + config.GLOBAL_CLASSIFIER_MARGIN:
            return "global"
        return "needle"

class AgentRouter:
    def __init__(self, hybrid_retriever):
        self.global_agent = GlobalAgent()
        self.needle_agent = NeedleAgent(hybrid_retriever)
        self.classifier = None
        
    def _init_classifier(self):
        if self.classifier is None:
            self.classifier = EmbeddingClassifier()
            
    def route(self, query: str, force_mode: str = None) -> Tuple[BaseAgent, str]:
        # Handle force_mode first (from UI toggle)
        if force_mode == "global":
            return self.global_agent, "global (forced via UI)"
        elif force_mode == "needle":
            return self.needle_agent, "needle (forced via UI)"

        # Check keyword override
        lowered = query.lower()
        for kw in config.GLOBAL_KEYWORDS:
            if kw.lower() in lowered:
                return self.global_agent, f"global (keyword match: '{kw}')"

        # Semantic classifier
        try:
            self._init_classifier()
            mode = self.classifier.classify(query)
            if mode == "global":
                return self.global_agent, "global (classifier match)"
            else:
                return self.needle_agent, "needle (classifier match)"
        except Exception as e:
            print(f"Classifier routing failed: {e}. Defaulting to needle.")
            return self.needle_agent, "needle (fallback error)"
