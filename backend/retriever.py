from typing import List
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder
import json
from backend import db

class HybridParentRetriever(BaseRetriever):
    vector_db: Chroma = Field(description="The underlying vector store for child chunks")
    bm25_retriever: BM25Retriever = Field(description="BM25 retriever for parent chunks")
    reranker: CrossEncoder = Field(description="CrossEncoder for reranking", default=None)
    search_k: int = 10
    final_k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        
        # 1. Semantic Search (Dense) - get child chunks from Chroma
        child_docs = self.vector_db.similarity_search(query, k=self.search_k)
        
        # Assemble Parent Docs from Chroma results
        dense_parent_docs = []
        unique_parent_ids = set()
        
        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if pid and pid not in unique_parent_ids:
                unique_parent_ids.add(pid)
                parent_data = db.get_parent_chunk(pid)
                if parent_data:
                    doc = Document(
                        page_content=parent_data["page_content"],
                        metadata=parent_data["metadata"]
                    )
                    dense_parent_docs.append(doc)
                    
        # 2. Keyword Search (Sparse) - get parent chunks from BM25
        # bm25 search directly on parents is fast enough
        sparse_docs = self.bm25_retriever.invoke(query)
        
        # Combine unique documents
        all_docs = {}
        for doc in dense_parent_docs + sparse_docs:
            pid = doc.metadata.get("parent_id")
            if pid not in all_docs:
                all_docs[pid] = doc
                
        candidate_docs = list(all_docs.values())
        
        # 3. Reranking (Cross-Encoder)
        if self.reranker and candidate_docs:
            # Prepare pairs of (query, document)
            pairs = [[query, doc.page_content] for doc in candidate_docs]
            scores = self.reranker.predict(pairs)
            
            # Sort documents by score
            scored_docs = list(zip(candidate_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return Top-K
            return [doc for doc, score in scored_docs[:self.final_k]]
            
        return candidate_docs[:self.final_k]

def init_hybrid_retriever(vector_db, reranker=None):
    print("Initializing BM25 Retriever from SQLite parent chunks...")
    conn = db.sqlite3.connect(db.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT parent_id, page_content, metadata_json FROM parent_chunks")
    rows = cursor.fetchall()
    conn.close()
    
    parent_docs = []
    for row in rows:
        metadata = json.loads(row[2])
        metadata["parent_id"] = row[0]
        parent_docs.append(Document(page_content=row[1], metadata=metadata))
        
    if parent_docs:
        bm25_retriever = BM25Retriever.from_documents(parent_docs)
        bm25_retriever.k = 5 # fetch top 5 from sparse
    else:
        # Dummy retriever if no documents exist, use a non-empty string to avoid ZeroDivisionError
        bm25_retriever = BM25Retriever.from_texts(["empty_database_dummy_document"])
    
    if reranker is None:
        print("Loading local CrossEncoder for Reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    
    retriever = HybridParentRetriever(
        vector_db=vector_db,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        search_k=10,
        final_k=3
    )
    return retriever
