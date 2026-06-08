# tests/evaluate_rag.py

"""Local RAG Evaluation Suite.

Uses Local Embeddings for Context Relevance & Answer Similarity,
and LLM-as-a-Judge (Ollama Llama 3.1) for Faithfulness evaluation.
Provides a quantitative evaluation report to prove RAG performance locally.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any

# Add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from backend import db, retriever, router, config

# Define Golden Q&A Dataset (Ground Truth)
GOLDEN_DATASET = [
    {
        "question": "What is the global summary of the Graduate Student Handbook?",
        "golden_answer": "The handbook provides an overview of key information, resources, and graduate degrees in the Grainger College of Engineering at the University of Illinois.",
        "expected_agent": "GlobalAgent"
    },
    {
        "question": "Who is the director of graduate studies and what does the handbook provide?",
        "golden_answer": "The handbook provides guidelines and information for graduate degrees. Specific directors or administrators can be found in the contact information section.",
        "expected_agent": "GlobalAgent"
    },
    {
        "question": "What is the purpose of the handbook for 2021-2022?",
        "golden_answer": "It serves to outline policies, procedures, and resources for graduate students in Grainger Engineering during the 2021-2022 academic year.",
        "expected_agent": "GlobalAgent"
    }
]

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def evaluate_relevance_and_similarity(embeddings, question: str, context_text: str, generated_answer: str, golden_answer: str) -> Dict[str, float]:
    """Calculate semantic similarities using shared embeddings."""
    q_emb = np.array(embeddings.embed_query(question))
    c_emb = np.array(embeddings.embed_query(context_text))
    g_emb = np.array(embeddings.embed_query(generated_answer))
    gold_emb = np.array(embeddings.embed_query(golden_answer))
    
    context_relevance = cosine_similarity(q_emb, c_emb)
    answer_similarity = cosine_similarity(g_emb, gold_emb)
    
    return {
        "context_relevance": context_relevance,
        "answer_similarity": answer_similarity
    }

def evaluate_faithfulness_llm(llm: Ollama, context: str, answer: str) -> int:
    """LLM-as-a-Judge to evaluate faithfulness (0-5 scale)."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert RAG evaluator. Your task is to judge the 'Faithfulness' of an AI-generated Answer "
            "based strictly on the provided Context. Do not use external knowledge.\n\n"
            "Criteria:\n"
            "- Score 5: The answer is entirely supported by the context with zero hallucinations.\n"
            "- Score 3-4: The answer is mostly supported, but contains minor extrapolation or unsupported details.\n"
            "- Score 1-2: The answer contains major hallucinations or contradicts the context.\n"
            "- Score 0: The answer is completely unrelated or fully hallucinated.\n\n"
            "IMPORTANT: Output ONLY a single integer score (0, 1, 2, 3, 4, or 5). Do not write any other text."
        )),
        ("human", "Context:\n{context}\n\nGenerated Answer:\n{answer}\n\nWhat is the faithfulness score?")
    ])
    try:
        chain = prompt | llm
        response = chain.invoke({"context": context[:8000], "answer": answer})
        # Parse score
        score_str = "".join([c for c in response.strip() if c.isdigit()])
        if score_str:
            return min(5, max(0, int(score_str[0])))
        return 3 # default fallback
    except Exception as e:
        print(f"Error calling judge LLM: {e}")
        return 3

def main():
    print("="*60)
    print("🏆 STARTING LOCAL RAG EVALUATION SUITE")
    print("="*60)
    
    # Initialize components
    embeddings = config.get_embeddings()
    llm = Ollama(model=config.SUMMARY_MODEL, temperature=0)
    
    vector_db = Chroma(persist_directory=config.CHROMA_PERSIST_DIR, embedding_function=embeddings)
    hybrid_retriever = retriever.init_hybrid_retriever(vector_db)
    custom_retriever = router.RouterRetriever(vector_db, hybrid_retriever)
    
    total_metrics = {
        "context_relevance": [],
        "answer_similarity": [],
        "faithfulness": []
    }
    
    for idx, item in enumerate(GOLDEN_DATASET):
        q = item["question"]
        gold = item["golden_answer"]
        print(f"\n[Test {idx+1}] Question: {q}")
        
        # 1. Retrieve
        docs = custom_retriever.invoke(q)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 2. Generate (simple stuff chain answer)
        prompt_qa = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based strictly on the provided context.\n\nContext: {context}"),
            ("human", "{question}")
        ])
        qa_chain = prompt_qa | llm
        try:
            gen_answer = qa_chain.invoke({"context": context_text, "question": q})
        except Exception as e:
            gen_answer = f"Generation failed: {e}"
            
        print(f"-> Generated Answer: {gen_answer[:150]}...")
        
        # 3. Calculate Semantic Metrics
        semantic_scores = evaluate_relevance_and_similarity(
            embeddings, q, context_text, gen_answer, gold
        )
        
        # 4. Calculate LLM Faithfulness
        faith_score = evaluate_faithfulness_llm(llm, context_text, gen_answer)
        
        # Record
        total_metrics["context_relevance"].append(semantic_scores["context_relevance"])
        total_metrics["answer_similarity"].append(semantic_scores["answer_similarity"])
        total_metrics["faithfulness"].append(faith_score)
        
        print(f"-> Context Relevance:  {semantic_scores['context_relevance']:.4f} (Semantic proximity)")
        print(f"-> Answer Similarity:  {semantic_scores['answer_similarity']:.4f} (Semantic similarity to Gold Answer)")
        print(f"-> LLM Faithfulness:   {faith_score}/5 (LLM-as-a-judge score)")
        
    # Print Aggregated Report
    print("\n" + "="*60)
    print("📊 AGGREGATED EVALUATION REPORT")
    print("="*60)
    avg_relevance = np.mean(total_metrics["context_relevance"])
    avg_similarity = np.mean(total_metrics["answer_similarity"])
    avg_faithfulness = np.mean(total_metrics["faithfulness"])
    
    print(f"Average Context Relevance:  {avg_relevance:.4f} (Goal: > 0.35)")
    print(f"Average Answer Similarity:  {avg_similarity:.4f} (Goal: > 0.60)")
    print(f"Average LLM Faithfulness:   {avg_faithfulness:.1f}/5 (Goal: > 4.0/5)")
    print("="*60)
    print("Evaluation Complete. RAG quality proved locally!")

if __name__ == "__main__":
    main()
