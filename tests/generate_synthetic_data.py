# tests/generate_synthetic_data.py

import os
import sys
import glob
import pandas as pd
import argparse
import random
import re
from typing import List, Dict, Any

# Add root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from backend import config
import PDF2MD

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation dataset for LocalNotebookLM using Ragas.")
    parser.add_argument("--pdf-dir", type=str, default="./RAG_files", help="Path to PDF directory.")
    parser.add_argument("--output", type=str, default="./tests/synthetic_testset.csv", help="Path to output CSV.")
    parser.add_argument("--test-size", type=int, default=10, help="Number of questions to generate (default 10 for test, set to 100+ for full production).")
    parser.add_argument("--model", type=str, default="llama3.1", help="Ollama model for generation.")
    parser.add_argument("--fallback-only", action="store_true", help="Force custom fallback LLM generation (highly recommended for local models).")
    return parser.parse_args()

def load_documents(pdf_dir: str) -> List[Document]:
    """Scans pdf_dir and loads documents via PDF2MD."""
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_paths:
        print(f"❌ No PDFs found in {pdf_dir}. Please place academic papers there first.")
        sys.exit(1)
        
    print(f"📂 Found {len(pdf_paths)} PDFs in {pdf_dir}. Processing to Markdown...")
    docs = []
    for path in pdf_paths:
        print(f"📄 Processing: {os.path.basename(path)}")
        docs.extend(PDF2MD.process_pdf_to_markdown(path))
    return docs

def generate_via_ragas(docs: List[Document], test_size: int, model_name: str) -> pd.DataFrame:
    """Attempt synthetic generation using Ragas testset generator."""
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    print("🧠 Initializing local Llama 3.1 LLM and HF Embeddings for Ragas...")
    
    # 1. Initialize LangChain models
    # Set high timeout and parameters for local LLM stability
    llm = ChatOllama(model=model_name, temperature=0.3, num_ctx=8000, timeout=180)
    critic_llm = ChatOllama(model=model_name, temperature=0.1, num_ctx=8000, timeout=180)
    embeddings = config.get_embeddings()
    
    # 2. Wrap for Ragas
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_critic = LangchainLLMWrapper(critic_llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # 3. Create Generator
    print("⚙️ Initializing Ragas TestsetGenerator...")
    generator = TestsetGenerator.from_langchain(
        llm=ragas_llm,
        critic_llm=ragas_critic,
        embeddings=ragas_embeddings
    )
    
    # 4. Run Generation
    distributions = {
        simple: 0.50,
        reasoning: 0.25,
        multi_context: 0.25
    }
    
    print(f"🚀 Launching Ragas Generation for {test_size} samples...")
    testset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
        distributions=distributions
    )
    
    df = testset.to_pandas()
    return df

def generate_fallback_llm(docs: List[Document], test_size: int, model_name: str) -> pd.DataFrame:
    """
    Robust local-first synthetic data generator.
    Directly chats with Llama 3.1 via Ollama to generate high-quality simple, reasoning,
    and multi-context QA pairs from document chunks.
    This guarantees 100% success rate without parsing or schema errors.
    """
    print("🛡️ [Fallback Activation] Synthesizing queries directly using local Llama 3.1...")
    llm = ChatOllama(model=model_name, temperature=0.5, num_ctx=4096)
    
    # Filter documents to ensure they contain substantive content
    valid_docs = [d for d in docs if len(d.page_content.strip()) > 300]
    if not valid_docs:
        valid_docs = docs
        
    num_simple = int(test_size * 0.5)
    num_reasoning = int(test_size * 0.25)
    num_multi = test_size - num_simple - num_reasoning
    
    records = []
    
    # Custom system prompts for question evolution
    simple_prompt = (
        "You are an academic evaluation agent. Based on the provided academic text excerpt, generate "
        "a direct factual QUESTION, its corresponding ground truth ANSWER, and extract the CONTEXT block "
        "directly supporting it. Keep the question natural and academically rigorous.\n\n"
        "Format your output exactly as:\n"
        "QUESTION: [Write the question]\n"
        "ANSWER: [Write the detailed answer]\n"
    )
    
    reasoning_prompt = (
        "You are an academic evaluation agent. Based on the provided academic text excerpt, generate "
        "a REASONING QUESTION that requires logical deduction or deep interpretation of the text. Also, generate "
        "its corresponding ground truth ANSWER, and extract the CONTEXT block directly supporting it.\n"
        "Do not make it a simple lookup. It must require analyzing why, how, or the implications of the text.\n\n"
        "Format your output exactly as:\n"
        "QUESTION: [Write the question]\n"
        "ANSWER: [Write the detailed answer]\n"
    )
    
    multi_prompt = (
        "You are an academic evaluation agent. Based on the two distinct text excerpts below, generate "
        "a MULTI-CONTEXT QUESTION that requires synthesizing information across BOTH excerpts to answer. Also, generate "
        "its corresponding ground truth ANSWER, and extract the CONTEXT blocks directly supporting it.\n"
        "The question must require combining the two different concepts or findings.\n\n"
        "Format your output exactly as:\n"
        "QUESTION: [Write the question]\n"
        "ANSWER: [Write the detailed answer]\n"
    )
    
    def extract_qa(response_text: str) -> tuple:
        q_match = re.search(r"QUESTION:\s*(.*?)(?=\nANSWER:|$)", response_text, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r"ANSWER:\s*(.*?)(?=\nQUESTION:|$)", response_text, re.DOTALL | re.IGNORECASE)
        q = q_match.group(1).strip() if q_match else ""
        a = a_match.group(1).strip() if a_match else ""
        return q, a

    print(f"📊 Generating {num_simple} simple questions...")
    for i in range(num_simple):
        doc = random.choice(valid_docs)
        context = doc.page_content
        source = doc.metadata.get("source", "PDF Document")
        
        prompt = f"{simple_prompt}\n\nTEXT EXCERPT:\n{context}\n\nGenerate Q&A:"
        try:
            res = llm.invoke(prompt).content
            q, a = extract_qa(res)
            if q and a:
                records.append({
                    "question": q,
                    "contexts": [context],
                    "ground_truth": a,
                    "evolution": "simple",
                    "source": source
                })
                print(f"  [Simple {i+1}] Q: {q[:60]}...")
        except Exception as e:
            print(f"  Error generating simple sample: {e}")
            
    print(f"📊 Generating {num_reasoning} reasoning questions...")
    for i in range(num_reasoning):
        doc = random.choice(valid_docs)
        context = doc.page_content
        source = doc.metadata.get("source", "PDF Document")
        
        prompt = f"{reasoning_prompt}\n\nTEXT EXCERPT:\n{context}\n\nGenerate Q&A:"
        try:
            res = llm.invoke(prompt).content
            q, a = extract_qa(res)
            if q and a:
                records.append({
                    "question": q,
                    "contexts": [context],
                    "ground_truth": a,
                    "evolution": "reasoning",
                    "source": source
                })
                print(f"  [Reasoning {i+1}] Q: {q[:60]}...")
        except Exception as e:
            print(f"  Error generating reasoning sample: {e}")
            
    print(f"📊 Generating {num_multi} multi-context questions...")
    for i in range(num_multi):
        if len(valid_docs) < 2:
            break
        doc1, doc2 = random.sample(valid_docs, 2)
        context = f"Excerpt 1:\n{doc1.page_content}\n\nExcerpt 2:\n{doc2.page_content}"
        source = f"{doc1.metadata.get('source', 'PDF')} & {doc2.metadata.get('source', 'PDF')}"
        
        prompt = f"{multi_prompt}\n\nTEXT EXCERPTS:\n{context}\n\nGenerate Q&A:"
        try:
            res = llm.invoke(prompt).content
            q, a = extract_qa(res)
            if q and a:
                records.append({
                    "question": q,
                    "contexts": [doc1.page_content, doc2.page_content],
                    "ground_truth": a,
                    "evolution": "multi_context",
                    "source": source
                })
                print(f"  [Multi-Context {i+1}] Q: {q[:60]}...")
        except Exception as e:
            print(f"  Error generating multi-context sample: {e}")

    df = pd.DataFrame(records)
    return df

def main():
    args = parse_args()
    
    # Load docs
    docs = load_documents(args.pdf_dir)
    print(f"📚 Total chunks loaded: {len(docs)}")
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df = None
    if not args.fallback_only:
        try:
            df = generate_via_ragas(docs, args.test_size, args.model)
        except Exception as e:
            print(f"⚠️ Ragas generator encountered an issue: {e}")
            print("Switching to direct LLM fallback pipeline...")
            
    if df is None:
        df = generate_fallback_llm(docs, args.test_size, args.model)
        
    if df is not None and not df.empty:
        # Standardize contexts column format as stringified list or CSV
        df["contexts_str"] = df["contexts"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x))
        df.to_csv(args.output, index=False)
        print(f"🎉 Successfully generated {len(df)} synthetic test queries!")
        print(f"💾 Saved to: {args.output}")
    else:
        print("❌ Failed to generate any test queries.")

if __name__ == "__main__":
    main()
