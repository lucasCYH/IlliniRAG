# 🎓 IlliniRAG: Privacy-First Local AI Assistant

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.x-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3-black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

## Overview
IlliniRAG is a fully localized, privacy-preserving Retrieval-Augmented Generation (RAG) system built to assist Master of Computer Science students with complex academic handbooks, registration policies, and curriculum guidelines. 

Designed and optimized to run entirely on **Apple Silicon (M4)** without relying on external cloud APIs, this system ensures 100% data privacy and near-zero latency while leveraging state-of-the-art open-source LLMs.

## ✨ Key Features
* **100% Local Processing:** Zero data leaves the machine, ensuring complete privacy for sensitive university documents or personal academic records.
* **Optimized for Apple Silicon:** Utilizes local hardware acceleration for fast embedding generation and LLM inference.
* **Precise Source Citation:** The RAG pipeline mitigates LLM hallucinations by retrieving and strictly displaying exact source documents and page numbers alongside its answers.
* **Interactive UI:** A clean, conversational web interface built with Streamlit.

## 🛠️ Tech Stack
* **LLM Engine:** [Ollama](https://ollama.com/) running `llama3` (8B)
* **Framework:** [LangChain](https://python.langchain.com/) (using LCEL - LangChain Expression Language)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
* **Frontend:** Streamlit

## 🚀 Quick Start

### 1. Prerequisites
* **macOS** (Optimized for M-series chips)
* **Python 3.13**
* **Ollama** installed and running on your machine.

### 2. Download the Local LLM
Ensure Ollama is running, then pull the Llama 3 model:
```bash
ollama run llama3
