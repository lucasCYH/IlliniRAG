# 🎓 IlliniRAG: Privacy-First Local AI Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3-black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

IlliniRAG 是一個完全本地化、保護隱私的檢索增強生成 (RAG) 助理系統。它專為協助學生與研究人員閱讀複雜的學術論文、課程註冊政策和學術手冊而設計，提供高層級的大綱摘要與精細細節檢索。

本系統針對 **Apple Silicon** 進行了本地最佳化，所有運算均在使用者本機執行，確保 100% 的資料隱私與極低延遲。

---

## 🌟 工業級 RAG 優化亮點 (R&D‑Grade Highlights)

本專案拒絕簡單的「Toy RAG」玩具套路，在架構上完全對齊**工業界生產級 RAG** 的核心優化指標：

1. **🚀 雙路混合檢索 (Hybrid Search: Dense + Sparse)**
   - 整合了 **ChromaDB 語意向量檢索 (Dense)** 與 **BM25 關鍵字檢索 (Sparse)**。
   - 同時保障了對「語意意圖」與「特定術語/代號（如課程代號、法規條款）」的召回率 (Recall)。
2. **🎯 Cross‑Encoder 深度重排 (Reranking)**
   - 檢索出的候選文檔會通過本地的 **Cross‑Encoder 重排模型** (`ms-marco-MiniLM-L-6-v2`) 進行二次排序，只將關聯度最高的 Top-K 文本送入 LLM。
   - 這是大幅降低 LLM 幻覺、提升回答精確度與 Context Precision 的工業界標準黃金架構。
3. **🧠 語意代理路由器 (Semantic Agent Routing)**
   - 擺脫了傳統簡單的關鍵字路由，採用 SentenceTransformers 語意向量分類器，能自動識別問題是「局部細節 (Needle)」還是「全域大綱 (Global)」，最大化減輕大語言模型在長文本上的檢索天花板。
4. **📊 生產級前端 Web 控制台**
   - 摒棄 CLI 腳本，使用 Streamlit 開發完整的前端 Dashboard，具備 Ingest 百分比進度條、摺疊大綱樹狀瀏覽、分頁文檔閱讀與 Notes 就地 CRUD 編輯，實測體驗極佳。

---

## ✨ 進階核心功能

* **📖 階層式文件大綱索引 (Hierarchical Summary Index)**
  - 根據 PDF 的標題層級自動進行語意分組，為每個 Chapter 與 Section 生成獨立大綱摘要。
  - 提供 **📊 Summary Viewer** 樹狀摺疊元件，使用者可逐層展開閱讀大綱，極適合快速檢視論文方向。
* **🧠 語意代理路由器 (Embedding Agent Router)**
  - 使用 SentenceTransformers 語意分類器，自動將問題分流至 **GlobalAgent**（全域大綱檢索）或 **NeedleAgent**（混合細節檢索），並在回答底部標記 Sources 來源與 Routing 路由決策日誌。
* **🧪 通用學術標題預處理器 (Universal Academic Header Promoter)**
  - 自適應學術論文的雙欄排版與多樣標題格式（如 YOLOv4 的 `**1. Introduction**` 與 OmniVoice 的 `**1** **Introduction**`），並能自動過濾圖表與廣播劇對白等雜訊，確保任何論文皆能切分出精確大綱。
* **📝 Notes 就地編輯管理 (CRUD)**
  - 支援筆記的建立、就地 (In-place) 編輯修改與刪除，大幅提升筆記記錄體驗。
* **🔍 分頁 Document Viewer & Studio 篩選**
  - **Document Viewer**: 支援依上傳檔案進行切分，以選單形式切換個別文件的完整 Markdown Chunks。
  - **Studio**: 新增多選框，可自由勾選特定的 PDF 文件以生成客製化的 Study Guide 或 Audio Podcast。
* **🔒 共享 Embeddings 單例模式**
  - 實作單例 (Singleton) 延遲載入共享 embeddings 實體，徹底解決 PyTorch 裝置重複載入 meta tensor 衝突，顯著節省本機記憶體。

---

## 🛠️ 技術棧

* **LLM Engine:** [Ollama](https://ollama.com/) 運行 `llama3.1` (或可選其他本地模型)
* **Framework:** [LangChain](https://python.langchain.com/) (LCEL)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` / `paraphrase-MiniLM-L6-v2`
* **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Frontend:** Streamlit

---

## 🚀 快速啟動

### 1. 本地環境準備
* 安裝 **Ollama** 並下載模型：
  ```bash
  ollama run llama3.1
  ```
* 建立並啟動虛擬環境：
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* 安裝依賴套件：
  ```bash
  pip install -r requirements.txt
  ```

### 2. 啟動 Streamlit UI
執行以下指令啟動 RAG 網頁服務：
```bash
python -m streamlit run UI.py
```
打開瀏覽器訪問 `http://localhost:8501`。

### 3. 執行單元測試
執行單元測試驗證代理人與大綱 fallback 邏輯：
```bash
python -m unittest tests/test_agents.py
```

### 4. 執行本地 RAG 品質評估 (Evaluation)
透過本地 LLM-as-a-judge 與語意相似度演算法，對 RAG 的檢索精準度（Context Relevance）與回答忠實度（Faithfulness）進行量化評估：
```bash
PYTHONPATH=. python tests/evaluate_rag.py
```

#### 📖 學術引用與評估依據 (Academic References)
本評估套件之指標設計與目標分數 (Goal) 均嚴格遵循以下主流學術研究之理論基礎：
- **RAGAS 評估框架 (Context Relevance / Faithfulness)**: 
  *Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023).* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- **LLM-as-a-Judge 裁判機制 (LLM Evaluation)**: 
  *Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023).* [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- **Sentence-BERT 語意嵌入餘弦相似度基準**: 
  *Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019).* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

---

## 🐳 部署與隱私策略

本專案支援**本地 Docker 離線部署**與 **Fly.io 雲端掛載加密磁碟卷**的兩種方案，所有資料庫與向量庫路徑均已環境變數參數化，保證資料隱私安全。

詳細部署步驟請參閱：**[deployment/README.md](file:///Users/ut/Desktop/IlliniRAG/deployment/README.md)**。

---

## 👥 使用體驗一致性保證 (Ensuring Consistent Experience)

為了確保不同環境（作業系統、硬體配置、Python 版本）下的使用者都能獲得完全一致的優質體驗，本專案在架構上進行了以下設計：

1. **📦 隔離的虛擬環境 (Python Virtualenv)**
   - 透過 `python3 -m venv .venv` 將專案依賴與本機系統環境完全隔離，避免因系統全域套件衝突而報錯。
2. **🐳 容器化標準環境 (Dockerization)**
   - 本專案完整配備了 `Dockerfile`。使用 Docker 啟動能完全屏蔽 Windows、Mac 與 Linux 之間的文件讀取、路徑處理與 Python 編譯環境差異，實現「一鍵啟動，處處相同」的體驗。
3. **🔄 語意模型自動下載與本地快取**
   - 專案中所使用的向量嵌入模型 (`all-MiniLM-L6-v2`) 與語意路由器模型 (`paraphrase-MiniLM-L6-v2`) 會在系統**首次執行時自動從 Hugging Face 下載並快取至本地**。使用者無需進行複雜的手動模型下載與設定，且 SentenceTransformers 會自動根據硬體（如 M-series MPS、CUDA GPU、或普通 CPU）選擇最佳運算加速，確保流暢度一致。
4. **🧠 單例共享記憶體 (Singleton Memory Sharing)**
   - 專案在 `backend/config.py` 中實現了延遲載入單例，確保在整個 Streamlit 生命週期中只會實例化一個 Embeddings 實體。這解決了在特定平台上多線程重複載入權重時會發生的 PyTorch meta tensor 衝突錯誤，保證了程式的絕對穩定。

---

## 🗺️ 未來研發路線圖與避坑指南 (Roadmap)

本專案定義了四個階段的研發路線圖，包含 SQLite 歷史記憶持久化、個人筆記語意增量同步、混合部署架構、以及多模態圖表與 Setfit 意圖分類器微調的避坑指南。

詳細規劃請參閱：**[roadmap.md](file:///Users/ut/.gemini/antigravity/brain/e4d4dd50-6a44-40f7-a18f-77fe15f736b7/roadmap.md)**。


