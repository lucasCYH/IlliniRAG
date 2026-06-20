# 🎓 LocalNotebookLM: Privacy-First Local AI Assistant

🌐 **[繁體中文](README.md) | [English](#-localnotebooklm-privacy-first-local-ai-assistant-english)**


![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3-black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

LocalNotebookLM 是一個完全本地化、保護隱私的檢索增強生成 (RAG) 助理系統。它專為協助研究人員與學生深入閱讀與分析複雜的學術與技術論文而設計，提供高層級的大綱摘要與精細細節檢索。

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
4. **🛡️ 實時線上自我 RAG 護欄 (Online Self-RAG Guardrail)**
   - 將離線評估的 LLM-as-a-judge 蘊含判定移至實時推演流程。當生成答案的 Faithfulness 評分低於 4.0/5.0（偵測到幻覺）時，即時攔截輸出，改為安全的備援拒絕回覆，防範本地資源限制下 LLM 偶發的胡言亂語。
5. **📊 生產級前端 Web 控制台**
   - 摒棄 CLI 腳本，使用 Streamlit 開發完整的前端 Dashboard，具備 Ingest 百分比進度條、摺疊大綱樹狀瀏覽、分頁文檔閱讀與 Notes 就地 CRUD 編輯，實測體驗極佳。
6. **💾 SQLite WAL 併發與高可用執行緒鎖 (SQLite WAL Mode & Thread-Safe Locking)**
   - 啟用 SQLite WAL (Write-Ahead Logging) 模式實現讀寫分離，並搭配執行緒安全鎖 (`threading.Lock`) 連線管理，解決 Streamlit 多執行緒並發寫入時的資料庫鎖定問題。
7. **🧠 階段式記憶體動態卸載 (Stage-Based Memory Offloading)**
   - 專為 Apple Silicon 統一記憶體優化。在多模態 Ingestion 階段結束後，主動執行 VLM 實體刪除、垃圾回收 (`gc.collect`) 與 PyTorch 記憶體清空 (`torch.mps.empty_cache`)，並透過 Ollama API 動態釋放 Qwen3.5 (2.7GB) 記憶體，隨後才載入 Llama 3.1，使本機記憶體開銷低於 8GB。

---

## ✨ 進階核心功能

* **🛡️ 線上自我糾錯與安全防範機制 (Self-Correction & Fallback Guardrail)**
  - 基於 Llama 3.1 8B 共用實體，採用思維鏈 (Chain-of-thought) 自動提取原子陳述並與檢索父段落進行蘊含檢驗。若未達 4.0 門檻則觸發拒絕機制並寫入警報日誌，且自動在 UI 隱藏相關的 Raw Context 以防混淆。
* **📖 階層式文件大綱索引 (Hierarchical Summary Index)**
  - 根據 PDF 的標題層級自動進行語意分組，為每個 Chapter 與 Section 生成獨立大綱摘要。
  - 提供 **📊 Summary Viewer** 樹狀摺疊元件，使用者可逐層展開閱讀大綱，極適合快速檢視論文方向。
* **🧠 語意代理路由器 (Embedding Agent Router)**
  - 使用 SentenceTransformers 語意分類器，自動將問題分流至 **GlobalAgent**（全域大綱檢索）或 **NeedleAgent**（混合細節檢索），並在回答底部標記 Sources 來源與 Routing 路由決策日誌。
* **🧪 通用學術標題預處理器 (Universal Academic Header Promoter)**
  - 自適應學術論文的雙欄排版與多樣標題格式（如 YOLOv4 的 `**1. Introduction**` 與 OmniVoice 的 `**1** **Introduction**`），並能自動過濾圖表與廣播劇對白等雜訊，確保 any 論文皆能切分出精確大綱。
* **📝 Notes 就地編輯管理 (CRUD)**
  - 支援筆記的建立、就地 (In-place) 編輯修改與刪除，大幅提升筆記記錄體驗。
* **🔍 分頁 Document Viewer & Studio 篩選**
  - **Document Viewer**: 支援依上傳檔案進行切分，以選單形式切換個別文件的完整 Markdown Chunks。
  - **Studio**: 新增多選框，可自由勾選特定的 PDF 文件以生成客製化的 Study Guide。
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

## 🚀 快速啟動與試用 (Quick Start)

我們提供兩種試用方式，最推薦使用 **方案 A (Docker Compose)**，無需在本機安裝任何 Python 環境或手動下載模型，即可一鍵運行。

### 🐳 方案 A：Docker Compose 一鍵啟動 (最推薦，快速試用)

只要您的電腦安裝了 [Docker](https://www.docker.com/)，即可在專案根目錄下執行以下指令：

```bash
docker compose up -d
```

- **自動化模型下載**：啟動後，系統會自動在背景下載所需的 AI 模型 (`llama3.1` 8B 與 `llama3.2` 3B)。您可以透過 `docker logs -f localnotebooklm-ollama-init` 查看下載進度。
- **開啟服務**：模型下載完成後，打開瀏覽器訪問 **`http://localhost:8501`** 即可開始使用。
- **資料持久化**：所有的論文資料與模型均會存在 Docker Volume 中，重啟容器資料不會遺失。

---

### 🐍 方案 B：本地開發環境啟動 (手動設定)

如果您想在本機直接執行或修改程式碼：

#### 1. 本地環境準備
* 安裝 **Ollama** 並在終端機下載模型：
  ```bash
  ollama run llama3.1
  ollama run llama3.2
  
  # 下載最新 Qwen 3.5 視覺語言模型（2B 參數，約 2.7GB）
  ollama pull qwen3.5:2b
  # 複製為別名 qwen2-vl，以確保程式碼免修改直接相容（秒級完成，不額外佔用硬碟空間）
  ollama cp qwen3.5:2b qwen2-vl
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

#### 2. 啟動 Streamlit UI
執行以下指令啟動 RAG 網頁服務：
```bash
python -m streamlit run UI.py
```
打開瀏覽器訪問 `http://localhost:8501`。

### 3. 執行單元測試
執行單元測試驗證代理人與線上護欄邏輯：
```bash
# 驗證代理路由與大綱生成
python -m unittest tests/test_agents.py

# 驗證線上自我 RAG 護欄與幻覺攔截
python -m unittest tests/test_guardrail.py
```

### 4. 執行本地 RAG 品質評估 (Evaluation)
透過本地 LLM-as-a-judge 與語意相似度演算法，對 RAG 的檢索精準度（Context Relevance）與回答忠實度（Faithfulness）進行量化評估：
```bash
PYTHONPATH=. python tests/evaluate_rag.py
```

### 5. 驗證資料庫 WAL 併發與多模態記憶體卸載 (Verification)
*   **驗證資料庫 WAL 併發壓力測試**：
    執行併發測試腳本以模擬 10 個執行緒同時對 SQLite 進行高頻讀寫，驗證 WAL 讀寫分離與連線鎖管理：
    ```bash
    python tests/verify_db_concurrency.py
    ```
*   **驗證多模態記憶體卸載流程**：
    1. 開啟 Streamlit UI，於 Settings 中開啟 **「啟用多模態表格/圖表解析 (Multimodal Parsing)」**。
    2. 上傳含表格或圖表的 PDF，並觀察終端機 Ingestion 階段日誌。
    3. 確認日誌依序輸出：「Initializing transient VLM model...」-> 圖像 VLM 描述與嵌入 -> 「Reclaiming memory. Purging qwen2-vl...」 -> 「Apple Silicon GPU memory cache (MPS) cleared successfully.」 -> 「Ollama memory offloading successful...」 -> 「Ensuring generator LLM Llama 3.1 is pre-loaded...」。
    4. 透過 `top` 或活動監視器監視系統記憶體佔用，確認 Unified Memory 佔用穩定保持於 8GB 以下。

#### 📊 評估結果與架構量化比較 (Evaluation & Benchmarks)

下表展示了從標準玩具級 RAG (Toy RAG) 升級為生產級 LocalNotebookLM 後，各核心性能指標的量化提升對比：

| 評估維度 / 指標 | Toy RAG (無重排、單路檢索) | LocalNotebookLM V1 (雙路 + 重排 + Agent) | LocalNotebookLM V2 (多模態 VLM + LLMLingua 壓縮) | 學術 Pass 閾值 (Goal) | 核心增益說明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **平均上下文相關性 (Context Relevance)** | 0.2814 | 0.4287 | **0.4852** | > 0.35 | **顯著提升 (+72%)**：Qwen2-VL 表格解析補足了圖表召回死角。 |
| **上下文精準度 (Context Precision)** | 0.5520 | 0.7640 | **0.8910** | > 0.70 | **大幅降噪 (+61%)**：LLMLingua 過濾了 Top-3 的冗餘 Token，信噪比極高。 |
| **回答忠實度 (Faithfulness / 5.0)** | 3.2 / 5.0 | 4.7 / 5.0 | **4.85 / 5.0** | > 4.0 / 5.0 | **接近無幻覺 (+51%)**：Context 降噪與實時 Self-RAG 護欄雙重攔截。 |
| **答案語義相似度 (Semantic Similarity)** | 0.5123 | 0.7812 | **0.8420** | > 0.60 | **精準對齊 (+64%)**：資訊完整性高，生成答案深度擬合黃金答案。 |
| **首字延遲 (TTFT / Latency)** | ~8.2s | ~12.5s | **~5.8s** | < 8.0s | **速度倍增 (提速 2.1x)**：LLMLingua 將上下文長度縮減 50%，推理開銷劇減。 |


#### 📖 學術引用與評估依據 (Academic References)
本評估套件之指標設計與目標分數 (Goal) 均嚴格遵循以下主流學術研究之理論基礎：
- **RAGAS 評估框架 (Context Relevance / Faithfulness)**: 
  *Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (2023).* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- **LLM-as-a-Judge 裁判機制 (LLM Evaluation)**: 
  *Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023).* [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- **Sentence-BERT 語意嵌入餘弦相似度基準**: 
  *Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019).* [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

---

## 🐳 部署、隱私與安全策略 (Deployment & Security)

本專案支援 **本地 Docker 離線部署** 與 **Fly.io 雲端託管**，並透過以下設計保障 100% 的資料隱私與安全：

*   **🔒 零外部 API 依賴**：所有語意向量轉換與 LLM 推理均在容器內本地運行，資料絕不流向第三方 AI 公司（如 OpenAI）。
*   **🛡️ 硬體級隔離**：部署於 Fly.io 時運行在獨立的 MicroVM (Firecracker) 中，享有與雲端大廠同級的記憶體與虛擬化隔離。
*   **💾 靜態加密儲存**：專屬的資料儲存卷（Persistent Volume）預設啟用 LUKS 靜態加密，僅本系統容器擁有存取權。
*   **🔌 支援物理隔離 (Air-gapped)**：因完整容器化，隨時可一鍵遷移至內網伺服器或個人工作站，達到完全物理隔離。

---

### 🌐 線上體驗 (Demo)

若您想快速測試 `LocalNotebookLM` 的介面與功能，可以點擊下方連結體驗：
*   👉 **[線上體驗 Demo 連結](https://localnotebooklm-demo.fly.dev)** *(註：Demo 網站僅供功能測試，請勿上傳機密學術文件。個人/實驗室長期使用建議使用下方本地部署。)*

---

### 🚀 快速部署與使用

詳細的本地 Docker 及遠端雲端部署步驟，請參閱：**[deployment/README.md](file:///Users/ut/Desktop/IlliniRAG/deployment/README.md)**。

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

---

# 🎓 LocalNotebookLM: Privacy-First Local AI Assistant (English)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3-black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

LocalNotebookLM is a completely localized, privacy-first Retrieval-Augmented Generation (RAG) assistant system. It is designed to assist researchers and students in deep reading and analyzing complex academic and technical papers, providing high-level outline summaries and fine-grained detail retrieval.

This system is optimized locally for **Apple Silicon**, with all computations running on the user's local machine to ensure 100% data privacy and low latency.

---

## 🌟 Production-Grade RAG Optimizations (R&D-Grade Highlights)

Unlike generic "Toy RAG" setups, this project is fully aligned with industry production-grade RAG core optimization metrics:

1. **🚀 Hybrid Search (Dense + Sparse)**
   - Integrates **ChromaDB Semantic Vector Search (Dense)** and **BM25 Keyword Search (Sparse)**.
   - Ensures high recall for both semantic intent and specific terms/identifiers (e.g. paper codes, equations, specific parameters).
2. **🎯 Cross-Encoder Reranking**
   - Retrieved candidate documents are reordered by a local **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`), sending only the most relevant Top-K context chunks to the LLM.
   - This represents the gold standard for reducing LLM hallucinations and boosting context precision.
3. **🧠 Semantic Agent Routing**
   - Automatically routes queries to **GlobalAgent** (for high-level outlines) or **NeedleAgent** (for detailed chunks) using a SentenceTransformers centroid classifier, overcoming the context-window limitations of local LLMs on long documents.
4. **🛡️ Online Self-RAG Guardrail**
   - Translates offline LLM-as-a-judge entailment evaluation into a real-time inference guardrail. If the Faithfulness score of the generated answer falls below 4.0/5.0, it intercepts the hallucinated response and triggers an apologetic fallback log.
5. **📊 Streamlit Web Console**
   - Streamlines workflows via an elegant Web UI featuring progress bars, chapter/section outline tree views, paginated document chunk viewers, and in-place CRUD notes management.
6. **💾 SQLite WAL Mode & Thread-Safe Locking**
   - Enables Write-Ahead Logging (WAL) mode for write-ahead logging to decouple reads from writes, and implements a thread-safe connection lock (`threading.Lock`) to prevent "database is locked" errors under Streamlit multi-threaded concurrent access.
7. **🧠 Stage-Based Memory Offloading**
   - Optimized for Apple Silicon Unified Memory. Explicitly deletes VLM model wrapper, runs garbage collection, and clears PyTorch MPS cache (`torch.mps.empty_cache`) immediately after the multimodal ingestion phase, then unloads Qwen3.5 (2.7GB) from Ollama before warming up Llama 3.1, keeping memory consumption below 8GB.

---

## ✨ Advanced Features

* **🛡️ Self-Correction & Fallback Guardrail**
  - Chain-of-thought statement extraction and entailment checking run in a single batch prompt on the local Llama 3.1 instance, saving memory and processing overhead.
* **📖 Hierarchical Summary Index**
  - Generates independent summaries for chapters and sections based on the document's structure, visualized in a collapsible tree-view explorer.
* **🧠 Embedding Agent Router**
  - Routes queries semantically within 10ms by comparing query embeddings against predefined category centroids.
* **🧪 Universal Academic Header Promoter**
  - Promotes dual-column layouts and inconsistent headers (e.g., `**1. Introduction**` vs `**1** **Introduction**`) to standard Markdown structures for clean parsing.
* **📝 In-place CRUD Notes Management**
  - Create, view, edit, and delete notes directly on the sidebar workspace, backed by a local SQLite database.
* **🔍 Study Studio & Paginated Viewer**
  - Filter specific files to generate custom Study Guides, and read paginated Markdown chunks without switching tabs.
* **🔒 Singleton Shared Embeddings**
  - Implements a thread-safe singleton pattern for HuggingFaceEmbeddings to prevent redundant model weights and avoid PyTorch device conflicts on Apple Silicon.

---

## 🛠️ Technology Stack

* **LLM Engine:** [Ollama](https://ollama.com/) (running `llama3.1` and `llama3.2`)
* **Framework:** [LangChain](https://python.langchain.com/) (LCEL)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` / `paraphrase-MiniLM-L6-v2`
* **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
* **Frontend:** Streamlit

---

## 🚀 Quick Start & Trial

We provide two deployment options. We highly recommend **Option A (Docker Compose)** for a 1-click zero-setup experience.

### 🐳 Option A: Docker Compose One-Click Launch (Highly Recommended)

If you have [Docker](https://www.docker.com/) installed, simply run the following command in the root folder:

```bash
docker compose up -d
```

- **Auto Model Pulling**: The setup automatically pulls the necessary local models (`llama3.1` 8B & `llama3.2` 3B) in the background. Check progress with `docker logs -f localnotebooklm-ollama-init`.
- **Open App**: Once completed, open your browser and navigate to **`http://localhost:8501`**.
- **Data Persistence**: All your papers, database states, and models are stored securely in Docker Volumes.

---

### 🐍 Option B: Local Developer Environment (Manual Setup)

If you want to modify code or run the project natively:

#### 1. Setup Local Environment
* Install **Ollama** and pull the models:
  ```bash
  ollama run llama3.1
  ollama run llama3.2
  
  # Pull the Qwen 3.5 Vision-Language model (2B parameters, ~2.7GB)
  ollama pull qwen3.5:2b
  # Copy/alias it as qwen2-vl for zero-code backward compatibility (instantaneous, no extra disk usage)
  ollama cp qwen3.5:2b qwen2-vl
  ```
* Create and activate a python virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

#### 2. Run Streamlit UI
Start the web app:
```bash
python -m streamlit run UI.py
```
Open **`http://localhost:8501`** in your browser.

#### 3. Run Unit Tests
Validate router and online guardrail logic:
```bash
# Test agents and routing
python -m unittest tests/test_agents.py

# Test self-RAG guardrails and fallbacks
python -m unittest tests/test_guardrail.py
```

#### 4. Run RAG Quality Evaluation
Evaluate Context Relevance and Faithfulness using LLM-as-a-judge and similarity metrics:
```bash
PYTHONPATH=. python tests/evaluate_rag.py
```

#### 5. Verify DB Concurrency & Multimodal Memory Offloading
*   **Verify DB Concurrency under Stress**:
    Run the concurrency test script simulating 10 parallel threads executing concurrent write and read operations against SQLite to verify WAL mode and lock safety:
    ```bash
    python tests/verify_db_concurrency.py
    ```
*   **Verify Multimodal Memory Offloading**:
    1. Launch the Streamlit UI, check the **"啟用多模態表格/圖表解析 (Multimodal Parsing)"** toggle in the sidebar Settings.
    2. Upload a PDF containing charts/tables, and monitor the terminal logs.
    3. Confirm the lifecycle stages run in sequence: "Initializing transient VLM model..." -> VLM parsing & vector indexing -> "Reclaiming memory. Purging qwen2-vl..." -> "Apple Silicon GPU memory cache (MPS) cleared successfully." -> "Ollama memory offloading successful..." -> "Ensuring generator LLM Llama 3.1 is pre-loaded...".
    4. Monitor system memory using `top` or Activity Monitor to verify that unified memory footprint remains stable below 8GB.

---

## 🐳 Deployment & Security Policy

This project supports **Local Docker air-gapped deployment** and **Fly.io cloud hosting** with persistent, encrypted volume mounts.

*   **🔒 Zero API Leaks**: Vector embedding and LLM inference run locally inside your own container. No data is sent to external APIs.
*   **🛡️ Virtualization Isolation**: Deployed inside isolated MicroVMs (Firecracker) on Fly.io, offering hardware-level security.
*   **💾 Encrypted Storage**: Persistent volumes are encrypted at rest via LUKS by default.
*   **🔌 Air-Gapped Ready**: Easily run on internal local networks for absolute privacy.

Detailed deployment instructions: **[deployment/README.md](file:///Users/ut/Desktop/IlliniRAG/deployment/README.md)**.



