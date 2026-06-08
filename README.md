# 🎓 IlliniRAG: Privacy-First Local AI Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3-black)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

IlliniRAG 是一個完全本地化、保護隱私的檢索增強生成 (RAG) 助理系統。它專為協助學生與研究人員閱讀複雜的學術論文、課程註冊政策和學術手冊而設計，提供高層級的大綱摘要與精細細節檢索。

本系統針對 **Apple Silicon** 進行了本地最佳化，所有運算均在使用者本機執行，確保 100% 的資料隱私與極低延遲。

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

---

## 🐳 部署與隱私策略

本專案支援**本地 Docker 離線部署**與 **Fly.io 雲端掛載加密磁碟卷**的兩種方案，所有資料庫與向量庫路徑均已環境變數參數化，保證資料隱私安全。

詳細部署步驟請參閱：**[deployment/README.md](file:///Users/ut/Desktop/IlliniRAG/deployment/README.md)**。
