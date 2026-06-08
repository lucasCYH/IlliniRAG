# IlliniRAG 部署指南 & 資料隱私策略

本文件說明如何將 IlliniRAG 部署至遠端伺服器 (如 Fly.io) 或進行本地 Docker 部署，同時保證資料庫與向量庫的隱私安全。

---

## 🔒 資料隱私策略 (Data Privacy Strategy)

為了保證資料完全保密，且保持在使用者端或受控制的環境，我們設計了以下架構：
1. **路徑參數化 (Parametrized Paths)**: 所有資料庫 (`notebook_store.db`) 與向量庫 (`chroma_db`) 的路徑皆使用環境變數讀取。
2. **持久化磁碟卷 (Volume Mounts)**: 
   - 遠端部署時，透過掛載**持久化卷卷 (Persistent Volumes)**，確保檔案不會寫入容器的可寫層，且只有該伺服器擁有存取權。
   - 本地 Docker 部署時，資料直接掛載在宿主機 (Host Machine) 的資料夾中，資料 100% 不會流向任何第三方雲端。
3. **無網上傳**: 所有的 PDF 檔案解析、向量切分、以及 embeddings 生成 (`all-MiniLM-L6-v2`) 均為 100% 本地運算。
4. **外部 LLM 設定**: 如果要遠端完全免費且私密，可在遠端安裝 Ollama 並將 `SUMMARY_MODEL` 指向本地的 Ollama 服務。

---

## 🐳 方案 A：本地 Docker 部署 (推薦，100% 私密 & 免費)

這是最安全的部署方式，直接在本機以隔離的 Docker 容器運行，並將資料保存在本機目錄。

### 1. 構建映像檔
在專案根目錄下運行：
```bash
docker build -t illinirag:latest .
```

### 2. 啟動容器 (掛載本地目錄)
將本機目錄 `~/illinirag_data` 掛載到容器的 `/app/data` 目錄中，並以環境變數指定路徑：
```bash
mkdir -p ~/illinirag_data
docker run -d \
  -p 8501:8501 \
  -v ~/illinirag_data:/app/data \
  -e DB_PATH=/app/data/notebook_store.db \
  -e CHROMA_PERSIST_DIR=/app/data/chroma_db \
  --name illinirag-app \
  illinirag:latest
```
打開瀏覽器訪問 `http://localhost:8501` 即可使用。

---

## 🚀 方案 B：遠端 Fly.io 部署 (免費方案)

Fly.io 提供 3 個免費的 256MB VM (或者可合併為 1 個 512MB VM) 以及 3GB 的免費卷儲存空間。

### 1. 安裝 Flyctl CLI 並登入
請參考 [Fly.io 官網](https://fly.io/docs/hands-on/install-flyctl/) 安裝，並在終端機登入：
```bash
fly auth login
```

### 2. 初始化專案 (建立 fly.toml)
在專案根目錄運行：
```bash
fly launch --no-deploy
```
這將建立 `fly.toml` 組態檔。在過程中，請為應用程式命名 (例如 `illinirag-assistant`)。

### 3. 建立並掛載免費的 Volume (3GB 內免費)
在您的應用程式所在的地區建立一個名為 `illinirag_data` 的 Volume (以 `hkg` 香港或 `nrt` 東京為例)：
```bash
fly volumes create illinirag_data --size 1 --region hkg
```

### 4. 設定 `fly.toml` 檔案
編輯 `fly.toml`，在 `[mounts]` 區段與 `[env]` 區段加上掛載與環境變數設定：

```toml
[env]
  DB_PATH = "/app/data/notebook_store.db"
  CHROMA_PERSIST_DIR = "/app/data/chroma_db"
  PORT = "8501"

[mounts]
  source = "illinirag_data"
  destination = "/app/data"
```

同時，將 Streamlit 的暴露埠口設為 `8501`：
```toml
[[services]]
  internal_port = 8501
  protocol = "tcp"
  # ... (其餘維持預設)
```

### 5. 部署
在根目錄下運行部署指令：
```bash
fly deploy
```
部署完成後，您的 RAG 助理即可透過 `https://<your-app-name>.fly.dev` 開啟，而所有使用者上傳的 PDF 檔案與產生的摘要，都會加密存放在您的 Fly Volume `/app/data` 中，絕不外流。
