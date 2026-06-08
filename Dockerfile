# 使用官方 Python 3.9 輕量版作為基礎映像檔
FROM python:3.9-slim

# 設定容器工作目錄
WORKDIR /app

# 安裝基本系統編譯與執行套件 (如 pymupdf4llm / sentence-transformers 編譯可能需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案其餘原始碼到工作目錄
COPY . .

# 建立預設的本地資料儲存目錄 (若部署時無掛載 Volume，則資料仍會暫時保存在容器內)
RUN mkdir -p /app/data

# 暴露 Streamlit 預設通訊埠
EXPOSE 8501

# Streamlit 執行參數與健康檢查設定
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 設定容器啟動指令，關閉 CORS 與啟用全域綁定
ENTRYPOINT ["streamlit", "run", "UI.py", "--server.port=8501", "--server.address=0.0.0.0"]
