import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter


def process_pdf_to_markdown(file_path):

    # 1. 讀取 Grainger Handbook PDF 並直接轉換為 Markdown 字串
    # 這個套件會自動辨識字體大小與粗細，將其轉換為 #, ##, ### 等 Markdown 標題
    md_text = pymupdf4llm.to_markdown(file_path)

    # 2. 定義要作為切分依據的 Markdown 標題層級
    # 對應手冊中的大章節 (如 GRADUATE STUDENT FUNDING) 與子章節 (如 Fellowships)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 3. 初始化 Markdown 標題切分器
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # 保留原本的標題文字，幫助 LLM 理解脈絡
    )

    # 4. 根據標題結構將整份文件切分為多個 Document 物件
    md_header_splits = markdown_splitter.split_text(md_text)

    # 觀察切分結果的 Metadata
    for i, split in enumerate(md_header_splits[:3]):
        print(f"--- Chunk {i+1} ---")
        print(f"Metadata (所屬章節): {split.metadata}")
        print(f"內文前 100 字:\n{split.page_content[:100]}...\n")

    return md_header_splits