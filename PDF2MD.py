import re
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter

def preprocess_markdown_headers(md_text: str) -> str:
    """Preprocess raw markdown text to identify academic paper headers formatted as bold or plain text lines, ensuring broad compatibility."""
    lines = md_text.split("\n")
    processed_lines = []
    
    # 1. Broad academic paper section keywords (case-insensitive, max 80 chars)
    academic_keywords = {
        "abstract", "introduction", "related work", "background",
        "methodology", "proposed method", "proposed architecture",
        "experiments", "experimental setup", "experimental results",
        "results", "discussion", "evaluation", "implementation",
        "conclusion", "conclusions", "references", "bibliography",
        "appendix", "acknowledgements", "ethics statements", "limitations"
    }
    
    # 2. Regex patterns
    # YOLOv4 format: **1. Introduction**
    pattern_bold_num_1 = re.compile(r'^\*\*(\d+(\.\d+)*\.?\s+[^:\n]+)\*\*\s*$')
    # OmniVoice format: **1** **Introduction** or **A** **Appendix**
    pattern_bold_num_2 = re.compile(r'^\*\*(\d+(\.\d+)*|[A-Z])\*\*\s+\*\*([^:\n]+)\*\*\s*$')
    # Plain number format: 1. Introduction or 2.1. System Overview
    pattern_plain_num = re.compile(r'^(\d+(\.\d+)*|[A-Z])\.\s+([A-Z][a-zA-Z\s\-]{1,50})$')
    
    # Exclude common captions, list items or UI dialogue prefixes
    exclude_prefixes = ("fig", "table", "note:", "algorithm", "equation", "host", "speaker")
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            processed_lines.append(line)
            continue
            
        # Headers are generally short
        if len(stripped) > 100:
            processed_lines.append(line)
            continue
            
        lowered = stripped.lower()
        if lowered.startswith(exclude_prefixes) or "figure" in lowered:
            processed_lines.append(line)
            continue
            
        # Normalize text by removing asterisks to match keywords
        clean_text = stripped.replace("**", "").strip()
        clean_lowered = clean_text.lower()
        
        is_keyword = clean_lowered in academic_keywords
        is_bold_num = pattern_bold_num_1.match(stripped) or pattern_bold_num_2.match(stripped)
        is_plain_num = pattern_plain_num.match(stripped)
        
        if is_keyword or is_bold_num or is_plain_num:
            # Promote to markdown header level 2 (##)
            processed_lines.append(f"## {stripped}")
        else:
            processed_lines.append(line)
            
    return "\n".join(processed_lines)

def process_pdf_to_markdown(file_path):

    # 1. 讀取 PDF 並轉換為 Markdown 字串
    md_text = pymupdf4llm.to_markdown(file_path)
    # 預處理粗體論文章節標題，轉換為標題格式
    md_text = preprocess_markdown_headers(md_text)

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