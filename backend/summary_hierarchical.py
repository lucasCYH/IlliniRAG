# backend/summary_hierarchical.py

"""Hierarchical Summary Index utilities.

Provides functions to group document content by headers (Chapter/Section),
generate summaries using an LLM, and store/retrieve them in dedicated Chroma collections.
"""

import os
import re
import concurrent.futures
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from backend import config

def _should_ignore_header(header_title: str) -> bool:
    if not header_title:
        return False
    normalized = header_title.replace("**", "").strip().lower()
    # Remove number prefix if any, e.g. "7. acknowledgements" -> "acknowledgements"
    normalized = re.sub(r'^\d+(\.\d+)*\.?\s*', '', normalized)
    ignore_keywords = {"references", "acknowledgements", "acknowledgement", "acknowledgment", "bibliography"}
    return normalized in ignore_keywords

def _init_embeddings():
    return config.get_embeddings()

def _get_chapter_store():
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=_init_embeddings(),
        collection_name=config.SUMMARY_COLLECTION_CHAPTER
    )

def _get_section_store():
    return Chroma(
        persist_directory=config.CHROMA_PERSIST_DIR,
        embedding_function=_init_embeddings(),
        collection_name=config.SUMMARY_COLLECTION_SECTION
    )

def generate_summary(text: str, level: str, title: str) -> str:
    """Generate summary for a given text using LLM, with fallback to basic truncation."""
    prompt = (
        f"You are a helpful academic assistant.\n"
        f"Provide a concise summary (max 150 words) for the {level} titled '{title}'.\n"
        f"Focus on the main ideas, structure, and key details.\n\n"
        f"Content:\n{text[:6000]}"  # Truncate to avoid context window issues
    )
    try:
        llm = Ollama(model=config.SUMMARY_MODEL, temperature=0)
        summary = llm.invoke(prompt)
        return summary.strip()
    except Exception as e:
        print(f"Warning: Failed to generate summary using LLM due to: {e}. Using fallback truncation.")
        # Fallback summary
        words = text.split()
        fallback_text = " ".join(words[:50]) + "..."
        return f"[Fallback Summary for {level} '{title}']: {fallback_text}"

def generate_hierarchical_summary(docs: List[Document], doc_id: int, filename: str, progress_callback=None):
    """
    Groups docs by Header 2 (chapters) and Header 3 (sections) if possible,
    falls back to Header 1 and Header 2. Generates summaries, and saves them to Chroma.
    """
    chapters = {}
    sections = {}
    
    # Group content
    for doc in docs:
        h1 = doc.metadata.get("Header 1", "General Introduction")
        h2 = doc.metadata.get("Header 2", "")
        h3 = doc.metadata.get("Header 3", "")
        
        # New grouping hierarchy strategy:
        # If Header 2 is present, treat it as Chapter, Header 3 as Section.
        # Otherwise, treat Header 1 as Chapter, Header 2 as Section.
        if h2:
            ch_name = h2
            sec_name = h3 if h3 else ""
        else:
            ch_name = h1
            sec_name = h2 if h2 else ""
            
        if _should_ignore_header(ch_name):
            continue
            
        if ch_name not in chapters:
            chapters[ch_name] = []
        chapters[ch_name].append(doc.page_content)
        
        if sec_name and not _should_ignore_header(sec_name):
            key = (ch_name, sec_name)
            if key not in sections:
                sections[key] = []
            sections[key].append(doc.page_content)

    total_tasks = len(chapters) + len(sections)
    if total_tasks == 0:
        if progress_callback:
            progress_callback(100, "No content headers found to summarize.")
        return

    completed_tasks = 0
    
    # 1. Process Chapters
    chapter_docs = []
    if chapters:
        print(f"Summarizing {len(chapters)} Chapters in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(generate_summary, "\n".join(contents), "chapter", h1): h1 
                for h1, contents in chapters.items()
            }
            
            for future in concurrent.futures.as_completed(futures):
                h1 = futures[future]
                try:
                    summary = future.result()
                except Exception as e:
                    summary = f"[Error generating summary for chapter '{h1}']: {e}"
                    print(f"Error for chapter '{h1}': {e}")
                    
                chapter_docs.append(Document(
                    page_content=summary,
                    metadata={
                        "doc_id": doc_id,
                        "source": filename,
                        "chapter": h1,
                        "level": "chapter",
                        "title": h1
                    }
                ))
                completed_tasks += 1
                if progress_callback:
                    progress_callback(
                        int((completed_tasks / total_tasks) * 100),
                        f"Generated chapter summary for: {h1}"
                    )

    # Save Chapters to Chroma
    if chapter_docs:
        chapter_store = _get_chapter_store()
        chapter_store.add_documents(chapter_docs)
        chapter_store.persist()
        print(f"Saved {len(chapter_docs)} chapter summaries to Chroma.")

    # 2. Process Sections
    section_docs = []
    if sections:
        print(f"Summarizing {len(sections)} Sections in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(generate_summary, "\n".join(contents), "section", h2): (h1, h2) 
                for (h1, h2), contents in sections.items()
            }
            
            for future in concurrent.futures.as_completed(futures):
                h1, h2 = futures[future]
                try:
                    summary = future.result()
                except Exception as e:
                    summary = f"[Error generating summary for section '{h2}']: {e}"
                    print(f"Error for section '{h2}' in chapter '{h1}': {e}")
                    
                section_docs.append(Document(
                    page_content=summary,
                    metadata={
                        "doc_id": doc_id,
                        "source": filename,
                        "chapter": h1,
                        "section": h2,
                        "level": "section",
                        "title": f"{h1} > {h2}"
                    }
                ))
                completed_tasks += 1
                if progress_callback:
                    progress_callback(
                        int((completed_tasks / total_tasks) * 100),
                        f"Generated section summary for: {h2} in {h1}"
                    )

    # Save Sections to Chroma
    if section_docs:
        section_store = _get_section_store()
        section_store.add_documents(section_docs)
        section_store.persist()
        print(f"Saved {len(section_docs)} section summaries to Chroma.")

    if progress_callback:
        progress_callback(100, "Hierarchical summaries generated successfully!")

def get_document_hierarchy(doc_id: int) -> Dict[str, Any]:
    """
    Returns the hierarchical summaries of a document.
    Structure:
    {
        "filename": str,
        "hierarchy": {
            "chapter_title": {
                "summary": str,
                "sections": {
                    "section_title": "summary"
                }
            }
        }
    }
    """
    chapter_store = _get_chapter_store()
    section_store = _get_section_store()
    
    # Query Chroma (note: we use doc_id integer comparison, but Chroma where filters can handle int)
    chapters_data = chapter_store.get(where={"doc_id": doc_id})
    sections_data = section_store.get(where={"doc_id": doc_id})
    
    hierarchy = {}
    filename = "Unknown"
    
    # Process chapters
    if chapters_data and chapters_data.get("documents"):
        for doc_str, meta in zip(chapters_data["documents"], chapters_data["metadatas"]):
            filename = meta.get("source", filename)
            ch_name = meta.get("chapter", "General")
            hierarchy[ch_name] = {
                "summary": doc_str,
                "sections": {}
            }
            
    # Process sections
    if sections_data and sections_data.get("documents"):
        for doc_str, meta in zip(sections_data["documents"], sections_data["metadatas"]):
            filename = meta.get("source", filename)
            ch_name = meta.get("chapter", "General")
            sec_name = meta.get("section", "General Section")
            
            if ch_name not in hierarchy:
                hierarchy[ch_name] = {
                    "summary": "No chapter summary available.",
                    "sections": {}
                }
            hierarchy[ch_name]["sections"][sec_name] = doc_str
            
    return {
        "filename": filename,
        "hierarchy": hierarchy
    }
