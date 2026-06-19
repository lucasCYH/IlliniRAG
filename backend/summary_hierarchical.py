# backend/summary_hierarchical.py

"""Hierarchical Summary Index utilities.

Provides functions to group document content by headers (Chapter/Section),
generate summaries using an LLM, and store/retrieve them in dedicated Chroma collections.
"""

import os
import re
import json
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

def generate_chapter_and_sections_summaries(ch_name: str, ch_text: str, ch_sections: List[str]) -> Dict[str, Any]:
    """Generate chapter summary and summaries for all its sections in a single LLM call."""
    if not ch_sections:
        # No sections, generate chapter summary only
        prompt = (
            f"You are a helpful academic assistant.\n"
            f"Provide a concise summary (max 150 words) for the chapter titled '{ch_name}'.\n"
            f"Focus on the main ideas, structure, and key details.\n\n"
            f"Content:\n{ch_text[:6000]}"
        )
        try:
            llm = Ollama(model=config.SUMMARY_MODEL, temperature=0)
            summary = llm.invoke(prompt).strip()
            return {"chapter_summary": summary, "sections": {}}
        except Exception as e:
            print(f"Warning: Failed to generate summary for chapter '{ch_name}': {e}")
            return {"chapter_summary": f"[Fallback Summary for chapter '{ch_name}']", "sections": {}}

    # We have sections, request JSON
    prompt = (
        f"You are a helpful academic assistant.\n"
        f"You are given a chapter from an academic paper titled '{ch_name}'.\n"
        f"This chapter contains the following sections: {', '.join(ch_sections)}.\n\n"
        f"Please provide:\n"
        f"1. A concise summary (max 150 words) for the entire chapter.\n"
        f"2. A concise summary (max 100 words) for each section listed above based on the text.\n\n"
        f"You MUST format your response strictly as a JSON object with the following schema:\n"
        f"{{\n"
        f"  \"chapter_summary\": \"string\",\n"
        f"  \"sections\": {{\n"
        f"    \"Section Name 1\": \"string\",\n"
        f"    \"Section Name 2\": \"string\"\n"
        f"  }}\n"
        f"}}\n\n"
        f"Content:\n{ch_text[:6000]}"
    )
    try:
        llm = Ollama(model=config.SUMMARY_MODEL, temperature=0, format="json")
        response = llm.invoke(prompt).strip()
        result = json.loads(response)
        if "chapter_summary" in result and isinstance(result.get("sections"), dict):
            # Normalize keys to match exactly
            normalized_sections = {}
            for sec in ch_sections:
                matched_key = None
                for k in result["sections"].keys():
                    if k.strip().lower() == sec.strip().lower():
                        matched_key = k
                        break
                if matched_key:
                    normalized_sections[sec] = result["sections"][matched_key]
                else:
                    print(f"Warning: Section '{sec}' missing in JSON response for chapter '{ch_name}'.")
            result["sections"] = normalized_sections
            return result
    except Exception as e:
        print(f"Error in batch generation for chapter '{ch_name}': {e}")
        
    # Fallback to individual summaries if JSON generation/parsing fails
    print(f"Falling back to individual summaries for chapter '{ch_name}'")
    ch_summary = generate_summary(ch_text, "chapter", ch_name)
    return {
        "chapter_summary": ch_summary,
        "sections": {},
        "need_fallback_sections": True
    }

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

    total_tasks = len(chapters)
    if total_tasks == 0:
        if progress_callback:
            progress_callback(100, "No content headers found to summarize.")
        return

    # Map from chapter name to section names under it
    chapter_to_sections = {}
    for ch_name in chapters.keys():
        chapter_to_sections[ch_name] = [s_name for (c_name, s_name) in sections.keys() if c_name == ch_name]

    completed_tasks = 0
    chapter_docs = []
    section_docs = []
    
    print(f"Summarizing {len(chapters)} Chapters in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                generate_chapter_and_sections_summaries,
                ch_name,
                "\n".join(chapters[ch_name]),
                chapter_to_sections[ch_name]
            ): ch_name
            for ch_name in chapters.keys()
        }
        
        for future in concurrent.futures.as_completed(futures):
            ch_name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Exception during batch generation for chapter '{ch_name}': {e}")
                result = {
                    "chapter_summary": f"[Error generating summary for chapter '{ch_name}']: {e}",
                    "sections": {},
                    "need_fallback_sections": True
                }
            
            # Save chapter summary document
            chapter_docs.append(Document(
                page_content=result["chapter_summary"],
                metadata={
                    "doc_id": doc_id,
                    "source": filename,
                    "chapter": ch_name,
                    "level": "chapter",
                    "title": ch_name
                }
            ))
            
            # Save section summary documents
            for sec_name, sec_summary in result.get("sections", {}).items():
                section_docs.append(Document(
                    page_content=sec_summary,
                    metadata={
                        "doc_id": doc_id,
                        "source": filename,
                        "chapter": ch_name,
                        "section": sec_name,
                        "level": "section",
                        "title": f"{ch_name} > {sec_name}"
                    }
                ))
            
            # Check for fallback sections if any were missed or if explicitly flagged
            missing_sections = []
            if result.get("need_fallback_sections", False):
                missing_sections = chapter_to_sections[ch_name]
            else:
                for sec_name in chapter_to_sections[ch_name]:
                    if sec_name not in result.get("sections", {}):
                        missing_sections.append(sec_name)
            
            if missing_sections:
                print(f"Generating individual summaries for {len(missing_sections)} missing sections in chapter '{ch_name}'...")
                for sec_name in missing_sections:
                    sec_content_list = sections.get((ch_name, sec_name), [])
                    sec_text = "\n".join(sec_content_list)
                    sec_summary = generate_summary(sec_text, "section", sec_name)
                    section_docs.append(Document(
                        page_content=sec_summary,
                        metadata={
                            "doc_id": doc_id,
                            "source": filename,
                            "chapter": ch_name,
                            "section": sec_name,
                            "level": "section",
                            "title": f"{ch_name} > {sec_name}"
                        }
                    ))
            
            completed_tasks += 1
            if progress_callback:
                progress_callback(
                    int((completed_tasks / total_tasks) * 100),
                    f"Generated chapter summary for: {ch_name} (with sections)"
                )

    # Save Chapters to Chroma
    if chapter_docs:
        chapter_store = _get_chapter_store()
        chapter_store.add_documents(chapter_docs)
        chapter_store.persist()
        print(f"Saved {len(chapter_docs)} chapter summaries to Chroma.")

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
