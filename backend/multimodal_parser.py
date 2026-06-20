# backend/multimodal_parser.py

import os
import re
import fitz  # PyMuPDF
import sqlite3
import json
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from backend import db, config
import ollama

def extract_page_crops(pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Extracts table and image crops from a PDF page-by-page.
    Saves the crops as PNGs and returns a list of crop descriptions.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    crops_info = []

    print(f"🖼️ Scanning {filename} for visual elements (tables and figures)...")
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # 1. Detect and crop tables
        tabs = page.find_tables()
        for t_idx, table in enumerate(tabs.tables):
            bbox = table.bbox  # Rect coordinate
            # Render with high zoom (2.0x) for legibility in the VLM
            pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(2, 2))
            crop_filename = f"crop_{filename}_p{page_idx}_table_{t_idx}.png"
            image_path = os.path.join(output_dir, crop_filename)
            pix.save(image_path)
            
            # Extract basic cell texts as keyword hints for alignment
            cell_text = ""
            try:
                extracted_cells = table.extract()
                cell_text = " ".join([str(item) for row in extracted_cells for item in row if item])
            except Exception:
                pass
                
            crops_info.append({
                "type": "table",
                "page": page_idx,
                "bbox": list(bbox),
                "image_path": image_path,
                "filename": crop_filename,
                "cell_text": cell_text
            })
            print(f"  Found Table on Page {page_idx + 1} -> Saved crop: {crop_filename}")

        # 2. Detect and crop figures/images
        image_list = page.get_images(full=True)
        for i_idx, img in enumerate(image_list):
            try:
                bbox_info = page.get_image_bbox(img)
                bbox = bbox_info[0] # Extract Rect object
                # Filter out tiny image elements (noise, icons, lines)
                width = bbox.x1 - bbox.x0
                height = bbox.y1 - bbox.y0
                if width < 80 or height < 80:
                    continue
                    
                pix = page.get_pixmap(clip=bbox, matrix=fitz.Matrix(2, 2))
                crop_filename = f"crop_{filename}_p{page_idx}_fig_{i_idx}.png"
                image_path = os.path.join(output_dir, crop_filename)
                pix.save(image_path)
                
                crops_info.append({
                    "type": "figure",
                    "page": page_idx,
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "image_path": image_path,
                    "filename": crop_filename,
                    "cell_text": ""
                })
                print(f"  Found Figure on Page {page_idx + 1} -> Saved crop: {crop_filename}")
            except Exception as e:
                # Bounding box or rendering issues for inline symbols
                continue

    doc.close()
    return crops_info

def query_vision_language_model(image_path: str, crop_type: str, model_name: str = "qwen2-vl") -> Dict[str, str]:
    """
    Sends the cropped image to local Qwen2-VL via Ollama.
    Returns a dictionary containing 'markdown_table' and 'summary'.
    """
    if crop_type == "table":
        prompt = (
            "Analyze the attached table image. First, reconstruct the table data completely and format it as a "
            "standard Github-flavored Markdown table. Second, write a detailed summary under the header 'SUMMARY:' "
            "describing what this table presents, defining its columns/variables, and highlighting key data points or trends."
        )
    else:
        prompt = (
            "Analyze this chart, graph, diagram or formula image. Write a detailed textual summary explaining "
            "what the visualization represents, describing its axes, key labels, variables, math equations, and main conclusions. "
            "Format your response under the header 'SUMMARY:'."
        )

    print(f"🤖 Querying local VLM ({model_name}) for image: {os.path.basename(image_path)}...")
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ]
        )
        content = response["message"]["content"]
        
        # Parse table markdown vs summary
        markdown_table = "N/A"
        summary = content
        
        if "SUMMARY:" in content:
            parts = content.split("SUMMARY:")
            summary = parts[1].strip()
            if crop_type == "table":
                markdown_table = parts[0].strip()
        elif "summary:" in content.lower():
            parts = re.split(r"summary:", content, flags=re.IGNORECASE)
            summary = parts[1].strip()
            if crop_type == "table":
                markdown_table = parts[0].strip()
                
        return {
            "markdown_table": markdown_table,
            "summary": summary
        }
    except Exception as e:
        print(f"❌ Error communicating with Ollama VLM: {e}")
        # Return fallback heuristic descriptions
        return {
            "markdown_table": "| Column 1 | Column 2 |\n|---|---|\n| Data | Data |" if crop_type == "table" else "N/A",
            "summary": f"Fallback: Local VLM evaluation failed for crop {os.path.basename(image_path)}."
        }

def find_best_matching_parent(parent_chunks: List[Dict[str, Any]], page_text: str, cell_text: str = "") -> Dict[str, Any]:
    """
    Aligns the crop to the best parent chunk of the same document using Jaccard word similarity.
    Includes cell texts to prioritize correct chunks for tables.
    """
    # Standardize target text (combine page text and table cell text if available)
    target_text = (page_text + " " + cell_text).lower()
    target_words = set(re.sub(r'[^\w\s]', ' ', target_text).split())
    if not target_words:
        return None

    best_chunk = None
    best_score = -1.0
    
    for chunk in parent_chunks:
        chunk_content = chunk["page_content"].lower()
        chunk_words = set(re.sub(r'[^\w\s]', ' ', chunk_content).split())
        if not chunk_words:
            continue
            
        intersection = target_words.intersection(chunk_words)
        union = target_words.union(chunk_words)
        score = len(intersection) / len(union) if union else 0
        
        if score > best_score:
            best_score = score
            best_chunk = chunk
            
    return best_chunk

def upsert_multimodal_enrichments(pdf_path: str, doc_id: int, crops_info: List[Dict[str, Any]], model_name: str = "qwen2-vl"):
    """
    Performs VLM generation and upserts the multimodal descriptions directly 
    into SQLite and ChromaDB bound to the aligned parent chunk.
    """
    # 1. Fetch all parent chunks belonging to this document from SQLite
    with db.db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, parent_id, page_content, metadata_json 
            FROM parent_chunks 
            WHERE document_id = ?
        """, (doc_id,))
        rows = cursor.fetchall()
    
    if not rows:
        print(f"⚠️ No parent chunks found in SQLite for Document ID {doc_id}.")
        return
        
    parent_chunks = []
    for r in rows:
        parent_chunks.append({
            "id": r[0],
            "parent_id": r[1],
            "page_content": r[2],
            "metadata": json.loads(r[3])
        })

    # Initialize ChromaDB connection to add new child chunks
    embeddings = config.get_embeddings()
    vector_db = Chroma(persist_directory=config.CHROMA_PERSIST_DIR, embedding_function=embeddings)
    
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)

    for idx, crop in enumerate(crops_info):
        # Retrieve page context text for alignment
        page_text = doc[crop["page"]].get_text("text")
        
        # Describe image via VLM
        descriptions = query_vision_language_model(crop["image_path"], crop["type"], model_name)
        markdown_table = descriptions["markdown_table"]
        summary = descriptions["summary"]
        
        # Locate parent chunk
        matched_parent = find_best_matching_parent(parent_chunks, page_text, crop["cell_text"])
        if not matched_parent:
            print(f"⚠️ Could not align crop {crop['filename']} with any parent chunk. Skipping indexing.")
            continue
            
        parent_id = matched_parent["parent_id"]
        print(f"🎯 Crop aligned to parent ID: {parent_id} (Jaccard Match)")
        
        # 2. Update parent chunk text and metadata in SQLite
        updated_content = matched_parent["page_content"]
        updated_content += f"\n\n### [Multimodal Enrichment - Page {crop['page'] + 1} ({crop['type'].upper()})]"
        if crop["type"] == "table" and markdown_table != "N/A":
            updated_content += f"\n#### Extracted Table Data:\n{markdown_table}\n"
        updated_content += f"\n#### Visual Summary:\n{summary}\n"
        
        # Update metadata JSON
        updated_metadata = matched_parent["metadata"]
        if "multimodal_enrichments" not in updated_metadata:
            updated_metadata["multimodal_enrichments"] = []
        updated_metadata["multimodal_enrichments"].append({
            "image_path": crop["image_path"],
            "type": crop["type"],
            "page": crop["page"] + 1,
            "markdown_table": markdown_table,
            "summary": summary
        })
        
        # Update SQLite
        with db.db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE parent_chunks 
                SET page_content = ?, metadata_json = ? 
                WHERE parent_id = ?
            """, (updated_content, json.dumps(updated_metadata), parent_id))
        
        # Reflect change inside the in-memory/list parent chunk for subsequent crops
        matched_parent["page_content"] = updated_content
        matched_parent["metadata"] = updated_metadata
        
        # 3. Insert a new child chunk in ChromaDB representing this table/figure summary
        # This allows similarity searches to hit the VLM-extracted details and map to this parent.
        enrichment_chunk_content = (
            f"Multimodal Data Description for Page {crop['page'] + 1} (Source: {filename})\n"
            f"Type: {crop['type'].capitalize()}\n"
            f"Summary details: {summary}\n"
        )
        if crop["type"] == "table" and markdown_table != "N/A":
            enrichment_chunk_content += f"Table data:\n{markdown_table}\n"
            
        enrichment_metadata = {
            "parent_id": parent_id,
            "source": filename,
            "page": str(crop["page"] + 1),
            "type": "multimodal_enrichment"
        }
        
        enrichment_doc = Document(
            page_content=enrichment_chunk_content,
            metadata=enrichment_metadata
        )
        
        vector_db.add_documents([enrichment_doc])
        print(f"✅ Upserted VLM enrichment chunk into ChromaDB bound to {parent_id}")

    doc.close()
    vector_db.persist()
    print("🎉 Multimodal metadata enrichment pipeline completed successfully!")

def run_pipeline(pdf_path: str, doc_id: int, model_name: str = "qwen2-vl", crops_dir: str = "./assets/multimodal_crops"):
    """Runs the entire multimodal extraction, VLM description, and upsert pipeline."""
    crops_info = extract_page_crops(pdf_path, crops_dir)
    if not crops_info:
        print("ℹ️ No tables or figures detected in this document.")
        return
    upsert_multimodal_enrichments(pdf_path, doc_id, crops_info, model_name)

def process_multimodal_ingestion(pdf_path: str, doc_id: int, model_name: str = "qwen2-vl", crops_dir: str = "./assets/multimodal_crops"):
    """
    Runs the multimodal parser and performs explicit memory reclamation 
    immediately after Qwen2-VL finishes chart parsing and chunk embedding.
    """
    print(f"Initializing transient VLM model: {model_name}...")
    class QwenModelWrapper:
        def __init__(self, name):
            self.name = name
            
    qwen_model = QwenModelWrapper(model_name)
    
    try:
        # Run the main processing pipeline
        crops_info = extract_page_crops(pdf_path, crops_dir)
        if crops_info:
            upsert_multimodal_enrichments(pdf_path, doc_id, crops_info, model_name)
        else:
            print("ℹ️ No tables or figures detected in this document.")
    finally:
        # Explicit memory reclamation triggered immediately after Qwen2-VL finishes
        print(f"Reclaiming memory. Purging {model_name}...")
        
        # Explicitly delete the model instance and trigger garbage collection
        del qwen_model
        import gc
        gc.collect()
        
        # Explicitly purge Apple Silicon GPU memory cache
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("Apple Silicon GPU memory cache (MPS) cleared successfully.")
            
        # Issue Ollama unload command to release from unified memory
        try:
            print(f"Sending unload command to Ollama for {model_name}...")
            ollama.chat(model=model_name, messages=[], keep_alive=0)
            print(f"Ollama memory offloading successful for {model_name}.")
        except Exception as e:
            print(f"Ollama model offload request failed (non-critical): {e}")

        # Signal or preload Llama 3.1 generator LLM to ensure it is loaded only AFTER this purge
        try:
            print("Ensuring generator LLM Llama 3.1 is pre-loaded...")
            ollama.chat(model="llama3.1", messages=[], keep_alive=-1)
            print("Generator LLM Llama 3.1 is active.")
        except Exception as e:
            print(f"Failed to pre-load Llama 3.1 (non-critical): {e}")

if __name__ == "__main__":
    # Test pipeline on the first document in the database
    import sys
    docs = db.get_all_documents()
    if not docs:
        print("❌ No documents found in database. Please run File_processing.py or ingestion first.")
        sys.exit(1)
        
    doc = docs[0]
    doc_id = doc["id"]
    filename = doc["filename"]
    pdf_path = os.path.join("./RAG_files", filename)
    
    if os.path.exists(pdf_path):
        run_pipeline(pdf_path, doc_id)
    else:
        print(f"❌ File not found at {pdf_path}")
