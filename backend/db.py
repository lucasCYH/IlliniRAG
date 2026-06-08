import sqlite3
import json
import os
from backend import config

DB_PATH = config.DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Table for uploaded documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Table for parent chunks (replaces parent_store.json)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parent_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            parent_id TEXT UNIQUE,
            page_content TEXT,
            metadata_json TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    """)
    # Table for user notes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_document(filename):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO documents (filename) VALUES (?)", (filename,))
        conn.commit()
        doc_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        doc_id = cursor.fetchone()[0]
    conn.close()
    return doc_id

def add_parent_chunk(document_id, parent_id, page_content, metadata):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    metadata_json = json.dumps(metadata)
    cursor.execute("""
        INSERT OR REPLACE INTO parent_chunks (document_id, parent_id, page_content, metadata_json)
        VALUES (?, ?, ?, ?)
    """, (document_id, parent_id, page_content, metadata_json))
    conn.commit()
    conn.close()

def get_parent_chunk(parent_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT page_content, metadata_json FROM parent_chunks WHERE parent_id = ?", (parent_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"page_content": row[0], "metadata": json.loads(row[1])}
    return None

def get_all_documents():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename FROM documents ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "filename": row[1]} for row in rows]

def delete_document(doc_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM parent_chunks WHERE document_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()

def add_note(content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO notes (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()

def get_all_notes():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, content, created_at FROM notes ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "content": row[1], "created_at": row[2]} for row in rows]

def delete_note(note_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()

def get_all_parent_chunks_text():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT page_content FROM parent_chunks")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def get_parent_chunks_text_by_docs(doc_ids=None):
    """Fetch all parent chunks page content for specific document IDs."""
    if not doc_ids:
        return get_all_parent_chunks_text()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(doc_ids))
    cursor.execute(f"SELECT page_content FROM parent_chunks WHERE document_id IN ({placeholders})", tuple(doc_ids))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def get_parent_chunks_by_document(document_id):
    """Fetch chunks content for a specific document ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT page_content FROM parent_chunks WHERE document_id = ?", (document_id,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

def update_note(note_id, content):
    """Update note content."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE notes SET content = ? WHERE id = ?", (content, note_id))
    conn.commit()
    conn.close()

# Initialize DB when module is loaded
init_db()
