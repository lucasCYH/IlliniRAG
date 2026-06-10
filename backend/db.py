import sqlite3
import json
import os
from backend import config

DB_PATH = config.DB_PATH

# Override sqlite3.connect to set a 30-second timeout, preventing "database is locked" in Streamlit
_original_connect = sqlite3.connect
def connect_with_timeout(*args, **kwargs):
    if "timeout" not in kwargs:
        kwargs["timeout"] = 30.0
    return _original_connect(*args, **kwargs)
sqlite3.connect = connect_with_timeout

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Table for uploaded documents
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            md5_hash TEXT UNIQUE,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Migration: Add md5_hash column if it doesn't exist yet in older databases
    try:
        cursor.execute("ALTER TABLE documents ADD COLUMN md5_hash TEXT")
    except sqlite3.OperationalError:
        pass
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
    # Virtual table for SQLite FTS5 Full-Text Search
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS parent_chunks_fts USING fts5(
            parent_id,
            page_content,
            content='parent_chunks',
            content_rowid='id'
        )
    """)
    # Triggers to keep FTS index synchronized on inserts, deletes, and updates
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS parent_chunks_ai AFTER INSERT ON parent_chunks BEGIN
            INSERT INTO parent_chunks_fts(rowid, parent_id, page_content) VALUES (new.id, new.parent_id, new.page_content);
        END;
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS parent_chunks_ad AFTER DELETE ON parent_chunks BEGIN
            INSERT INTO parent_chunks_fts(parent_chunks_fts, rowid, parent_id, page_content) VALUES('delete', old.id, old.parent_id, old.page_content);
        END;
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS parent_chunks_au AFTER UPDATE ON parent_chunks BEGIN
            INSERT INTO parent_chunks_fts(parent_chunks_fts, rowid, parent_id, page_content) VALUES('delete', old.id, old.parent_id, old.page_content);
            INSERT INTO parent_chunks_fts(rowid, parent_id, page_content) VALUES(new.id, new.parent_id, new.page_content);
        END;
    """)
    # Populate FTS table with any existing pre-migration chunks
    cursor.execute("""
        INSERT OR IGNORE INTO parent_chunks_fts(rowid, parent_id, page_content)
        SELECT id, parent_id, page_content FROM parent_chunks;
    """)
    conn.commit()
    conn.close()

def add_document(filename, md5_hash=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO documents (filename, md5_hash) VALUES (?, ?)", (filename, md5_hash))
        conn.commit()
        doc_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        # IntegrityError is triggered by the unique filename constraint.
        # Find the existing document by filename and update its md5_hash if currently NULL.
        cursor.execute("SELECT id, md5_hash FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        if row:
            doc_id = row[0]
            existing_md5 = row[1]
            if md5_hash and not existing_md5:
                cursor.execute("UPDATE documents SET md5_hash = ? WHERE id = ?", (md5_hash, doc_id))
                conn.commit()
        else:
            # Fallback if query by filename somehow returned nothing
            if md5_hash:
                cursor.execute("SELECT id FROM documents WHERE md5_hash = ?", (md5_hash,))
                res = cursor.fetchone()
                doc_id = res[0] if res else None
            else:
                doc_id = None
    conn.close()
    return doc_id

def get_document_by_md5(md5_hash):
    """Retrieve document by its MD5 hash fingerprint."""
    if not md5_hash:
        return None
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename FROM documents WHERE md5_hash = ?", (md5_hash,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "filename": row[1]}
    return None

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

def search_parent_chunks_fts(query_str: str, limit: int = 5) -> list:
    """Perform BM25 search on parent chunks using SQLite FTS5 virtual table."""
    import re
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Format query for FTS5 (remove non-alphanumeric characters to avoid query syntax errors)
    clean_query = re.sub(r'[^\w\s]', ' ', query_str).strip()
    
    # We split query terms and search for them matching, or simple matching.
    # If the search query contains quotes or operators, FTS5 MATCH might throw.
    # So we wrap in double quotes or fallback to LIKE if FTS query syntax is invalid.
    try:
        cursor.execute("""
            SELECT pc.parent_id, pc.page_content, pc.metadata_json, bm25(parent_chunks_fts) as rank
            FROM parent_chunks_fts fts
            JOIN parent_chunks pc ON pc.id = fts.rowid
            WHERE parent_chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (clean_query, limit))
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"[FTS5 Search] Query syntax failed for '{clean_query}': {e}. Falling back to LIKE.")
        cursor.execute("""
            SELECT parent_id, page_content, metadata_json, 1.0 as rank
            FROM parent_chunks
            WHERE page_content LIKE ?
            LIMIT ?
        """, (f"%{clean_query}%", limit))
        rows = cursor.fetchall()
        
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "parent_id": row[0],
            "page_content": row[1],
            "metadata": json.loads(row[2]),
            "rank": row[3]
        })
    return results

# Initialize DB when module is loaded
init_db()
