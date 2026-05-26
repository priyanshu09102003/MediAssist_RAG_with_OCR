import os
import re
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import fitz                                    
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from core.vector_store import VectorStore


# ── Category auto-detection keywords

CATEGORY_KEYWORDS = {
    "disease":   ["disease", "disorder", "syndrome", "infection", "fever",
                  "cancer", "diabetes", "hypertension", "asthma", "tuberculosis",
                  "malaria", "dengue", "hepatitis", "cholera", "typhoid",
                  "pneumonia", "influenza", "covid", "anaemia", "anemia"],
    "drug":      ["drug", "medicine", "medication", "tablet", "capsule",
                  "injection", "syrup", "dose", "dosage", "paracetamol",
                  "antibiotic", "analgesic", "antiviral", "vaccine"],
    "symptom":   ["symptom", "sign", "pain", "ache", "nausea", "vomiting",
                  "diarrhea", "rash", "swelling", "fatigue", "breathlessness",
                  "cough", "headache", "dizziness", "bleeding"],
    "nutrition": ["nutrition", "diet", "vitamin", "mineral", "calorie",
                  "protein", "carbohydrate", "fat", "supplement", "food"],
    "mental":    ["mental", "anxiety", "depression", "stress", "sleep",
                  "psychiatric", "psychological", "mood", "phobia", "trauma"],
}


def detect_category(text: str) -> str:
    """Guess category from document content using keyword matching."""
    text_lower = text.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            scores[cat] += text_lower.count(kw)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Ingestion tracker (avoid re-indexing) 

TRACKER_DB = config.SQLITE_DB_PATH.parent / "ingest_tracker.db"


def _tracker_conn():
    conn = sqlite3.connect(str(TRACKER_DB))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS indexed_files (
               file_hash TEXT PRIMARY KEY,
               file_name TEXT,
               chunks    INTEGER,
               indexed_at TEXT DEFAULT (datetime('now'))
           )"""
    )
    conn.commit()
    return conn


def _is_indexed(file_hash: str) -> bool:
    with _tracker_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM indexed_files WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row is not None


def _mark_indexed(file_hash: str, file_name: str, chunks: int) -> None:
    with _tracker_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO indexed_files (file_hash, file_name, chunks) VALUES (?,?,?)",
            (file_hash, file_name, chunks),
        )


def _file_hash(path: Path) -> str:
    """MD5 hash of file contents — used as unique identity."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Loaders

def load_pdf(path: Path) -> list[Document]:
    
    docs = []
    try:
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf, 1):
            text = page.get_text("text").strip()
            if len(text) < 50:          
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source":      str(path),
                    "source_name": path.stem.replace("_", " ").title(),
                    "page":        page_num,
                    "file_type":   "pdf",
                    "category":    "",  
                },
            ))
        pdf.close()
    except Exception as e:
        print(f"  ⚠️  PDF error ({path.name}): {e}")
    return docs


def load_txt(path: Path) -> list[Document]:
    """Load a plain text file as a single Document."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        return [Document(
            page_content=text,
            metadata={
                "source":      str(path),
                "source_name": path.stem.replace("_", " ").title(),
                "page":        1,
                "file_type":   "txt",
                "category":    "",
            },
        )]
    except Exception as e:
        print(f"  ⚠️  TXT error ({path.name}): {e}")
        return []


def load_medlineplus_xml(path: Path) -> list[Document]:

    docs = []
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()

        for topic in root.findall(".//health-topic"):
            title    = topic.get("title", "")
            meta_desc = topic.findtext("meta-desc", "")
            full_sum  = topic.findtext("full-summary", "")

            # Gather also-called names and groups
            also_called = [e.text for e in topic.findall("also-called") if e.text]
            groups      = [e.get("url", "") for e in topic.findall("group")]

            content_parts = [
                f"Topic: {title}",
            ]
            if also_called:
                content_parts.append(f"Also known as: {', '.join(also_called)}")
            if meta_desc:
                content_parts.append(f"Summary: {meta_desc}")
            if full_sum:
                # Strip HTML tags from full summary
                clean = re.sub(r"<[^>]+>", " ", full_sum)
                clean = re.sub(r"\s+", " ", clean).strip()
                content_parts.append(f"Details: {clean}")

            text = "\n".join(content_parts).strip()
            if len(text) < 80:
                continue

            docs.append(Document(
                page_content=text,
                metadata={
                    "source":      str(path),
                    "source_name": f"MedlinePlus — {title}",
                    "page":        1,
                    "file_type":   "xml",
                    "category":    "",
                    "topic_title": title,
                },
            ))

    except Exception as e:
        print(f"  ⚠️  XML error ({path.name}): {e}")

    return docs


# ── Chunker 

def chunk_documents(documents: list[Document]) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Assign category to each chunk
    for chunk in chunks:
        if not chunk.metadata.get("category"):
            chunk.metadata["category"] = detect_category(chunk.page_content)

    return chunks


# ── Main ingestion pipeline 

LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".xml": load_medlineplus_xml,
}


def ingest_file(path: Path, vs: VectorStore, force: bool = False) -> int:
    """
    Ingest a single file into ChromaDB.
    Skips if already indexed (unless force=True).
    Returns number of chunks added (0 if skipped).
    """
    if path.suffix.lower() not in LOADERS:
        print(f"  ⏭️  Skipping unsupported type: {path.name}")
        return 0

    file_hash = _file_hash(path)

    if not force and _is_indexed(file_hash):
        print(f"  ⏭️  Already indexed: {path.name}")
        return 0

    print(f"  📄 Loading: {path.name}")
    loader   = LOADERS[path.suffix.lower()]
    raw_docs = loader(path)

    if not raw_docs:
        print(f"  ⚠️  No content extracted from: {path.name}")
        return 0

    print(f"     → {len(raw_docs)} pages/sections extracted")

    chunks = chunk_documents(raw_docs)
    print(f"     → {len(chunks)} chunks created")

    added = vs.add_documents(chunks)
    _mark_indexed(file_hash, path.name, added)

    print(f"  ✅ {path.name}: {added} chunks indexed")
    return added


def ingest_all(
    kb_dir: Optional[Path] = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    
    kb_dir = kb_dir or config.MEDICAL_KB_PATH
    vs     = VectorStore()

    supported = [
        p for p in Path(kb_dir).rglob("*")
        if p.is_file() and p.suffix.lower() in LOADERS
    ]

    if not supported:
        print(f"⚠️  No supported files found in {kb_dir}")
        print("   Supported formats: .pdf, .txt, .xml")
        print("   Download medical documents from:")
        print("   • https://medlineplus.gov/xml.html  (XML bulk)")
        print("   • https://www.who.int/news-room/fact-sheets  (PDFs)")
        return {"files_found": 0, "files_indexed": 0, "total_chunks": 0}

    print(f"\n{'='*55}")
    print(f" MediAssist — Knowledge Base Ingestion")
    print(f" Directory : {kb_dir}")
    print(f" Files found: {len(supported)}")
    print(f"{'='*55}\n")

    total_chunks  = 0
    files_indexed = 0

    for path in sorted(supported):
        n = ingest_file(path, vs, force=force)
        if n > 0:
            total_chunks  += n
            files_indexed += 1

    stats = vs.stats()
    print(f"\n{'='*55}")
    print(f" Ingestion complete!")
    print(f" Files indexed this run : {files_indexed}")
    print(f" Chunks added this run  : {total_chunks}")
    print(f" Total chunks in KB     : {stats['total_chunks']}")
    print(f" Categories             : {stats['by_category']}")
    print(f"{'='*55}\n")

    return {
        "files_found":   len(supported),
        "files_indexed": files_indexed,
        "total_chunks":  total_chunks,
        "kb_stats":      stats,
    }


# ── CLI entry point 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest medical documents into ChromaDB")
    parser.add_argument("--force",   action="store_true", help="Re-index all files")
    parser.add_argument("--dir",     type=str, default=None, help="Custom KB directory path")
    parser.add_argument("--file",    type=str, default=None, help="Ingest a single file")
    args = parser.parse_args()

    if args.file:
        vs = VectorStore()
        n  = ingest_file(Path(args.file), vs, force=args.force)
        print(f"Done. {n} chunks added.")
    else:
        kb = Path(args.dir) if args.dir else None
        ingest_all(kb_dir=kb, force=args.force)