"""
core/vector_store.py
---------------------
ChromaDB wrapper for the medical knowledge base.

Collections:
    medical_kb   — chunked medical documents (diseases, drugs, symptoms)

Provides:
    - VectorStore class  : add, search, filter, delete, stats
    - get_retriever()    : LangChain-compatible retriever for RAG chain
"""

import os
from typing import Optional
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


# ── Embedding model (singleton) 

_embeddings_instance = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the multilingual sentence-transformers embedding model (cached)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        print("  Loading embedding model (first run may take ~30s to download)...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("  Embedding model ready.")
    return _embeddings_instance


# ── VectorStore 
class VectorStore:
    """
    Thin wrapper around ChromaDB + LangChain Chroma.

    Usage:
        vs = VectorStore()
        vs.add_documents(docs)
        results = vs.search("fever and headache", k=5)
    """

    COLLECTION_NAME = "medical_kb"

    def __init__(self):
        self.embeddings  = get_embeddings()
        self.persist_dir = config.CHROMA_DB_PATH

        # Persistent Chroma client
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # LangChain Chroma wrapper (used for search + retriever)
        self._db = Chroma(
            client=self._client,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    # ── Write 

    def add_documents(self, documents: list[Document], batch_size: int = 50) -> int:
        """
        Embed and store a list of LangChain Documents.
        Processes in batches to avoid API rate limits.
        Returns total number of chunks stored.
        """
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self._db.add_documents(batch)
            total += len(batch)
            print(f"  Stored {total}/{len(documents)} chunks...", end="\r")
        print()
        return total

    def delete_by_source(self, source_path: str) -> None:
        """Remove all chunks that came from a specific file."""
        collection = self._client.get_collection(self.COLLECTION_NAME)
        results = collection.get(where={"source": source_path})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            print(f"  Deleted {len(results['ids'])} chunks from: {source_path}")

    # ── Search 

    def search(
        self,
        query: str,
        k: int = None,
        category: Optional[str] = None,
    ) -> list[Document]:
        """
        Semantic similarity search.

        Args:
            query    : natural-language query
            k        : number of results (defaults to config.RETRIEVER_TOP_K)
            category : optional metadata filter — 'disease' | 'drug' | 'symptom'
                       | 'nutrition' | 'general'
        Returns:
            List of LangChain Documents with page_content + metadata
        """
        k = k or config.RETRIEVER_TOP_K

        search_kwargs = {"k": k}
        if category:
            search_kwargs["filter"] = {"category": category}

        return self._db.similarity_search(query, **search_kwargs)

    def search_with_scores(
        self,
        query: str,
        k: int = None,
        category: Optional[str] = None,
    ) -> list[tuple[Document, float]]:
        """Same as search() but returns (Document, relevance_score) tuples."""
        k = k or config.RETRIEVER_TOP_K
        search_kwargs = {"k": k}
        if category:
            search_kwargs["filter"] = {"category": category}

        return self._db.similarity_search_with_relevance_scores(query, **search_kwargs)

    def format_context(self, query: str, category: Optional[str] = None) -> str:
        """
        Search and return a single formatted string for injection into prompts.
        Each chunk is separated by a divider with its source label.
        """
        docs = self.search(query, category=category)
        if not docs:
            return "No relevant medical information found in knowledge base."

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_name", "Medical Reference")
            cat    = doc.metadata.get("category", "general")
            parts.append(
                f"[Reference {i} | {source} | {cat}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Retriever (for LangChain RAG chain)

    def get_retriever(self, k: int = None, category: Optional[str] = None):
        """
        Return a LangChain BaseRetriever for use in RetrievalQA / LCEL chains.

        Example:
            retriever = vs.get_retriever(k=5)
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        """
        search_kwargs = {"k": k or config.RETRIEVER_TOP_K}
        if category:
            search_kwargs["filter"] = {"category": category}

        return self._db.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    # ── Stats 

    def stats(self) -> dict:
        """Return collection stats: total chunks, categories breakdown."""
        try:
            collection = self._client.get_collection(self.COLLECTION_NAME)
            total = collection.count()

            # Count per category
            all_meta = collection.get(include=["metadatas"])["metadatas"] or []
            category_counts: dict[str, int] = {}
            source_counts:   dict[str, int] = {}

            for m in all_meta:
                cat = m.get("category", "unknown")
                src = m.get("source_name", "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1
                source_counts[src]   = source_counts.get(src, 0)  + 1

            return {
                "total_chunks":    total,
                "by_category":     category_counts,
                "by_source":       source_counts,
                "collection_name": self.COLLECTION_NAME,
                "persist_dir":     self.persist_dir,
            }
        except Exception:
            return {"total_chunks": 0, "by_category": {}, "by_source": {}}

    def is_empty(self) -> bool:
        return self.stats()["total_chunks"] == 0


# ── Module-level singleton 

vector_store = VectorStore()


# ── Quick test 
if __name__ == "__main__":
    print("Testing VectorStore...")
    vs = VectorStore()

    # Insert a test document
    test_docs = [
        Document(
            page_content=(
                "Dengue fever is a mosquito-borne viral infection causing "
                "high fever, severe headache, pain behind the eyes, muscle "
                "and joint pains, rash, and mild bleeding. It is caused by "
                "the dengue virus transmitted by Aedes mosquitoes."
            ),
            metadata={
                "source":      "test_doc",
                "source_name": "WHO Dengue Fact Sheet",
                "category":    "disease",
                "page":        1,
            },
        ),
        Document(
            page_content=(
                "Paracetamol (acetaminophen) is used to treat mild to moderate "
                "pain and fever. Standard adult dose: 500mg–1g every 4–6 hours, "
                "maximum 4g per day. Avoid in severe liver disease."
            ),
            metadata={
                "source":      "test_doc",
                "source_name": "WHO Essential Medicines",
                "category":    "drug",
                "page":        1,
            },
        ),
    ]

    added = vs.add_documents(test_docs)
    print(f"  ✅ Added {added} test chunks")

    results = vs.search("fever headache mosquito", k=2)
    print(f"  ✅ Search returned {len(results)} results")
    print(f"     Top result: {results[0].page_content[:80]}...")

    ctx = vs.format_context("dengue symptoms")
    print(f"  ✅ Context formatted ({len(ctx)} chars)")

    stats = vs.stats()
    print(f"  ✅ Stats: {stats['total_chunks']} total chunks | {stats['by_category']}")

    # Clean up test data
    vs.delete_by_source("test_doc")
    print("  ✅ Test data cleaned up")
    print("\nVectorStore working correctly.")