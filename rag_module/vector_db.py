"""
Vector Database Module
Handles embedding generation and FAISS/ChromaDB-backed vector storage
for efficient similarity search over a knowledge corpus.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (default)
"""

import os
import json
import numpy as np
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer


class VectorDB:
    """Manages document embeddings and FAISS-based similarity search."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        index_path: Optional[str] = None,
    ):
        """
        Args:
            model_name: SentenceTransformer model to use for embeddings.
            index_path: Directory to save/load the FAISS index + metadata.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        self.index_path = index_path

        # FAISS index (L2 distance)
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self.embedding_dim)
        # Parallel list of document metadata aligned with index rows
        self.documents: list[dict] = []

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into dense embeddings.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[dict], text_key: str = "text") -> None:
        """Embed and add documents to the FAISS index.

        Args:
            documents: List of dicts. Each must contain `text_key`.
            text_key: Key in each dict that holds the main text.
        """
        texts = [doc[text_key] for doc in documents]
        embeddings = self.encode(texts)
        self.index.add(embeddings)
        self.documents.extend(documents)

    def build_index(self, documents: list[dict], text_key: str = "text") -> None:
        """Reset the index and build from scratch with the given documents."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.documents = []
        self.add_documents(documents, text_key=text_key)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for the top-k most similar documents to a query.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts, each with keys: 'document', 'score', 'rank'.
        """
        if self.index.ntotal == 0:
            return []

        query_embedding = self.encode([query])
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                continue
            results.append({
                "document": self.documents[idx],
                "score": float(dist),
                "rank": rank + 1,
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save the FAISS index and document metadata to disk."""
        save_dir = path or self.index_path
        if save_dir is None:
            raise ValueError("No save path provided. Pass `path` or set `index_path` at init.")
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))
        with open(os.path.join(save_dir, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "embedding_dim": self.embedding_dim}, f)

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved FAISS index and document metadata."""
        load_dir = path or self.index_path
        if load_dir is None:
            raise ValueError("No load path provided.")
        index_file = os.path.join(load_dir, "index.faiss")
        docs_file = os.path.join(load_dir, "documents.json")
        if not os.path.exists(index_file) or not os.path.exists(docs_file):
            raise FileNotFoundError(f"Index files not found in {load_dir}")
        self.index = faiss.read_index(index_file)
        with open(docs_file, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def total_documents(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"VectorDB(model='{self.model_name}', "
            f"dim={self.embedding_dim}, docs={self.total_documents})"
        )


# ------------------------------------------------------------------
# CLI usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test with sample documents
    sample_docs = [
        {"text": "The Eiffel Tower is located in Paris, France.", "source": "wiki"},
        {"text": "Python is a popular programming language for AI.", "source": "wiki"},
        {"text": "The mitochondria is the powerhouse of the cell.", "source": "biology"},
        {"text": "Shakespeare wrote Romeo and Juliet.", "source": "literature"},
        {"text": "FAISS is a library for efficient similarity search.", "source": "tech"},
    ]

    db = VectorDB()
    print(f"Initialized: {db}")
    db.build_index(sample_docs)
    print(f"After indexing: {db}")

    query = "Where is the Eiffel Tower?"
    results = db.search(query, top_k=3)
    print(f"\nQuery: '{query}'")
    for r in results:
        print(f"  [{r['rank']}] (score={r['score']:.4f}) {r['document']['text']}")
