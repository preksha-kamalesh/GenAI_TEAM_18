"""
Retriever Module
High-level interface that ties together DatasetLoader and VectorDB
to provide end-to-end retrieval: Question → Top-k Relevant Documents.

This is the main entry point the downstream LLM Generation module will call.
"""

import os
from typing import Optional

from .dataset_loader import DatasetLoader
from .vector_db import VectorDB


class Retriever:
    """End-to-end retriever: indexes a knowledge corpus and serves queries."""

    def __init__(
        self,
        model_name: str = VectorDB.DEFAULT_MODEL,
        index_path: Optional[str] = None,
        top_k: int = 5,
    ):
        """
        Args:
            model_name: SentenceTransformer model for embeddings.
            index_path: Path to save/load the FAISS index.
            top_k: Default number of documents to retrieve per query.
        """
        self.top_k = top_k
        self.loader = DatasetLoader()
        self.db = VectorDB(model_name=model_name, index_path=index_path)
        self._indexed = False

    # ------------------------------------------------------------------
    # Corpus Loading
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[dict], text_key: str = "text") -> None:
        """Build the vector index from a list of documents.

        Args:
            documents: List of dicts, each with at least a `text_key` field.
            text_key: The key containing the document text.
        """
        self.db.build_index(documents, text_key=text_key)
        self._indexed = True

    def index_from_directory(self, docs_dir: str) -> None:
        """Load .txt files from a directory and index them.

        Args:
            docs_dir: Path to directory containing .txt document files.
        """
        documents = self.loader.load_documents(docs_dir)
        if not documents:
            raise ValueError(f"No .txt documents found in {docs_dir}")
        self.index_documents(documents)

    def index_from_qa_dataset(self, dataset_name: str, split: str = "train",
                              max_samples: Optional[int] = None) -> None:
        """Load a HuggingFace QA dataset and index the questions + answers as documents.

        Useful for building a corpus from existing QA pairs.
        """
        records = self.loader.load_hf_dataset(dataset_name, split=split, max_samples=max_samples)
        documents = [
            {"text": f"{r.question} {r.answer}", "source": r.source, "question": r.question, "answer": r.answer}
            for r in records
        ]
        self.index_documents(documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Retrieve the most relevant documents for a given query.

        Args:
            query: User question or search string.
            top_k: Number of results (overrides default if provided).

        Returns:
            List of dicts with keys: 'document', 'score', 'rank'.
        """
        if not self._indexed and self.db.total_documents == 0:
            raise RuntimeError("No documents indexed. Call index_documents() or load() first.")
        k = top_k or self.top_k
        return self.db.search(query, top_k=k)

    def retrieve_text(self, query: str, top_k: Optional[int] = None, text_key: str = "text") -> list[str]:
        """Convenience method: retrieve and return only the document texts."""
        results = self.retrieve(query, top_k=top_k)
        return [r["document"][text_key] for r in results]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save the FAISS index and metadata to disk."""
        self.db.save(path)

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously saved index."""
        self.db.load(path)
        self._indexed = True

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def total_documents(self) -> int:
        return self.db.total_documents

    def __repr__(self) -> str:
        return f"Retriever(db={self.db}, top_k={self.top_k})"


# ------------------------------------------------------------------
# CLI usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Retriever – query documents")
    parser.add_argument("--docs-dir", type=str, help="Directory of .txt documents to index")
    parser.add_argument("--query", type=str, required=True, help="Question to search for")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-path", type=str, default=None, help="Path to save/load index")
    args = parser.parse_args()

    retriever = Retriever(top_k=args.top_k, index_path=args.index_path)

    if args.index_path and os.path.exists(os.path.join(args.index_path, "index.faiss")):
        print(f"Loading existing index from {args.index_path}...")
        retriever.load()
    elif args.docs_dir:
        print(f"Indexing documents from {args.docs_dir}...")
        retriever.index_from_directory(args.docs_dir)
        if args.index_path:
            retriever.save()
            print(f"Index saved to {args.index_path}")
    else:
        print("Provide --docs-dir to index documents or --index-path to load an existing index.")
        exit(1)

    print(f"\nIndex contains {retriever.total_documents} documents.\n")
    print(f"Query: '{args.query}'\n")

    results = retriever.retrieve(args.query)
    for r in results:
        doc = r["document"]
        print(f"  [{r['rank']}] (score={r['score']:.4f})")
        print(f"      Source: {doc.get('source', 'N/A')}")
        print(f"      Text:   {doc['text'][:200]}...\n")
