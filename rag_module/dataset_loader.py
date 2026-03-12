"""
Dataset Loader Module
Handles loading, preprocessing, and converting QA datasets into
a standardized (question, answer, source) format.

Supports:
  - Natural Questions (NQ)
  - TriviaQA
  - PubMedQA
  - Custom CSV/JSON datasets
"""

import json
import csv
import os
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset


@dataclass
class QARecord:
    """A single QA record in standardized format."""
    question: str
    answer: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


class DatasetLoader:
    """Loads and preprocesses QA datasets into a unified format."""

    SUPPORTED_HF_DATASETS = {
        "natural_questions": "google-research-datasets/natural_questions",
        "trivia_qa": "mandarjoshi/trivia_qa",
        "pubmed_qa": "qiaojin/PubMedQA",
    }

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "rag_module")
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_hf_dataset(self, name: str, split: str = "train", max_samples: Optional[int] = None) -> list[QARecord]:
        """Load a HuggingFace QA dataset and convert to QARecords.

        Args:
            name: One of 'natural_questions', 'trivia_qa', 'pubmed_qa'.
            split: Dataset split (train / validation / test).
            max_samples: Optional cap on the number of records loaded.

        Returns:
            List of QARecord objects.
        """
        if name not in self.SUPPORTED_HF_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{name}'. Choose from: {list(self.SUPPORTED_HF_DATASETS.keys())}"
            )

        hf_name = self.SUPPORTED_HF_DATASETS[name]
        config = "rc.nocontext" if name == "trivia_qa" else None

        if name == "pubmed_qa":
            ds = load_dataset(hf_name, "pqa_labeled", split=split, cache_dir=self.cache_dir)
        elif config:
            ds = load_dataset(hf_name, config, split=split, cache_dir=self.cache_dir)
        else:
            ds = load_dataset(hf_name, split=split, cache_dir=self.cache_dir)

        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        parser = {
            "natural_questions": self._parse_nq,
            "trivia_qa": self._parse_triviaqa,
            "pubmed_qa": self._parse_pubmedqa,
        }[name]

        return [rec for rec in (parser(row) for row in ds) if rec is not None]

    def load_json(self, filepath: str) -> list[QARecord]:
        """Load a custom JSON dataset.

        Expected format – a JSON array of objects, each with keys:
            question (str), answer (str), source (str, optional).
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            QARecord(
                question=item["question"],
                answer=item["answer"],
                source=item.get("source", ""),
            )
            for item in data
        ]

    def load_csv(self, filepath: str) -> list[QARecord]:
        """Load a custom CSV dataset.

        Expected columns: question, answer, source (optional).
        """
        records: list[QARecord] = []
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    QARecord(
                        question=row["question"],
                        answer=row["answer"],
                        source=row.get("source", ""),
                    )
                )
        return records

    def load_documents(
        self, docs_dir: str, chunk_size: int = 0, chunk_overlap: int = 50
    ) -> list[dict]:
        """Load a knowledge corpus from a directory of text files.

        Each .txt file is treated as one document. Returns a list of dicts
        with keys: 'text', 'source'.

        Args:
            docs_dir: Path to directory of .txt files.
            chunk_size: If > 0, split long documents into chunks of this many
                        words. Recommended: 200 (fits within the 256-token
                        limit of all-MiniLM-L6-v2). 0 = no chunking.
            chunk_overlap: Number of overlapping words between consecutive
                          chunks (helps preserve context at boundaries).
        """
        documents = []
        for fname in sorted(os.listdir(docs_dir)):
            fpath = os.path.join(docs_dir, fname)
            if not os.path.isfile(fpath):
                continue
            if not fname.endswith(".txt"):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                continue

            text = self.preprocess_text(text)

            if chunk_size > 0:
                chunks = self.chunk_text(text, chunk_size, chunk_overlap)
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "source": fname,
                        "chunk_index": i,
                    })
            else:
                documents.append({"text": text, "source": fname})
        return documents

    # ------------------------------------------------------------------
    # Text Preprocessing & Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and normalize raw text.

        - Collapse multiple whitespace/newlines into single spaces
        - Strip leading/trailing whitespace
        """
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
        """Split text into overlapping word-level chunks.

        Args:
            text: The text to split.
            chunk_size: Max words per chunk.
            overlap: Words shared between consecutive chunks.

        Returns:
            List of text chunks.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        step = max(chunk_size - overlap, 1)
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += step
        return chunks

    # ------------------------------------------------------------------
    # Internal parsers for HuggingFace datasets
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_nq(row) -> Optional[QARecord]:
        """Parse a Natural Questions row."""
        question = row.get("question", {})
        if isinstance(question, dict):
            question_text = question.get("text", "")
        else:
            question_text = str(question)

        annotations = row.get("annotations", {})
        short_answers = annotations.get("short_answers", [[]])
        if short_answers and len(short_answers) > 0 and len(short_answers[0]) > 0:
            answer_text = short_answers[0][0].get("text", "") if isinstance(short_answers[0][0], dict) else str(short_answers[0][0])
        else:
            answer_text = ""

        if not question_text or not answer_text:
            return None

        return QARecord(
            question=question_text,
            answer=answer_text,
            source="natural_questions",
        )

    @staticmethod
    def _parse_triviaqa(row) -> Optional[QARecord]:
        """Parse a TriviaQA row."""
        question_text = row.get("question", "")
        answer_obj = row.get("answer", {})
        answer_text = answer_obj.get("value", "") if isinstance(answer_obj, dict) else str(answer_obj)

        if not question_text or not answer_text:
            return None

        return QARecord(
            question=question_text,
            answer=answer_text,
            source="trivia_qa",
        )

    @staticmethod
    def _parse_pubmedqa(row) -> Optional[QARecord]:
        """Parse a PubMedQA row."""
        question_text = row.get("question", "")
        answer_text = row.get("long_answer", "") or row.get("final_decision", "")

        if not question_text or not answer_text:
            return None

        return QARecord(
            question=question_text,
            answer=answer_text,
            source="pubmed_qa",
            metadata={"context": row.get("context", "")},
        )


# ------------------------------------------------------------------
# CLI usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and preview a QA dataset")
    parser.add_argument("--dataset", type=str, default="trivia_qa",
                        choices=list(DatasetLoader.SUPPORTED_HF_DATASETS.keys()),
                        help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()

    loader = DatasetLoader()
    records = loader.load_hf_dataset(args.dataset, split=args.split, max_samples=args.max_samples)

    print(f"\nLoaded {len(records)} records from '{args.dataset}':\n")
    for i, r in enumerate(records):
        print(f"  [{i+1}] Q: {r.question}")
        print(f"       A: {r.answer}")
        print(f"       Source: {r.source}\n")
