from __future__ import annotations

import logging
import os
import time
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from generation import RAGGenerator, ClaimExtractor
from rag_module import Retriever
from verification import ClaimVerifier


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class EvidenceAwareFallbackBackend:
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]

    @staticmethod
    def _extract_documents(prompt: str) -> list[str]:
        pattern = (
            r"\[Document\s+\d+\]\s+\([^\)]*\)\n(.*?)"
            r"(?=\n\n\[Document\s+\d+\]|\n\nQuestion:|\Z)"
        )
        matches = re.findall(pattern, prompt, flags=re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @staticmethod
    def _extract_question(prompt: str) -> str:
        match = re.search(r"\nQuestion:\n(.*?)\n\nInstructions:", prompt, flags=re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _clean_sentence(sent: str) -> str:
        cleaned = re.sub(r"\s+", " ", sent).strip()
        cleaned = re.sub(r"^[Qq]uestion\s*[:\-]\s*", "", cleaned)
        cleaned = re.sub(r"^[Ss]ource\s*\d+\s*[:\-]\s*", "", cleaned)
        if cleaned.endswith("?"):
            cleaned = cleaned[:-1]
        return cleaned

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        question = self._extract_question(prompt)
        query_terms = set(self._tokenize(question))
        docs = self._extract_documents(prompt)
        candidates: list[tuple[int, str]] = []
        seen = set()

        for doc in docs:
            for sent in self._split_sentences(doc):
                normalized = sent.lower().strip()
                if normalized in seen or len(normalized) < 25:
                    continue
                seen.add(normalized)
                sent_terms = set(self._tokenize(sent))
                overlap = len(query_terms.intersection(sent_terms))
                score = overlap * 10 + min(len(sent_terms), 20)
                candidates.append((score, sent))

        if not candidates:
            return "The provided documents do not contain sufficient information."

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:3]
        lines = [f"Evidence {i}: {self._clean_sentence(sent)}." for i, (_, sent) in enumerate(selected, 1)]
        return " ".join(lines)[:1200].strip()


class PipelineService:
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "have", "has", "had", "but", "not", "you", "your", "about", "into", "than",
        "their", "there", "which", "what", "when", "where", "how", "does", "did",
        "can", "could", "would", "should", "may", "might", "been", "being", "also",
        "question", "effect", "effective", "help", "helps",
    }

    def __init__(self) -> None:
        t_start = time.perf_counter()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("PipelineService")

        self.retriever = Retriever(top_k=5, index_path="./data/index")
        self.extractor = ClaimExtractor()
        self.verifier = ClaimVerifier(model_name="facebook/bart-large-mnli", device=-1)
        self.generator = self._build_generator()
        self.min_index_docs = int(os.getenv("RAG_MIN_INDEX_DOCS", "500"))
        self.relevance_distance_threshold = float(os.getenv("RAG_RELEVANCE_MAX_DISTANCE", "1.18"))
        self.intent_coverage_threshold = float(os.getenv("RAG_INTENT_MIN_COVERAGE", "0.4"))
        self.rebuild_samples = int(os.getenv("RAG_REBUILD_SAMPLES", "1000"))
        self._ensure_index()
        self._warmup()
        self.logger.info(f"PipelineService ready in {time.perf_counter() - t_start:.1f}s")

    def _build_generator(self) -> RAGGenerator:
        try:
            return RAGGenerator(
                backend_type="huggingface",
                model_name="mistralai/Mistral-7B-Instruct-v0.1",
                device="cuda",
            )
        except Exception:
            return RAGGenerator(llm_backend=EvidenceAwareFallbackBackend())

    def _ensure_index(self) -> None:
        index_file = Path("./data/index/index.faiss")
        docs_file = Path("./data/index/documents.json")

        if index_file.exists() and docs_file.exists():
            self.retriever.load()
            if self.retriever.total_documents >= self.min_index_docs:
                return

        self.retriever.index_from_qa_dataset(
            dataset_name="pubmed_qa",
            split="train",
            max_samples=self.rebuild_samples,
        )
        self.retriever.save()

    def _warmup(self) -> None:
        """Pre-encode a dummy query to warm up the embedding model."""
        try:
            self.retriever.db.encode(["warmup"])
            self.logger.info("Embedding model warmed up.")
        except Exception:
            pass

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2]

    def _query_keywords(self, question: str) -> list[str]:
        tokens = self._tokenize(question)
        return [t for t in tokens if t not in self.STOPWORDS]

    def _intent_coverage(self, question: str, retrieved_docs: list[dict]) -> tuple[float, list[str], list[str]]:
        keywords = self._query_keywords(question)
        if not keywords:
            return 1.0, [], []

        corpus = " ".join(d.get("text", "") for d in retrieved_docs).lower()
        matched = [k for k in keywords if k in corpus]
        coverage = len(matched) / len(keywords)
        missing = [k for k in keywords if k not in matched]
        return coverage, matched, missing

    def ask(self, question: str, top_k: int) -> dict[str, Any]:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()

        retrieved = self.retriever.retrieve(question, top_k=top_k)
        timings["retrieval_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        docs_for_generation = [r["document"] for r in retrieved]

        if not retrieved:
            return {
                "question": question,
                "generated_answer": "The provided documents do not contain sufficient information for this question.",
                "final_verified_answer": "The provided documents do not contain sufficient information for this question.",
                "claims": [],
                "metrics": {
                    "num_claims": 0,
                    "num_supported": 0,
                    "num_contradictions": 0,
                    "num_neutral": 0,
                    "fact_score": 0.0,
                    "hallucination_rate": 0.0,
                },
                "retrieval_guard": {
                    "triggered": True,
                    "reason": "no_retrieval_results",
                },
                "retrieved_documents": [],
                "timings": timings,
            }

        top_distance = float(retrieved[0]["score"])
        coverage, matched_terms, missing_terms = self._intent_coverage(question, docs_for_generation)

        guard_triggered = (
            top_distance > self.relevance_distance_threshold
            or coverage < self.intent_coverage_threshold
        )

        if guard_triggered:
            return {
                "question": question,
                "generated_answer": "The provided documents do not contain sufficient information for this question.",
                "final_verified_answer": "The provided documents do not contain sufficient information for this question.",
                "claims": [],
                "metrics": {
                    "num_claims": 0,
                    "num_supported": 0,
                    "num_contradictions": 0,
                    "num_neutral": 0,
                    "fact_score": 0.0,
                    "hallucination_rate": 0.0,
                },
                "retrieval_guard": {
                    "triggered": True,
                    "reason": "weak_retrieval_or_intent_mismatch",
                    "top_distance": round(top_distance, 4),
                    "distance_threshold": self.relevance_distance_threshold,
                    "intent_coverage": round(coverage, 3),
                    "intent_threshold": self.intent_coverage_threshold,
                    "matched_terms": matched_terms,
                    "missing_terms": missing_terms,
                },
                "retrieved_documents": [
                    {
                        "rank": r["rank"],
                        "score": r["score"],
                        "source": r["document"].get("source", "unknown"),
                        "text": r["document"].get("text", ""),
                    }
                    for r in retrieved
                ],
                "timings": timings,
            }

        t1 = time.perf_counter()
        result = self.generator.generate_answer(
            question=question,
            retrieved_docs=docs_for_generation,
            max_tokens=256,
            temperature=0.5,
        )
        timings["generation_ms"] = round((time.perf_counter() - t1) * 1000, 1)

        t2 = time.perf_counter()
        claims = self.extractor.extract_claims_with_metadata(result["answer"])
        timings["claim_extraction_ms"] = round((time.perf_counter() - t2) * 1000, 1)

        t3 = time.perf_counter()
        verification = self.verifier.run_verification(
            generated_answer=result["answer"],
            extracted_claims=claims,
            retrieved_documents=docs_for_generation,
            baseline_hallucination_rate=None,
        )
        timings["verification_ms"] = round((time.perf_counter() - t3) * 1000, 1)
        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "question": question,
            "generated_answer": result["answer"],
            "final_verified_answer": verification["final_verified_answer"],
            "claims": verification["verified_claims"],
            "metrics": {
                **verification["metrics"],
                "verification_mode": "heuristic" if self.verifier._use_heuristic else "nli_model",
            },
            "retrieval_guard": {
                "triggered": False,
                "top_distance": round(top_distance, 4),
                "intent_coverage": round(coverage, 3),
                "matched_terms": matched_terms,
                "missing_terms": missing_terms,
            },
            "retrieved_documents": [
                {
                    "rank": r["rank"],
                    "score": r["score"],
                    "source": r["document"].get("source", "unknown"),
                    "text": r["document"].get("text", ""),
                }
                for r in retrieved
            ],
            "timings": timings,
        }


app = FastAPI(title="RAG Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = PipelineService()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    try:
        return service.ask(req.question, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
