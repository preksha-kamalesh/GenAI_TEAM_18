"""
End-to-end example: RAG Pipeline (Retrieval -> Generation -> Claim Extraction -> Verification -> Correction)

Run: python end_to_end_example.py
"""

import argparse
import os
import re
from typing import Optional

from rag_module import Retriever
from generation import RAGGenerator, ClaimExtractor
from verification import ClaimVerifier


def end_to_end_example(
    dataset_name: str = "pubmed_qa",
    max_samples: int = 200,
    index_path: str = "./data/index",
    user_question: Optional[str] = None,
):
    """Complete pipeline example."""

    print("=" * 80)
    print("END-TO-END RAG + GENERATION + VERIFICATION PIPELINE")
    print("=" * 80)

    # Step 1: Setup retriever
    print("\n[Step 1] RETRIEVAL")
    print("-" * 80)

    retriever = Retriever(top_k=5, index_path=index_path)

    index_file = os.path.join(index_path, "index.faiss")
    docs_file = os.path.join(index_path, "documents.json")

    if os.path.exists(index_file) and os.path.exists(docs_file):
        print(f"Loading existing real-data index from {index_path}...")
        retriever.load()
    else:
        print(
            f"No index found at {index_path}. Building from real dataset "
            f"'{dataset_name}' (max_samples={max_samples})..."
        )
        retriever.index_from_qa_dataset(
            dataset_name=dataset_name,
            split="train",
            max_samples=max_samples,
        )
        retriever.save()
        print(f"Saved real-data index to {index_path}")
    print(f"Indexed {retriever.total_documents} real dataset documents")

    # Step 2: Retrieve relevant documents
    if not user_question:
        default_questions = {
            "pubmed_qa": "What are common approaches to manage type 2 diabetes?",
            "trivia_qa": "Who wrote Romeo and Juliet?",
            "natural_questions": "What is the capital of France?",
        }
        user_question = default_questions.get(dataset_name, default_questions["pubmed_qa"])

    print(f"\nUser Question: {user_question}")

    retrieved_docs = retriever.retrieve(user_question, top_k=3)
    print(f"Retrieved {len(retrieved_docs)} relevant documents:")
    for r in retrieved_docs:
        print(f"   [{r['rank']}] (score={r['score']:.4f}) {r['document']['text'][:80]}...")

    # Extract just the doc dicts for the generator
    docs_for_generation = [r["document"] for r in retrieved_docs]

    # Step 2: Generate answer
    print("\n[Step 2] GENERATION")
    print("-" * 80)

    print("Initializing RAG Generator...")

    try:
        # Try to use a real LLM backend
        generator = RAGGenerator(
            backend_type="huggingface",
            model_name="mistralai/Mistral-7B-Instruct-v0.1",
            device="cuda"  # Change to "cpu" if cuda unavailable
        )
        print("Using Mistral-7B backend")
    except Exception as e:
        print(f"Could not load Mistral: {e}")
        print("Using evidence-aware fallback backend")

        class EvidenceAwareMockBackend:
            _STOPWORDS = {
                "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
                "have", "has", "had", "but", "not", "you", "your", "about", "into", "than",
                "their", "there", "which", "what", "when", "where", "how", "does", "did",
                "can", "could", "would", "should", "may", "might", "been", "being", "also",
            }

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
            def _extract_question(prompt: str) -> str:
                match = re.search(r"\nQuestion:\n(.*?)\n\nInstructions:", prompt, flags=re.DOTALL)
                return match.group(1).strip() if match else ""

            @staticmethod
            def _split_sentences(text: str) -> list[str]:
                return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

            def _keyword_summary(self, question: str, docs: list[str]) -> str:
                counts: dict[str, int] = {}
                q_terms = set(self._tokenize(question))
                for doc in docs:
                    for tok in self._tokenize(doc):
                        if tok in self._STOPWORDS:
                            continue
                        counts[tok] = counts.get(tok, 0) + (3 if tok in q_terms else 1)

                if not counts:
                    return ""

                top_terms = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
                terms = [t for t, _ in top_terms]
                return ", ".join(terms)

            @staticmethod
            def _clean_sentence(sent: str) -> str:
                cleaned = re.sub(r"\s+", " ", sent).strip()
                cleaned = re.sub(r"^[Qq]uestion\s*[:\-]\s*", "", cleaned)
                cleaned = re.sub(r"^[Ss]ource\s*\d+\s*[:\-]\s*", "", cleaned)
                if cleaned.endswith("?"):
                    cleaned = cleaned[:-1]
                return cleaned

            def generate(self, prompt, max_tokens=512, temperature=0.7):
                question = self._extract_question(prompt)
                query_terms = set(self._tokenize(question))
                docs = self._extract_documents(prompt)

                candidates = []
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
                        candidates.append((score, sent, doc))

                if not candidates:
                    return "The provided documents do not contain sufficient information."

                candidates.sort(key=lambda x: x[0], reverse=True)
                selected = candidates[:3]
                details = []
                for i, (_, sent, _) in enumerate(selected, 1):
                    cleaned = self._clean_sentence(sent)
                    if not cleaned:
                        continue
                    details.append(f"Evidence {i}: {cleaned}.")

                answer_parts = details

                if not answer_parts:
                    return "The provided documents do not contain sufficient information."
                return " ".join(answer_parts)[:1200].strip()

        generator = RAGGenerator(llm_backend=EvidenceAwareMockBackend())
        print("Using evidence-aware fallback backend")

    print("\nGenerating answer...")
    result = generator.generate_answer(
        question=user_question,
        retrieved_docs=docs_for_generation,
        max_tokens=256,
        temperature=0.5
    )

    print("\nGenerated Answer:")
    print(result['answer'])

    # Step 3: Extract claims
    print("\n[Step 3] CLAIM EXTRACTION")
    print("-" * 80)

    extractor = ClaimExtractor()
    claims = extractor.extract_claims(result['answer'])

    print(f"Extracted {len(claims)} atomic claims:")
    for i, claim in enumerate(claims, 1):
        print(f"   {i:2d}. {claim}")

    # Extract claims with metadata for verification
    claims_with_meta = extractor.extract_claims_with_metadata(result['answer'])

    # Step 4: Verify claims + surgical correction
    print("\n[Step 4] VERIFICATION + SURGICAL CORRECTION")
    print("-" * 80)

    verifier = ClaimVerifier(model_name="facebook/bart-large-mnli", device=-1)
    verification_output = verifier.run_verification(
        generated_answer=result["answer"],
        extracted_claims=claims_with_meta,
        retrieved_documents=docs_for_generation,
        baseline_hallucination_rate=None,  # set baseline (%) if available
    )

    print(f"Verified {len(verification_output['verified_claims'])} claims")
    print("\nPer-claim verification labels:")
    for claim in verification_output["verified_claims"]:
        print(
            f"   [{claim['label']:<13}] "
            f"(confidence={claim['confidence']:.3f}) {claim['claim']}"
        )

    metrics = verification_output["metrics"]
    print("\nVerification metrics:")
    print(f"   - FactScore: {metrics['fact_score']}%")
    print(f"   - Hallucination Rate: {metrics['hallucination_rate']}%")
    print(f"   - Supported claims: {metrics['num_supported']}/{metrics['num_claims']}")
    print(f"   - Contradictions: {metrics['num_contradictions']}")

    # Step 5: Final verified answer
    print("\n[Step 5] FINAL VERIFIED ANSWER")
    print("-" * 80)
    print(verification_output["final_verified_answer"])

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nThe full pipeline (Retrieve -> Generate -> Extract -> Verify -> Correct) ran successfully.")

    return {
        "question": user_question,
        "generated_answer": result["answer"],
        "retrieved_documents": docs_for_generation,
        "claims": claims_with_meta,
        "num_claims": len(claims_with_meta),
        "verification": verification_output,
        "final_verified_answer": verification_output["final_verified_answer"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end RAG + verification pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pubmed_qa",
        choices=["natural_questions", "trivia_qa", "pubmed_qa"],
        help="Real dataset to build/load index from",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max samples to index when building a new real-data index",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="./data/index",
        help="Directory containing/saving FAISS index files",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Custom user question for retrieval/generation",
    )
    args = parser.parse_args()

    result = end_to_end_example(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        index_path=args.index_path,
        user_question=args.question,
    )

    print("\nSummary Statistics:")
    print(f"   - Question: {result['question'][:60]}...")
    print(f"   - Generated answer length: {len(result['generated_answer'])} chars")
    print(f"   - Retrieved documents: {len(result['retrieved_documents'])}")
    print(f"   - Claims extracted: {result['num_claims']}")
    print(f"   - Final verified answer length: {len(result['final_verified_answer'])} chars")

    metrics = result["verification"]["metrics"]
    print(f"   - FactScore: {metrics['fact_score']}%")
    print(f"   - Hallucination Rate: {metrics['hallucination_rate']}%")
