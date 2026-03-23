"""
End-to-end example: RAG Pipeline (Retrieval -> Generation -> Claim Extraction -> Verification -> Correction)
Using Preksha's retriever + Navya's generation + Pranav's verification

Run: python end_to_end_example.py
"""

from rag_module import Retriever
from generation import RAGGenerator, ClaimExtractor
from verification import ClaimVerifier


def end_to_end_example():
    """Complete pipeline example."""

    print("=" * 80)
    print("END-TO-END RAG + GENERATION + VERIFICATION PIPELINE")
    print("=" * 80)

    # Step 1: Setup Retriever (Preksha's module)
    print("\n[Step 1] RETRIEVAL (Preksha's Module)")
    print("-" * 80)

    retriever = Retriever(top_k=5)

    # Index sample documents
    sample_corpus = [
        {
            "text": "Diabetes mellitus is a group of metabolic diseases characterized by high blood sugar levels.",
            "source": "medical_wiki"
        },
        {
            "text": "Type 1 diabetes must be managed with insulin injections. It is an autoimmune condition.",
            "source": "medical_wiki"
        },
        {
            "text": "Type 2 diabetes may be treated with oral antidiabetic medications and lifestyle changes.",
            "source": "medical_wiki"
        },
        {
            "text": "Insulin is a hormone produced by the pancreas that regulates blood glucose levels.",
            "source": "medical_wiki"
        },
        {
            "text": "Complications of untreated diabetes include cardiovascular disease, stroke, and kidney damage.",
            "source": "medical_wiki"
        },
    ]

    print("Indexing sample corpus...")
    retriever.index_documents(sample_corpus)
    print(f"Indexed {retriever.total_documents} documents")

    # Step 2: Retrieve relevant documents
    user_question = "What is diabetes and how is it managed?"
    print(f"\nUser Question: {user_question}")

    retrieved_docs = retriever.retrieve(user_question, top_k=3)
    print(f"Retrieved {len(retrieved_docs)} relevant documents:")
    for r in retrieved_docs:
        print(f"   [{r['rank']}] (score={r['score']:.4f}) {r['document']['text'][:80]}...")

    # Extract just the doc dicts for the generator
    docs_for_generation = [r["document"] for r in retrieved_docs]

    # Step 2: Generate Answer (Navya's module)
    print("\n[Step 2] GENERATION (Navya's Module)")
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
        print("Using mock backend for demonstration")

        # Create a mock backend for demo
        class MockBackend:
            def generate(self, prompt, max_tokens=512, temperature=0.7):
                # Return a realistic mock response based on the retrieved docs
                return """Diabetes mellitus is a group of metabolic diseases characterized by high blood sugar levels.

Type 1 diabetes is an autoimmune condition that requires insulin injections to manage. In contrast, Type 2 diabetes can often be treated with oral medications and lifestyle modifications including exercise and diet changes.

Insulin, a hormone produced by the pancreas, plays a crucial role in regulating blood glucose levels. When left untreated, diabetes can lead to serious complications such as cardiovascular disease, stroke, and kidney damage."""

        generator = RAGGenerator(llm_backend=MockBackend())
        print("Using mock backend for demonstration")

    print("\nGenerating answer...")
    result = generator.generate_answer(
        question=user_question,
        retrieved_docs=docs_for_generation,
        max_tokens=256,
        temperature=0.5
    )

    print("\nGenerated Answer:")
    print(result['answer'])

    # Step 3: Extract Claims (Navya's module - part 2)
    print("\n[Step 3] CLAIM EXTRACTION (Navya's Module)")
    print("-" * 80)

    extractor = ClaimExtractor()
    claims = extractor.extract_claims(result['answer'])

    print(f"Extracted {len(claims)} atomic claims:")
    for i, claim in enumerate(claims, 1):
        print(f"   {i:2d}. {claim}")

    # Extract claims with metadata for verification
    claims_with_meta = extractor.extract_claims_with_metadata(result['answer'])

    # Step 4: Verify Claims + Surgical Correction (Pranav's module)
    print("\n[Step 4] VERIFICATION + SURGICAL CORRECTION (Pranav's Module)")
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
    result = end_to_end_example()

    print("\nSummary Statistics:")
    print(f"   - Question: {result['question'][:60]}...")
    print(f"   - Generated answer length: {len(result['generated_answer'])} chars")
    print(f"   - Retrieved documents: {len(result['retrieved_documents'])}")
    print(f"   - Claims extracted: {result['num_claims']}")
    print(f"   - Final verified answer length: {len(result['final_verified_answer'])} chars")

    metrics = result["verification"]["metrics"]
    print(f"   - FactScore: {metrics['fact_score']}%")
    print(f"   - Hallucination Rate: {metrics['hallucination_rate']}%")
