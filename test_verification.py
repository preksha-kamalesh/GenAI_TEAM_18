"""
Test script for Pranav's Verification Module.
Run: python test_verification.py
"""

from verification import ClaimVerifier


class MockClaimVerifier(ClaimVerifier):
    """
    Deterministic verifier for tests.
    Avoids heavy model downloads by overriding NLI scoring.
    """

    def __init__(self):
        self.model_name = "mock-nli"
        self.device = -1
        self._use_heuristic = False
        self.nli = None

    def _compute_nli_scores(self, premise: str, hypothesis: str):
        p = premise.lower()
        h = hypothesis.lower()

        # Strong contradiction case
        if "type 1 diabetes can be cured" in h:
            return {"entailment": 0.05, "contradiction": 0.92, "neutral": 0.03}

        # Supported cases
        if "diabetes is a metabolic disease" in h and "metabolic diseases" in p:
            return {"entailment": 0.91, "contradiction": 0.03, "neutral": 0.06}
        if "type 1 diabetes requires insulin" in h and "insulin injections" in p:
            return {"entailment": 0.89, "contradiction": 0.04, "neutral": 0.07}

        # Default neutral
        return {"entailment": 0.18, "contradiction": 0.14, "neutral": 0.68}


def test_claim_labeling():
    print("=" * 70)
    print("1. Testing Claim Labeling (Entailment/Contradiction/Neutral)")
    print("=" * 70)

    verifier = MockClaimVerifier()

    docs = [
        {
            "text": "Diabetes mellitus is a group of metabolic diseases characterized by high blood sugar levels.",
            "source": "medical_wiki",
        },
        {
            "text": "Type 1 diabetes must be managed with insulin injections.",
            "source": "medical_wiki",
        },
    ]

    claims = [
        {"id": 0, "claim": "diabetes is a metabolic disease"},
        {"id": 1, "claim": "type 1 diabetes requires insulin"},
        {"id": 2, "claim": "type 1 diabetes can be cured"},
        {"id": 3, "claim": "exercise is always sufficient for all diabetes cases"},
    ]

    verified = verifier.verify_claims(claims, docs)
    labels = [v["label"] for v in verified]

    print("   Labels:", labels)
    assert labels[0] == "Entailment"
    assert labels[1] == "Entailment"
    assert labels[2] == "Contradiction"
    assert labels[3] == "Neutral"
    print("   PASS: claim labels are correct")
    print()


def test_metrics_and_targets():
    print("=" * 70)
    print("2. Testing Metrics (FactScore + Hallucination Rate)")
    print("=" * 70)

    verifier = MockClaimVerifier()

    verified_claims = [
        {"label": "Entailment"},
        {"label": "Entailment"},
        {"label": "Contradiction"},
        {"label": "Neutral"},
    ]

    metrics = verifier.compute_metrics(
        verified_claims=verified_claims,
        baseline_hallucination_rate=40.0,  # baseline contradictions in %
    )

    print("   Metrics:", metrics)
    assert metrics["num_claims"] == 4
    assert metrics["fact_score"] == 50.0
    assert metrics["hallucination_rate"] == 25.0
    assert metrics["contradiction_reduction_vs_baseline"] == 37.5
    assert metrics["target_contradiction_reduction_met"] is True
    assert metrics["target_fact_score_met"] is False
    print("   PASS: metrics and target flags are correct")
    print()


def test_surgical_correction():
    print("=" * 70)
    print("3. Testing Surgical Correction")
    print("=" * 70)

    verifier = MockClaimVerifier()

    generated_answer = (
        "Diabetes is a metabolic disease. "
        "Type 1 diabetes requires insulin. "
        "Type 1 diabetes can be cured. "
        "Patients should follow medical advice."
    )

    verified_claims = [
        {
            "claim": "diabetes is a metabolic disease",
            "label": "Entailment",
            "source_sentence": "Diabetes is a metabolic disease.",
        },
        {
            "claim": "type 1 diabetes requires insulin",
            "label": "Entailment",
            "source_sentence": "Type 1 diabetes requires insulin.",
        },
        {
            "claim": "type 1 diabetes can be cured",
            "label": "Contradiction",
            "source_sentence": "Type 1 diabetes can be cured.",
        },
        {
            "claim": "patients should follow medical advice",
            "label": "Neutral",
            "source_sentence": "Patients should follow medical advice.",
        },
    ]

    corrected = verifier.surgically_correct_answer(generated_answer, verified_claims)
    print("   Corrected answer:", corrected)

    assert "can be cured" not in corrected.lower()
    assert "metabolic disease" in corrected.lower()
    assert "requires insulin" in corrected.lower()
    print("   PASS: contradicted claim removed, supported claims preserved")
    print()


def test_integration_run_verification():
    print("=" * 70)
    print("4. Testing run_verification Integration")
    print("=" * 70)

    verifier = MockClaimVerifier()
    answer = (
        "Diabetes is a metabolic disease. "
        "Type 1 diabetes requires insulin. "
        "Type 1 diabetes can be cured."
    )
    claims = [
        {
            "id": 0,
            "claim": "diabetes is a metabolic disease",
            "source_sentence": "Diabetes is a metabolic disease.",
        },
        {
            "id": 1,
            "claim": "type 1 diabetes requires insulin",
            "source_sentence": "Type 1 diabetes requires insulin.",
        },
        {
            "id": 2,
            "claim": "type 1 diabetes can be cured",
            "source_sentence": "Type 1 diabetes can be cured.",
        },
    ]
    docs = [
        {"text": "Diabetes mellitus is a group of metabolic diseases.", "source": "wiki"},
        {"text": "Type 1 diabetes must be managed with insulin injections.", "source": "wiki"},
    ]

    output = verifier.run_verification(
        generated_answer=answer,
        extracted_claims=claims,
        retrieved_documents=docs,
        baseline_hallucination_rate=50.0,
    )

    assert "verified_claims" in output
    assert "metrics" in output
    assert "final_verified_answer" in output
    assert len(output["verified_claims"]) == 3
    assert "can be cured" not in output["final_verified_answer"].lower()
    print("   PASS: end-to-end verification output structure is valid")
    print()


if __name__ == "__main__":
    test_claim_labeling()
    test_metrics_and_targets()
    test_surgical_correction()
    test_integration_run_verification()

    print("=" * 70)
    print("ALL VERIFICATION TESTS PASSED")
    print("=" * 70)
