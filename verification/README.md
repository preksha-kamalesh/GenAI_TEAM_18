# Verification Module - Pranav's Part
## Claim-Level Self-Verification + Surgical Correction

This module is the final stage of the pipeline:

```
Retrieve -> Generate -> Extract Claims -> Verify Claims -> Surgical Correction
```

It verifies each atomic claim against retrieved evidence using an NLI model (`facebook/bart-large-mnli`) and produces:
- Per-claim labels: `Entailment`, `Contradiction`, `Neutral`
- Metrics: FactScore and Hallucination Rate
- `final_verified_answer` with contradicted claims removed

## Files

- `verification/verifier.py` - Main `ClaimVerifier` implementation
- `verification/__init__.py` - Package exports
- `verification/README.md` - This document

## Usage

```python
from verification import ClaimVerifier

verifier = ClaimVerifier(model_name="facebook/bart-large-mnli", device=-1)

verification_output = verifier.run_verification(
    generated_answer=generated_answer,
    extracted_claims=claims_with_metadata,   # list[str] or list[dict]
    retrieved_documents=retrieved_docs,      # list[dict] with 'text' (or retrieval wrappers)
    baseline_hallucination_rate=30.0,        # optional (%)
)

print(verification_output["metrics"])
print(verification_output["final_verified_answer"])
```

## Metrics

- **FactScore** = (supported claims / total claims) * 100
- **Hallucination Rate** = (contradicted claims / total claims) * 100

Targets tracked in output:
- `target_fact_score_met`: `FactScore > 80`
- `target_contradiction_reduction_met`: `>= 15%` reduction vs baseline (if provided)

## Notes

- If the NLI model cannot be loaded, the verifier falls back to a lexical-overlap heuristic so the pipeline remains runnable.
- Surgical correction removes only contradicted claim content (sentence-level matching), preserving supported content when possible.

