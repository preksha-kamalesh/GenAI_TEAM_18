# Pipeline Run Guide

This guide shows how to run the full project pipeline:

`Retrieve -> Generate -> Extract -> Verify -> Correct`

## 1. Setup

Open PowerShell in:

`C:\Users\Pranav\Desktop\JackFruits\GenAI_TEAM_18`

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional (better claim splitting):

```powershell
python -m spacy download en_core_web_sm
```

## 2. Run Full End-to-End Pipeline

```powershell
python end_to_end_example.py
```

What it does:
- Indexes sample docs with `rag_module`
- Retrieves top documents
- Generates an answer with `generation` (or mock fallback if model is unavailable)
- Extracts atomic claims
- Verifies each claim with `verification.ClaimVerifier`
- Produces a `final_verified_answer` with contradicted claims removed
- Prints `FactScore` and `Hallucination Rate`

## 3. Run Module Tests

RAG module:

```powershell
python test_rag_module.py
```

Generation module:

```powershell
python test_generation.py
```

Verification module (deterministic, no model download required):

```powershell
python test_verification.py
```

Optional combined run:

```powershell
python test_rag_module.py; python test_generation.py; python test_verification.py
```

## 4. NLI Model Notes

- Default verifier model: `facebook/bart-large-mnli`
- If model loading fails (missing deps/network/GPU constraints), verifier auto-falls back to a lexical heuristic so pipeline still runs.
- For best verification quality, keep `transformers` + `torch` installed and allow model download once.

## 5. How to Plug in Your Own Data

In [end_to_end_example.py](C:\Users\Pranav\Desktop\JackFruits\GenAI_TEAM_18\end_to_end_example.py):
- Replace `sample_corpus` with your actual corpus
- Replace `user_question` with your evaluation queries
- Optionally pass your baseline contradiction rate into:
  - `baseline_hallucination_rate=<your %>`

Then rerun:

```powershell
python end_to_end_example.py
```

