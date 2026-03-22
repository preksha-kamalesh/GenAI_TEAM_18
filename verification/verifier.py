"""
Verification Module
Claim-level verification and surgical correction for hallucination reduction.

Pipeline:
Generated Answer + Atomic Claims + Retrieved Documents
    -> NLI Verification
    -> Contradiction Detection
    -> Surgical Correction
    -> Final Verified Answer + Metrics
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union


class ClaimVerifier:
    """Verify factual claims against retrieved evidence using an NLI model."""

    DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
    LABEL_ENTAILMENT = "Entailment"
    LABEL_CONTRADICTION = "Contradiction"
    LABEL_NEUTRAL = "Neutral"

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, device: int = -1):
        """
        Args:
            model_name: HuggingFace NLI model ID.
            device: Pipeline device (-1=CPU, 0+=GPU index).
        """
        self.model_name = model_name
        self.device = device
        self._use_heuristic = False

        try:
            from transformers import pipeline

            self.nli = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                device=device,
                return_all_scores=True,
            )
        except Exception as exc:
            print(f"Warning: could not load NLI model '{model_name}': {exc}")
            print("         Falling back to lexical-overlap verification.")
            self.nli = None
            self._use_heuristic = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_claims(
        self,
        claims: List[Union[str, Dict]],
        retrieved_documents: List[Dict],
    ) -> List[Dict]:
        """Verify each claim against retrieved documents."""
        verified = []
        for idx, claim_item in enumerate(claims):
            claim_dict = self._normalize_claim_item(claim_item, idx)
            claim_text = claim_dict["claim"]
            claim_result = self.verify_single_claim(claim_text, retrieved_documents)

            verified.append(
                {
                    **claim_dict,
                    "label": claim_result["label"],
                    "confidence": claim_result["confidence"],
                    "evidence_doc_index": claim_result["evidence_doc_index"],
                    "evidence_text": claim_result["evidence_text"],
                    "scores": claim_result["scores"],
                }
            )
        return verified

    def verify_single_claim(self, claim: str, retrieved_documents: List[Dict]) -> Dict:
        """Verify a single claim and return best evidence with an NLI label."""
        docs = self._normalize_documents(retrieved_documents)
        if not docs:
            return {
                "label": self.LABEL_NEUTRAL,
                "confidence": 0.0,
                "evidence_doc_index": None,
                "evidence_text": "",
                "scores": {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0},
            }

        best_by_label = {
            "entailment": (0.0, None, ""),
            "contradiction": (0.0, None, ""),
            "neutral": (0.0, None, ""),
        }

        for doc_index, doc_text in enumerate(docs):
            if not doc_text.strip():
                continue

            scores = self._compute_nli_scores(doc_text, claim)
            for label_key in ("entailment", "contradiction", "neutral"):
                if scores[label_key] > best_by_label[label_key][0]:
                    best_by_label[label_key] = (scores[label_key], doc_index, doc_text)

        best_ent, ent_idx, ent_text = best_by_label["entailment"]
        best_con, con_idx, con_text = best_by_label["contradiction"]
        best_neu, neu_idx, neu_text = best_by_label["neutral"]

        # Conservative decision rule to reduce false contradiction spikes.
        if best_ent >= 0.50 and best_ent >= (best_con + 0.05):
            return {
                "label": self.LABEL_ENTAILMENT,
                "confidence": best_ent,
                "evidence_doc_index": ent_idx,
                "evidence_text": ent_text,
                "scores": {
                    "entailment": best_ent,
                    "contradiction": best_con,
                    "neutral": best_neu,
                },
            }
        if best_con >= 0.50 and best_con > best_ent:
            return {
                "label": self.LABEL_CONTRADICTION,
                "confidence": best_con,
                "evidence_doc_index": con_idx,
                "evidence_text": con_text,
                "scores": {
                    "entailment": best_ent,
                    "contradiction": best_con,
                    "neutral": best_neu,
                },
            }
        return {
            "label": self.LABEL_NEUTRAL,
            "confidence": best_neu,
            "evidence_doc_index": neu_idx,
            "evidence_text": neu_text,
            "scores": {
                "entailment": best_ent,
                "contradiction": best_con,
                "neutral": best_neu,
            },
        }

    def run_verification(
        self,
        generated_answer: str,
        extracted_claims: List[Union[str, Dict]],
        retrieved_documents: List[Dict],
        baseline_hallucination_rate: Optional[float] = None,
    ) -> Dict:
        """
        End-to-end verification:
        1) verify claims
        2) compute metrics
        3) surgically correct contradicted claims
        """
        verified_claims = self.verify_claims(extracted_claims, retrieved_documents)
        metrics = self.compute_metrics(verified_claims, baseline_hallucination_rate)
        final_answer = self.surgically_correct_answer(generated_answer, verified_claims)

        return {
            "generated_answer": generated_answer,
            "verified_claims": verified_claims,
            "metrics": metrics,
            "final_verified_answer": final_answer,
        }

    def compute_metrics(
        self,
        verified_claims: List[Dict],
        baseline_hallucination_rate: Optional[float] = None,
    ) -> Dict:
        """Compute FactScore and Hallucination Rate."""
        total = len(verified_claims)
        if total == 0:
            return {
                "num_claims": 0,
                "num_supported": 0,
                "num_contradictions": 0,
                "num_neutral": 0,
                "fact_score": 0.0,
                "hallucination_rate": 0.0,
                "target_fact_score_met": False,
                "contradiction_reduction_vs_baseline": None,
                "target_contradiction_reduction_met": False,
            }

        supported = sum(1 for c in verified_claims if c["label"] == self.LABEL_ENTAILMENT)
        contradictions = sum(1 for c in verified_claims if c["label"] == self.LABEL_CONTRADICTION)
        neutral = total - supported - contradictions

        fact_score = (supported / total) * 100.0
        hallucination_rate = (contradictions / total) * 100.0

        reduction = None
        target_reduction_met = False
        if baseline_hallucination_rate is not None and baseline_hallucination_rate > 0:
            reduction = (
                (baseline_hallucination_rate - hallucination_rate) / baseline_hallucination_rate
            ) * 100.0
            target_reduction_met = reduction >= 15.0

        return {
            "num_claims": total,
            "num_supported": supported,
            "num_contradictions": contradictions,
            "num_neutral": neutral,
            "fact_score": round(fact_score, 2),
            "hallucination_rate": round(hallucination_rate, 2),
            "target_fact_score_met": fact_score > 80.0,
            "contradiction_reduction_vs_baseline": None if reduction is None else round(reduction, 2),
            "target_contradiction_reduction_met": target_reduction_met,
        }

    def surgically_correct_answer(self, generated_answer: str, verified_claims: List[Dict]) -> str:
        """
        Remove only contradicted content while keeping supported content and fluency.

        Strategy:
        - Remove source sentences that contain contradicted claims.
        - Fallback: remove sentences matching contradicted claim text.
        - If all sentences are removed, rebuild from supported claims.
        """
        contradictions = [
            c for c in verified_claims if c.get("label") == self.LABEL_CONTRADICTION
        ]
        if not contradictions:
            return generated_answer.strip()

        contradicted_source_sents = set()
        contradicted_claims = []
        for c in contradictions:
            claim_text = c.get("claim", "").strip()
            source_sentence = c.get("source_sentence", "").strip()
            if claim_text:
                contradicted_claims.append(claim_text)
            if source_sentence:
                contradicted_source_sents.add(self._normalize_text(source_sentence))

        sentences = self._split_sentences(generated_answer)
        kept = []
        for sent in sentences:
            normalized_sent = self._normalize_text(sent)
            if normalized_sent in contradicted_source_sents:
                continue

            if self._sentence_matches_any_claim(normalized_sent, contradicted_claims):
                continue

            kept.append(sent.strip())

        if kept:
            return self._join_sentences(kept)

        supported_claims = [
            c["claim"].strip()
            for c in verified_claims
            if c.get("label") == self.LABEL_ENTAILMENT and c.get("claim", "").strip()
        ]
        if supported_claims:
            return self._join_sentences(supported_claims)

        return "The provided documents do not contain sufficient information to provide a verified answer."

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        if self._use_heuristic:
            return self._heuristic_scores(premise, hypothesis)

        # BART-MNLI style input: premise </s></s> hypothesis
        model_input = f"{premise} </s></s> {hypothesis}"
        outputs = self.nli(model_input)[0]

        scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for item in outputs:
            label = str(item["label"]).lower()
            score = float(item["score"])
            if "entail" in label:
                scores["entailment"] = score
            elif "contrad" in label:
                scores["contradiction"] = score
            elif "neutral" in label:
                scores["neutral"] = score
        return scores

    def _heuristic_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Fallback when NLI model is unavailable."""
        p_tokens = set(self._tokenize(premise))
        h_tokens = set(self._tokenize(hypothesis))

        if not h_tokens:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

        overlap = len(p_tokens.intersection(h_tokens)) / max(1, len(h_tokens))
        contradiction_cues = {"not", "no", "never", "none", "without"}
        has_neg_mismatch = any(w in contradiction_cues for w in p_tokens) ^ any(
            w in contradiction_cues for w in h_tokens
        )

        entailment = min(0.95, 0.15 + overlap)
        contradiction = 0.65 if has_neg_mismatch and overlap > 0.4 else max(0.0, 0.35 - overlap)
        neutral = max(0.0, 1.0 - max(entailment, contradiction))
        return {
            "entailment": round(entailment, 4),
            "contradiction": round(contradiction, 4),
            "neutral": round(neutral, 4),
        }

    def _normalize_documents(self, retrieved_documents: List[Dict]) -> List[str]:
        texts = []
        for item in retrieved_documents:
            if not isinstance(item, dict):
                continue
            if "document" in item and isinstance(item["document"], dict):
                text = item["document"].get("text", "")
            else:
                text = item.get("text", "")
            if text:
                texts.append(text)
        return texts

    def _normalize_claim_item(self, claim_item: Union[str, Dict], fallback_id: int) -> Dict:
        if isinstance(claim_item, str):
            return {"id": fallback_id, "claim": claim_item}

        if isinstance(claim_item, dict):
            claim_text = claim_item.get("claim") or claim_item.get("text") or ""
            return {
                "id": claim_item.get("id", fallback_id),
                "claim": claim_text,
                "source_sentence": claim_item.get("source_sentence", ""),
                "sentence_position": claim_item.get("sentence_position"),
            }
        return {"id": fallback_id, "claim": str(claim_item)}

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _join_sentences(self, sentences: List[str]) -> str:
        fixed = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            if s[-1] not in ".!?":
                s = f"{s}."
            fixed.append(s)
        return " ".join(fixed)

    def _sentence_matches_any_claim(self, normalized_sentence: str, claims: List[str]) -> bool:
        for claim in claims:
            normalized_claim = self._normalize_text(claim)
            if not normalized_claim:
                continue
            if normalized_claim in normalized_sentence:
                return True

            claim_tokens = set(self._tokenize(normalized_claim))
            sent_tokens = set(self._tokenize(normalized_sentence))
            if not claim_tokens:
                continue
            overlap = len(claim_tokens.intersection(sent_tokens)) / len(claim_tokens)
            if overlap >= 0.8:
                return True
        return False

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]

