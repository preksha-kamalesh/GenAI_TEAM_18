"""
Claim Extractor Module — Extract Atomic Factual Claims
Navya's Module | genai-team-18

Takes a generated answer and breaks it into individual atomic factual claims.
Uses sentence-level splitting for claim extraction.

Example:
    Input: "Drug X causes nausea and dizziness. It is highly effective."
    Output: [
        "Drug X causes nausea",
        "Drug X causes dizziness",
        "Drug X is highly effective"
    ]
"""

from typing import List, Dict, Optional, Tuple
import re


class ClaimExtractor:
    """Extract atomic factual claims from generated text."""

    def __init__(self):
        """Initialize the claim extractor."""
        try:
            import spacy
            # Try to load model, with fallback
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️  spaCy model 'en_core_web_sm' not found.")
                print("Install with: python -m spacy download en_core_web_sm")
                # Fallback to simple splitting
                self.nlp = None
        except ImportError:
            print("⚠️  spaCy not installed. Using simple sentence splitting.")
            self.nlp = None
        except Exception as e:
            print(f"⚠️  spaCy initialization failed: {e}")
            print("    Using simple sentence splitting as fallback...")
            self.nlp = None

    def extract_claims(self, text: str, lowercase: bool = True) -> List[str]:
        """Extract atomic factual claims from text.

        Args:
            text: Generated answer text
            lowercase: Convert claims to lowercase

        Returns:
            List of claim strings
        """
        # Remove citations and metadata
        text = self._clean_text(text)

        # Split into sentences
        sentences = self._split_sentences(text)

        # Extract individual claims from each sentence
        claims = []
        for sentence in sentences:
            atomic_claims = self._extract_atomic_claims(sentence)
            claims.extend(atomic_claims)

        # Clean and deduplicate claims
        claims = [c.strip() for c in claims if c and len(c.strip()) > 5]
        if lowercase:
            claims = [c.lower() for c in claims]

        # Remove duplicates while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            if claim not in seen:
                seen.add(claim)
                unique_claims.append(claim)

        return unique_claims

    def extract_claims_with_metadata(self, text: str) -> List[Dict]:
        """Extract claims with metadata (source sentence, position, etc.).

        Args:
            text: Generated answer text

        Returns:
            List of dicts with keys: 'claim', 'source_sentence', 'position'
        """
        text = self._clean_text(text)
        sentences = self._split_sentences(text)

        claims_with_meta = []
        claim_id = 0

        for sent_idx, sentence in enumerate(sentences):
            atomic_claims = self._extract_atomic_claims(sentence)
            for claim in atomic_claims:
                claim = claim.strip()
                if claim and len(claim) > 5:
                    claims_with_meta.append({
                        "id": claim_id,
                        "claim": claim.lower(),
                        "source_sentence": sentence.strip(),
                        "sentence_position": sent_idx,
                    })
                    claim_id += 1

        return claims_with_meta

    def _clean_text(self, text: str) -> str:
        """Remove citations, metadata, and clean text."""
        # Remove [1], [Citation], etc.
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[Citation.*?\]", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy or fallback method."""
        if self.nlp:
            # Use spaCy for robust sentence splitting
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            # Simple regex-based splitting
            return self._simple_sentence_split(text)

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting using regex (fallback)."""
        # Split on periods, question marks, exclamation marks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def _extract_atomic_claims(self, sentence: str) -> List[str]:
        """Split a sentence into atomic claims.

        Handles:
        - Comma-separated facts: "X causes A, B, and C"
        - Coordinated clauses: "X is good and Y is bad"
        - Negations: "X does not cause Y"
        """
        if not sentence.strip():
            return []

        claims = []

        # Handle "X causes A, B, and C" -> ["X causes A", "X causes B", "X causes C"]
        if " causes " in sentence or " is " in sentence or " contains " in sentence:
            claims.extend(self._handle_list_facts(sentence))
        else:
            claims.append(sentence)

        # Handle coordinations: "X is good and Y is bad"
        for claim in list(claims):
            if " and " in claim and len(claim.split(" and ")) > 1:
                parts = [p.strip() for p in claim.split(" and ")]
                if len(parts) == 2:
                    # Check if this is a property list or separate facts
                    if self._is_property_list(parts):
                        # Keep as is: "X is good and bad"
                        pass
                    else:
                        # Split: "X is good" and "Y is bad"
                        claims.remove(claim)
                        claims.extend(parts)

        return [c.strip() for c in claims if c.strip()]

    def _handle_list_facts(self, sentence: str) -> List[str]:
        """Handle lists like 'X causes A, B, and C'."""
        claims = []

        # Match patterns like "X [verb] A, B, and C"
        match = re.match(r"^(.+?)\s+(causes|is|contains|includes|requires|prevents)\s+(.+?)\.?$", sentence)
        if match:
            subject = match.group(1).strip()
            verb = match.group(2).strip()
            objects_str = match.group(3).strip()

            # Split the objects by commas and 'and'
            objects = re.split(r",\s*(?:and\s+)?", objects_str)
            for obj in objects:
                obj = obj.strip().rstrip(".")
                if obj:
                    claims.append(f"{subject} {verb} {obj}")
        else:
            claims.append(sentence)

        return claims

    def _is_property_list(self, parts: List[str]) -> bool:
        """Check if parts are property lists (e.g., 'good and bad') or separate facts."""
        # Simple heuristic: if both parts start with a capital letter, likely separate facts
        if parts[0][0].isupper() and parts[1][0].isupper():
            return False
        return True


class ClaimExtractorAdvanced(ClaimExtractor):
    """Extended claim extractor with dependency parsing for more accurate claims."""

    def __init__(self):
        """Initialize with spaCy and additional NLP tools."""
        try:
            super().__init__()
        except Exception as e:
            # If spacy fails (e.g., Python 3.14 compatibility), use fallback
            print(f"⚠️  spaCy initialization failed: {e}")
            print("    Using base ClaimExtractor without spacy...")
            self.nlp = None

    def extract_claims_with_scoring(self, text: str) -> List[Dict]:
        """Extract claims with a confidence score based on linguistic features.

        Returns:
            List of dicts with: 'claim', 'confidence_score', 'evidence_type'
        """
        claims_meta = self.extract_claims_with_metadata(text)

        scored_claims = []
        for claim_meta in claims_meta:
            score = self._compute_claim_confidence(claim_meta["claim"])
            evidence_type = self._classify_evidence_type(claim_meta["claim"])

            scored_claims.append({
                **claim_meta,
                "confidence_score": score,
                "evidence_type": evidence_type,
            })

        return scored_claims

    def _compute_claim_confidence(self, claim: str) -> float:
        """Compute confidence score for a claim (0.0 to 1.0).

        Higher scores for:
        - Specific statements (with numbers, dates, names)
        - Definitive language
        Lower scores for:
        - Hedged claims ("may", "might", "could")
        - Vague statements
        """
        confidence = 0.75  # Base score

        # Reduce for hedging language
        hedges = ["may", "might", "could", "seems", "appears", "might be", "possibly", "arguably"]
        if any(hedge in claim.lower() for hedge in hedges):
            confidence -= 0.2

        # Increase for specificity
        if any(c.isdigit() for c in claim):
            confidence += 0.1
        if re.search(r"\b(20\d{2}|19\d{2})\b", claim):
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _classify_evidence_type(self, claim: str) -> str:
        """Classify the type of evidence in the claim."""
        lower_claim = claim.lower()

        if any(word in lower_claim for word in ["study", "research", "found", "showed", "found that"]):
            return "empirical"
        elif any(word in lower_claim for word in ["is", "are", "was", "were"]):
            return "definitional"
        elif any(word in lower_claim for word in ["cause", "lead", "result in"]):
            return "causal"
        elif any(word in lower_claim for word in ["associated", "correlated", "linked"]):
            return "correlational"
        else:
            return "general"


# Example usage
if __name__ == "__main__":
    extractor = ClaimExtractor()

    # Sample generated answer
    sample_answer = """Diabetes mellitus is a group of metabolic diseases characterized by high blood sugar levels. 
    Type 1 diabetes must be managed with insulin injections. Type 2 diabetes may be treated with oral medications and lifestyle changes. 
    The disease can cause serious complications including cardiovascular disease, stroke, and kidney damage."""

    print("=" * 70)
    print("CLAIM EXTRACTION EXAMPLE")
    print("=" * 70)
    print(f"\n📝 Original Answer:\n{sample_answer}\n")

    # Extract simple claims
    claims = extractor.extract_claims(sample_answer)
    print(f"✅ Extracted Claims ({len(claims)} claims):")
    for i, claim in enumerate(claims, 1):
        print(f"   {i}. {claim}")

    # Extract claims with metadata
    print("\n📊 Claims with Metadata:")
    claims_meta = extractor.extract_claims_with_metadata(sample_answer)
    for claim_meta in claims_meta:
        print(f"   Claim {claim_meta['id']}: {claim_meta['claim']}")
        print(f"      Source: {claim_meta['source_sentence'][:80]}...")

    # Advanced extraction with scoring
    print("\n🔬 Advanced Extraction with Confidence Scores:")
    try:
        advanced = ClaimExtractorAdvanced()
        scored_claims = advanced.extract_claims_with_scoring(sample_answer)
        for sc in scored_claims[:5]:  # Show first 5
            print(f"   [{sc['confidence_score']:.2f}] {sc['claim']} ({sc['evidence_type']})")
    except Exception as e:
        print(f"   Note: {e}")
