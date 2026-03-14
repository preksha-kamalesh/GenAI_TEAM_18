# Generation Module — Navya's Part
## RAG-based Answer Generation + Claim Extraction

**Owner:** Navya  
**Branch:** `feature/generation-module`

### Overview

This module handles the **second stage** of the pipeline. It takes:
- User question
- Top-k retrieved documents (from Preksha's Retrieval module)

And produces:
- Generated answer from an LLM
- Extracted atomic factual claims (for Pranav's verification module)

### System Pipeline

```
           Preksha (Retrieval)
                   ↓
    Question → Vector Search → Top-k Documents
                   ↓
           Navya (Generation) ← YOU ARE HERE
                   ↓
    Question + Docs → LLM → Generated Answer
                   ↓
           Claim Extraction
                   ↓
        Atomic Factual Claims
                   ↓
           Pranav (Verification)
                   ↓
    Verify Claims → Correction → Final Answer
```

---

## Architecture

### Components

#### 1. RAGGenerator (`rag_generator.py`)
High-level interface for LLM-based answer generation.

**Supported LLM Backends:**
- **OpenAI** (GPT-3.5 / GPT-4 via API)
- **HuggingFace** (Llama 3, Mistral, etc. - local)
- **Custom** (implement `LLMBackend` interface)

**Key Methods:**
```python
generator = RAGGenerator(backend_type="huggingface", model_name="mistralai/Mistral-7B-Instruct-v0.1")
result = generator.generate_answer(
    question="What is diabetes?",
    retrieved_docs=[{text: "...", source: "..."}],
    top_k=5,
    max_tokens=512,
    temperature=0.7
)
# Returns: {question, answer, context, num_docs_used, documents}
```

#### 2. ClaimExtractor (`claim_extractor.py`)
Breaks generated answers into atomic factual claims.

**Features:**
- Sentence splitting (using spaCy or regex fallback)
- List handling: "X causes A, B, and C" → ["X causes A", "X causes B", "X causes C"]
- Metadata tracking (source sentence, position)
- Deduplication

**Key Methods:**
```python
extractor = ClaimExtractor()

# Simple claim extraction
claims = extractor.extract_claims(answer_text)
# Returns: ["claim1", "claim2", ...]

# With metadata
claims_meta = extractor.extract_claims_with_metadata(answer_text)
# Returns: [{"id": 0, "claim": "...", "source_sentence": "...", "sentence_position": 0}, ...]
```

#### 3. ClaimExtractorAdvanced (`claim_extractor.py`)
Enhanced extraction with confidence scoring and evidence classification.

```python
advanced = ClaimExtractorAdvanced()
scored_claims = advanced.extract_claims_with_scoring(answer_text)
# Returns: [{"claim": "...", "confidence_score": 0.85, "evidence_type": "empirical"}, ...]
```

---

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Optional: spaCy Model (for better sentence splitting)
```bash
python -m spacy download en_core_web_sm
```

### Optional: LLM Models

**For Mistral (HuggingFace):**
```bash
# Models are downloaded on first use (auto-cached)
# Requires GPU (~8GB VRAM) for optimal speed
```

**For GPT (OpenAI):**
```bash
export OPENAI_API_KEY="your-api-key"
```

---

## Quick Start

### Example 1: Generate Answer from Retrieved Docs

```python
from generation import RAGGenerator, ClaimExtractor
from rag_module import Retriever

# Step 1: Retrieve documents (using Preksha's module)
retriever = Retriever(top_k=5)
retriever.index_documents([...])  # Load your corpus
retrieved_docs = retriever.retrieve("What is diabetes?")

# Step 2: Generate answer (YOUR MODULE)
generator = RAGGenerator(backend_type="huggingface")
result = generator.generate_answer(
    question="What is diabetes?",
    retrieved_docs=retrieved_docs,
    max_tokens=512
)
print(result['answer'])

# Step 3: Extract claims
extractor = ClaimExtractor()
claims = extractor.extract_claims(result['answer'])
for claim in claims:
    print(f"  • {claim}")
```

### Example 2: Full End-to-End Pipeline

```bash
python end_to_end_example.py
```

This runs:
1. Retrieve relevant documents
2. Generate answer
3. Extract claims
4. Show output for Pranav's verification

### Example 3: Run Tests

```bash
python test_generation.py
```

Tests:
- ✅ Claim extraction
- ✅ Advanced claim scoring
- ✅ RAG generation
- ✅ Full integration

---

## Usage Details

### RAGGenerator

#### Initialization

**Option A: HuggingFace (Local LLM)**
```python
from generation import RAGGenerator

generator = RAGGenerator(
    backend_type="huggingface",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device="cuda"  # or "cpu"
)
```

**Option B: OpenAI (API)**
```python
generator = RAGGenerator(
    backend_type="openai",
    api_key="sk-...",  # or use OPENAI_API_KEY env var
    model="gpt-3.5-turbo"
)
```

**Option C: Custom Backend**
```python
from generation import RAGGenerator, LLMBackend

class MyLLMBackend(LLMBackend):
    def generate(self, prompt, max_tokens=512, temperature=0.7):
        # Your implementation
        return "answer text"

backend = MyLLMBackend()
generator = RAGGenerator(llm_backend=backend)
```

#### Generating Answers

```python
result = generator.generate_answer(
    question="What are the symptoms of diabetes?",
    retrieved_docs=[
        {"text": "Diabetes causes...", "source": "wiki"},
        {"text": "Symptoms include...", "source": "wiki"},
    ],
    top_k=5,              # Use top 5 docs (if more provided)
    max_tokens=512,       # Max response length
    temperature=0.5       # 0.0=deterministic, 1.0=random
)

print(result['question'])
print(result['answer'])
print(result['context'])
print(result['num_docs_used'])
```

### ClaimExtractor

#### Basic Extraction

```python
extractor = ClaimExtractor()
claims = extractor.extract_claims(
    "Diabetes causes nausea and dizziness. It is serious.",
    lowercase=True  # Convert to lowercase
)
# Output: ["diabetes causes nausea", "diabetes causes dizziness", "it is serious"]
```

#### With Metadata

```python
claims_meta = extractor.extract_claims_with_metadata(
    "Type 1 diabetes requires insulin. Type 2 does not."
)
# Output:
# [
#   {
#     "id": 0,
#     "claim": "type 1 diabetes requires insulin",
#     "source_sentence": "Type 1 diabetes requires insulin.",
#     "sentence_position": 0
#   },
#   ...
# ]
```

#### Advanced Scoring

```python
advanced = ClaimExtractorAdvanced()
scored = advanced.extract_claims_with_scoring(text)
# Output:
# [
#   {
#     "claim": "research shows exercise prevents type 2 diabetes",
#     "confidence_score": 0.95,  # High confidence (mentions study)
#     "evidence_type": "empirical"
#   },
#   {
#     "claim": "diabetes might cause complications",
#     "confidence_score": 0.65,  # Lower (hedged language)
#     "evidence_type": "general"
#   }
# ]
```

---

## Output Format for Pranav (Verification Module)

Your module outputs claims in this format (for Pranav to verify):

```python
output = {
    "question": "What is diabetes?",
    "generated_answer": "Diabetes is...",
    "retrieved_documents": [
        {"text": "...", "source": "..."},
        ...
    ],
    "claims": [
        {
            "id": 0,
            "claim": "diabetes is a metabolic disease",
            "source_sentence": "Diabetes mellitus is a group of metabolic diseases...",
            "sentence_position": 0,
            "confidence_score": 0.85,  # Optional
            "evidence_type": "definitional"  # Optional
        },
        ...
    ]
}
```

Pranav will:
1. Check each claim against retrieved documents
2. Mark as **Supported**, **Hallucination**, or **Uncertain**
3. Correct or remove hallucinations
4. Return final verified answer

---

## File Structure

```
generation/
├── __init__.py              # Package exports
├── rag_generator.py         # LLM integration + RAG prompting
├── claim_extractor.py       # Claim extraction + scoring
└── README.md                # This file
```

---

## Technical Details

### LLM Prompting Strategy

The RAG prompt is designed to:
1. **Ground answers in context** - "Answer ONLY based on provided context"
2. **Reduce hallucinations** - "Do NOT make up information"
3. **Include sources** - "Include relevant source references"

**Prompt Template:**
```
You are a helpful assistant providing accurate answers based on provided context.

Context (Retrieved Documents):
[Document 1] (source)
...
[Document k] (source)
...

Question:
{user_question}

Instructions:
- Answer ONLY based on the provided context.
- If the context doesn't contain sufficient information to answer, say so.
- Be concise but thorough.
- Include relevant source references when applicable.
- Do NOT make up information or hallucinate facts not in the documents.

Answer:
```

### Claim Extraction Algorithm

1. **Text Cleaning**: Remove citations, normalize whitespace
2. **Sentence Splitting**: Use spaCy (or regex fallback)
3. **Atomic Splitting**: Break sentences into smaller claims
   - Handle lists: "X causes A, B, and C" → 3 claims
   - Handle coordination: "X is good and Y is bad" → 2 claims
4. **Deduplication**: Remove duplicate claims while preserving order
5. **Filtering**: Keep claims > 5 characters

**Example:**
```
Input:  "Diabetes causes nausea and dizziness. It is serious."
Output: [
  "diabetes causes nausea",
  "diabetes causes dizziness",
  "it is serious"
]
```

---

## Integration with Other Modules

### Receives from Preksha (Retrieval):
- `question`: str
- `retrieved_docs`: List[dict] with keys "text" and "source"

### Outputs to Pranav (Verification):
- `generated_answer`: str
- `claims`: List[dict] with "claim", "source_sentence", "confidence_score"
- `retrieved_documents`: List[dict] (for reference)

### Example Integration:
```python
# Preksha's output
retrieved = retriever.retrieve("What is X?", top_k=5)

# Your generation (Navya)
result = generator.generate_answer("What is X?", retrieved)
claims = extractor.extract_claims_with_metadata(result['answer'])

# Pranav's input
verification_input = {
    "question": result['question'],
    "answer": result['answer'],
    "claims": claims,
    "documents": retrieved
}
```

---

## Configuration & Customization

### Change LLM Model

```python
# Use a different Mistral version
generator = RAGGenerator(
    backend_type="huggingface",
    model_name="mistralai/Mistral-7B-v0.1"
)

# Use GPT-4 instead of GPT-3.5
generator = RAGGenerator(
    backend_type="openai",
    model="gpt-4"
)
```

### Adjust Generation Parameters

```python
result = generator.generate_answer(
    question=q,
    retrieved_docs=docs,
    max_tokens=1024,    # Longer responses
    temperature=0.3     # More deterministic
)
```

### Custom Prompt Template

```python
# Subclass RAGGenerator and override _build_rag_prompt()
class CustomRAGGenerator(RAGGenerator):
    def _build_rag_prompt(self, question, context):
        return f"""Custom prompt format:
        Context: {context}
        Q: {question}
        A:"""

generator = CustomRAGGenerator()
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `transformers` not installed | `pip install transformers torch` |
| spaCy model not found | `python -m spacy download en_core_web_sm` |
| CUDA out of memory | Switch to CPU: `device="cpu"` or use smaller model |
| OpenAI API errors | Check `OPENAI_API_KEY` env var is set |
| Slow generation | Use smaller model or switch to GPT-3.5 |

---

## Next Steps

✅ **Your deliverables (Navya):**
- `generation/rag_generator.py` — LLM generation
- `generation/claim_extractor.py` — Claim extraction
- Generated answers + claims ready for Pranav's verification

👉 **Pranav's next steps:**
- Verify claims against retrieved documents
- Detect hallucinations using NLI
- Correct/filter final answer

---

## Related Modules

- **Preksha's Retrieval Module** → [rag_module/README.md](../rag_module/README.md)
- **Pranav's Verification Module** → [verification/README.md](../verification/README.md) (coming soon)

---

## References

- Sentence-BERT (Embeddings): https://www.sbert.net/
- FAISS (Vector DB): https://github.com/facebookresearch/faiss
- HuggingFace Transformers: https://huggingface.co/transformers/
- spaCy (NLP): https://spacy.io/
