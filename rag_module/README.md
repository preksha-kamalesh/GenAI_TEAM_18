# GenAI Team 18 — RAG + Self-Verification Pipeline

## System Pipeline

```
Dataset → Retrieval (RAG) → LLM Generation → Claim Extraction → Claim Verification → Correction/Filtering → Final Answer
```

---

## Module: Retrieval & Knowledge Grounding (RAG)

**Owner:** Preksha  
**Branch:** `feature/rag-retrieval-module`

### Overview

This module handles the first stage of the pipeline — given a user question, it retrieves the top-k most relevant documents from a knowledge corpus using dense vector similarity search.

### Architecture

```
User Question
     │
     ▼
┌──────────────────┐
│  Sentence-        │
│  Transformer      │  (all-MiniLM-L6-v2)
│  Embedding        │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  FAISS Vector     │
│  Index            │
│  (L2 similarity)  │
└────────┬─────────┘
         ▼
   Top-k Documents
```

### Files

| File | Description |
|------|-------------|
| `rag_module/__init__.py` | Package init, exports `DatasetLoader`, `VectorDB`, `Retriever` |
| `rag_module/dataset_loader.py` | Loads QA datasets (NQ, TriviaQA, PubMedQA, CSV, JSON) |
| `rag_module/vector_db.py` | Embedding generation + FAISS index management |
| `rag_module/retriever.py` | High-level retrieval interface for downstream modules |

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Index documents and query
python -m rag_module.retriever \
  --docs-dir ./data/corpus/ \
  --query "What causes diabetes?" \
  --top-k 5 \
  --index-path ./data/index/
```

### Usage in Code

```python
from rag_module import Retriever

# Initialize
retriever = Retriever(top_k=5, index_path="./data/index")

# Index a corpus of documents
documents = [
    {"text": "Diabetes is a metabolic disease...", "source": "wiki"},
    {"text": "Insulin is produced by the pancreas...", "source": "wiki"},
]
retriever.index_documents(documents)

# Retrieve relevant documents for a question
results = retriever.retrieve("What causes diabetes?")
for r in results:
    print(f"[{r['rank']}] {r['document']['text'][:100]}...")
```

### Integration Interface

The downstream **LLM Generation** module should call:

```python
retriever = Retriever(index_path="./data/index")
retriever.load()

# Returns top-5 document texts to feed into the LLM prompt
context_docs = retriever.retrieve_text("user question here", top_k=5)
```

### Supported Datasets

| Dataset | HuggingFace ID | Domain |
|---------|---------------|--------|
| Natural Questions | `google-research-datasets/natural_questions` | General |
| TriviaQA | `mandarjoshi/trivia_qa` | General |
| PubMedQA | `qiaojin/PubMedQA` | Medical |
| Custom CSV/JSON | — | Any |

### Dependencies

- `sentence-transformers` — Embedding model
- `faiss-cpu` — Vector similarity search
- `datasets` — HuggingFace dataset loading
- `numpy` — Numerical operations
