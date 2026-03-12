"""
Quick verification script for the RAG module.
Run: python test_rag_module.py
"""

from rag_module.vector_db import VectorDB
from rag_module.retriever import Retriever
from rag_module.dataset_loader import DatasetLoader, QARecord


def test_dataset_loader():
    print("=" * 50)
    print("1. Testing DatasetLoader")
    print("=" * 50)
    loader = DatasetLoader()

    # Test QARecord creation
    rec = QARecord(question="What is AI?", answer="Artificial Intelligence", source="test")
    assert rec.question == "What is AI?"
    print("   ✅ QARecord creation works")

    # Test load_documents (we'll create a temp dir)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            with open(os.path.join(tmpdir, f"doc_{i}.txt"), "w") as f:
                f.write(f"This is test document number {i}.")
        docs = loader.load_documents(tmpdir)
        assert len(docs) == 3
        assert "text" in docs[0] and "source" in docs[0]
    print(f"   ✅ load_documents works (loaded {len(docs)} docs from temp dir)")
    print()


def test_vector_db():
    print("=" * 50)
    print("2. Testing VectorDB")
    print("=" * 50)

    db = VectorDB()
    print(f"   Model: {db.model_name}")
    print(f"   Embedding dim: {db.embedding_dim}")

    sample_docs = [
        {"text": "The Eiffel Tower is a famous landmark in Paris, France.", "source": "geography"},
        {"text": "Python is widely used in machine learning and data science.", "source": "tech"},
        {"text": "The mitochondria is the powerhouse of the cell.", "source": "biology"},
        {"text": "Albert Einstein developed the theory of relativity.", "source": "physics"},
        {"text": "The Great Wall of China is visible from space.", "source": "geography"},
        {"text": "Deep learning uses neural networks with many layers.", "source": "tech"},
        {"text": "DNA carries genetic information in living organisms.", "source": "biology"},
        {"text": "Shakespeare wrote Hamlet and Macbeth.", "source": "literature"},
    ]

    # Test embedding
    embeddings = db.encode(["test query"])
    assert embeddings.shape == (1, db.embedding_dim)
    print(f"   ✅ Embedding generation works (shape: {embeddings.shape})")

    # Test indexing
    db.build_index(sample_docs)
    assert db.total_documents == len(sample_docs)
    print(f"   ✅ FAISS indexing works ({db.total_documents} docs indexed)")

    # Test search
    results = db.search("Where is the Eiffel Tower?", top_k=3)
    assert len(results) == 3
    assert results[0]["rank"] == 1
    print(f"   ✅ Vector search works (top result: '{results[0]['document']['text'][:60]}...')")

    # Test save/load
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        db.save(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "index.faiss"))
        assert os.path.exists(os.path.join(tmpdir, "documents.json"))
        print(f"   ✅ Index save works")

        db2 = VectorDB()
        db2.load(tmpdir)
        assert db2.total_documents == len(sample_docs)
        results2 = db2.search("Where is the Eiffel Tower?", top_k=1)
        assert results2[0]["document"]["text"] == results[0]["document"]["text"]
        print(f"   ✅ Index load works (restored {db2.total_documents} docs)")
    print()


def test_retriever():
    print("=" * 50)
    print("3. Testing Retriever (end-to-end)")
    print("=" * 50)

    retriever = Retriever(top_k=5)

    documents = [
        {"text": "Diabetes is a chronic metabolic disease characterized by elevated blood sugar levels.", "source": "medical"},
        {"text": "Insulin is a hormone produced by the pancreas that regulates blood glucose.", "source": "medical"},
        {"text": "Type 2 diabetes is often linked to obesity and lack of physical activity.", "source": "medical"},
        {"text": "The Python programming language was created by Guido van Rossum.", "source": "tech"},
        {"text": "Machine learning models can predict disease outcomes from patient data.", "source": "tech"},
        {"text": "Regular exercise helps prevent cardiovascular disease.", "source": "health"},
        {"text": "FAISS is a library developed by Facebook for efficient similarity search.", "source": "tech"},
        {"text": "Blood pressure monitoring is essential for heart disease prevention.", "source": "health"},
    ]

    retriever.index_documents(documents)
    assert retriever.total_documents == len(documents)
    print(f"   ✅ Indexed {retriever.total_documents} documents")

    # Test retrieval
    query = "What causes diabetes?"
    results = retriever.retrieve(query, top_k=5)
    print(f"\n   Query: '{query}'")
    print(f"   Top {len(results)} retrieved documents:")
    for r in results:
        print(f"      [{r['rank']}] (score={r['score']:.4f}) {r['document']['text'][:80]}...")

    # Verify medical docs rank higher than unrelated ones for this query
    top3_sources = [r["document"]["source"] for r in results[:3]]
    medical_in_top3 = sum(1 for s in top3_sources if s == "medical")
    assert medical_in_top3 >= 2, f"Expected >=2 medical docs in top 3, got {medical_in_top3}"
    print(f"\n   ✅ Relevance check passed ({medical_in_top3}/3 top results are medical)")

    # Test retrieve_text convenience method
    texts = retriever.retrieve_text(query, top_k=3)
    assert len(texts) == 3
    assert all(isinstance(t, str) for t in texts)
    print(f"   ✅ retrieve_text works (returns {len(texts)} strings)")

    # Test save/load round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        retriever.save(tmpdir)
        retriever2 = Retriever(index_path=tmpdir)
        retriever2.load()
        results2 = retriever2.retrieve(query, top_k=1)
        assert results2[0]["document"]["text"] == results[0]["document"]["text"]
        print(f"   ✅ Save/load round-trip works")
    print()


if __name__ == "__main__":
    print("\n🔍 RAG Module Verification\n")
    try:
        test_dataset_loader()
        test_vector_db()
        test_retriever()
        print("=" * 50)
        print("🎉 ALL TESTS PASSED — Your RAG module is working!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
