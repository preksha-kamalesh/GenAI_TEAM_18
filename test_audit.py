"""
Comprehensive audit test — checks all edge cases and missing coverage.
"""
import tempfile
import os
import json
import csv

from rag_module.dataset_loader import DatasetLoader, QARecord
from rag_module.vector_db import VectorDB
from rag_module.retriever import Retriever

# Shared instances to avoid reloading the model for every test
_shared_db = None
_shared_loader = DatasetLoader()


def get_shared_db():
    global _shared_db
    if _shared_db is None:
        _shared_db = VectorDB()
    return _shared_db


def fresh_db():
    """Return a VectorDB that reuses the shared model but has a fresh FAISS index."""
    import faiss
    db = get_shared_db()
    db.index = faiss.IndexFlatL2(db.embedding_dim)
    db.documents = []
    return db


def fresh_retriever(top_k=5):
    """Return a Retriever that reuses the shared model's VectorDB."""
    import faiss
    db = get_shared_db()
    db.index = faiss.IndexFlatL2(db.embedding_dim)
    db.documents = []
    retriever = Retriever.__new__(Retriever)
    retriever.top_k = top_k
    retriever.loader = _shared_loader
    retriever.db = db
    retriever._indexed = False
    return retriever


def test_csv_loader():
    print("Testing CSV loader...")
    loader = _shared_loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "source"])
        writer.writeheader()
        writer.writerow({"question": "What is AI?", "answer": "Artificial Intelligence", "source": "test"})
        writer.writerow({"question": "What is ML?", "answer": "Machine Learning", "source": "test"})
        csv_path = f.name
    try:
        records = loader.load_csv(csv_path)
        assert len(records) == 2
        assert records[0].question == "What is AI?"
        assert records[1].answer == "Machine Learning"
        print("   ✅ CSV loader works")
    finally:
        os.unlink(csv_path)


def test_json_loader():
    print("Testing JSON loader...")
    loader = _shared_loader
    data = [
        {"question": "What is DL?", "answer": "Deep Learning", "source": "test"},
        {"question": "What is NLP?", "answer": "Natural Language Processing"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        json_path = f.name
    try:
        records = loader.load_json(json_path)
        assert len(records) == 2
        assert records[0].answer == "Deep Learning"
        assert records[1].source == ""  # missing source defaults to ""
        print("   ✅ JSON loader works (including missing 'source' field)")
    finally:
        os.unlink(json_path)


def test_empty_directory():
    print("Testing empty directory...")
    loader = _shared_loader
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = loader.load_documents(tmpdir)
        assert len(docs) == 0
    print("   ✅ Empty directory returns empty list")


def test_non_txt_files_skipped():
    print("Testing non-.txt files are skipped...")
    loader = _shared_loader
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create .txt and non-.txt files
        with open(os.path.join(tmpdir, "good.txt"), "w") as f:
            f.write("Valid document.")
        with open(os.path.join(tmpdir, "bad.pdf"), "w") as f:
            f.write("Should be skipped.")
        with open(os.path.join(tmpdir, "bad.py"), "w") as f:
            f.write("Should be skipped.")
        docs = loader.load_documents(tmpdir)
        assert len(docs) == 1
        assert docs[0]["source"] == "good.txt"
    print("   ✅ Non-.txt files correctly skipped")


def test_empty_txt_files_skipped():
    print("Testing empty .txt files are skipped...")
    loader = _shared_loader
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "empty.txt"), "w") as f:
            f.write("")
        with open(os.path.join(tmpdir, "whitespace.txt"), "w") as f:
            f.write("   \n\n  ")
        with open(os.path.join(tmpdir, "valid.txt"), "w") as f:
            f.write("Has content.")
        docs = loader.load_documents(tmpdir)
        assert len(docs) == 1
    print("   ✅ Empty/whitespace-only .txt files correctly skipped")


def test_long_document_embedding():
    print("Testing long document embedding (>512 tokens)...")
    db = fresh_db()
    long_text = "The quick brown fox jumps over the lazy dog. " * 500
    db.build_index([{"text": long_text, "source": "long_doc"}])
    results = db.search("fox jumping", top_k=1)
    assert len(results) == 1
    print(f"   ✅ Long doc handled (score={results[0]['score']:.4f})")


def test_search_with_top_k_larger_than_index():
    print("Testing top_k > number of documents...")
    db = fresh_db()
    db.build_index([
        {"text": "Only one document here.", "source": "solo"}
    ])
    results = db.search("document", top_k=10)
    assert len(results) == 1  # should return 1, not crash
    print("   ✅ Gracefully returns available docs when top_k > index size")


def test_search_empty_index():
    print("Testing search on empty index...")
    db = fresh_db()
    results = db.search("anything", top_k=5)
    assert len(results) == 0
    print("   ✅ Empty index returns empty results")


def test_retriever_error_on_no_index():
    print("Testing retriever error when no docs indexed...")
    retriever = fresh_retriever()
    try:
        retriever.retrieve("test query")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "No documents indexed" in str(e)
    print("   ✅ Raises RuntimeError when no documents indexed")


def test_add_documents_incremental():
    print("Testing incremental document addition...")
    db = fresh_db()
    db.build_index([{"text": "First document.", "source": "a"}])
    assert db.total_documents == 1
    db.add_documents([{"text": "Second document.", "source": "b"}])
    assert db.total_documents == 2
    results = db.search("Second", top_k=1)
    assert "Second" in results[0]["document"]["text"]
    print("   ✅ Incremental add_documents works")


def test_retriever_index_from_directory():
    print("Testing retriever.index_from_directory()...")
    retriever = fresh_retriever(top_k=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            with open(os.path.join(tmpdir, f"doc_{i}.txt"), "w") as f:
                f.write(f"Document number {i} about topic {['AI','biology','math','history','physics'][i]}.")
        retriever.index_from_directory(tmpdir)
        assert retriever.total_documents == 5
        results = retriever.retrieve("artificial intelligence")
        assert len(results) == 2
    print(f"   ✅ index_from_directory works ({retriever.total_documents} docs)")


def test_retriever_empty_dir_error():
    print("Testing retriever error on empty directory...")
    retriever = fresh_retriever()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            retriever.index_from_directory(tmpdir)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No .txt documents found" in str(e)
    print("   ✅ Raises ValueError for empty corpus directory")


def test_unsupported_dataset_error():
    print("Testing unsupported dataset name error...")
    loader = _shared_loader
    try:
        loader.load_hf_dataset("nonexistent_dataset")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported dataset" in str(e)
    print("   ✅ Raises ValueError for unsupported dataset name")


def test_qarecord_metadata():
    print("Testing QARecord metadata field...")
    rec = QARecord(question="Q", answer="A", metadata={"key": "value"})
    assert rec.metadata == {"key": "value"}
    rec2 = QARecord(question="Q", answer="A")
    assert rec2.metadata == {}
    print("   ✅ QARecord metadata works correctly")


if __name__ == "__main__":
    print("\n🔎 Comprehensive RAG Module Audit\n")
    tests = [
        test_csv_loader,
        test_json_loader,
        test_empty_directory,
        test_non_txt_files_skipped,
        test_empty_txt_files_skipped,
        test_long_document_embedding,
        test_search_with_top_k_larger_than_index,
        test_search_empty_index,
        test_retriever_error_on_no_index,
        test_add_documents_incremental,
        test_retriever_index_from_directory,
        test_retriever_empty_dir_error,
        test_unsupported_dataset_error,
        test_qarecord_metadata,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"   ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("🎉 ALL AUDIT TESTS PASSED")
    else:
        print(f"⚠️  {failed} test(s) FAILED — needs fixing")
    print(f"{'='*50}")
