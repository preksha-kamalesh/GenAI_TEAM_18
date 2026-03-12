"""
Test the RAG retriever on the real PubMedQA dataset.
Loads the saved FAISS index and runs multiple real-world medical queries.
"""
from rag_module import Retriever

retriever = Retriever(index_path="./data/index")
retriever.load()
print(f"Loaded index with {retriever.total_documents} real PubMedQA documents\n")

queries = [
    "What are the symptoms of diabetes?",
    "Does smoking cause lung cancer?",
    "What is the treatment for hypertension?",
    "Is surgery effective for knee osteoarthritis?",
    "How does chemotherapy affect the immune system?",
    "What causes Alzheimer's disease?",
    "Is aspirin effective in preventing heart attacks?",
    "What are the side effects of radiation therapy?",
]

for query in queries:
    print(f"Q: {query}")
    results = retriever.retrieve(query, top_k=3)
    for r in results:
        doc = r["document"]
        text = doc["text"][:150].replace("\n", " ")
        print(f"  [{r['rank']}] (score={r['score']:.4f}) {text}...")
    print()

print("=" * 60)
print("DONE — Retriever is working on real PubMedQA data.")
