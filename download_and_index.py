"""
Download a real dataset and build the FAISS index.
This only needs to be run once — the index is saved to data/index/.

Usage:
    python download_and_index.py --dataset pubmed_qa --max-samples 1500
    python download_and_index.py --dataset trivia_qa --max-samples 500
    python download_and_index.py --corpus ./data/sample_corpus
"""

import argparse
import os
from rag_module import DatasetLoader, Retriever


def main():
    parser = argparse.ArgumentParser(description="Download dataset & build FAISS index")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["natural_questions", "trivia_qa", "pubmed_qa"],
                        help="HuggingFace dataset to download")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to local .txt corpus directory instead")
    parser.add_argument("--max-samples", type=int, default=1500,
                        help="Max records to download (keeps it fast)")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Words per chunk for corpus docs (0 = no chunking)")
    parser.add_argument("--index-path", type=str, default="./data/index",
                        help="Where to save the FAISS index")
    args = parser.parse_args()

    if not args.dataset and not args.corpus:
        print("Provide --dataset or --corpus. Example:")
        print("  python download_and_index.py --dataset pubmed_qa --max-samples 1500")
        print("  python download_and_index.py --corpus ./data/sample_corpus")
        return

    retriever = Retriever(top_k=5, index_path=args.index_path)

    if args.corpus:
        print(f"Loading documents from {args.corpus}...")
        loader = DatasetLoader()
        documents = loader.load_documents(args.corpus, chunk_size=args.chunk_size)
        print(f"  Loaded {len(documents)} chunks")
        retriever.index_documents(documents)
    else:
        print(f"Downloading '{args.dataset}' (max {args.max_samples} samples)...")
        print("  (First run downloads the dataset — may take a few minutes)\n")
        retriever.index_from_qa_dataset(args.dataset, split="train", max_samples=args.max_samples)

    print(f"\nIndexed {retriever.total_documents} documents.")

    # Save index
    retriever.save()
    print(f"Index saved to {args.index_path}/")

    # Quick demo query
    demo_queries = {
        "pubmed_qa": "What are the symptoms of diabetes?",
        "trivia_qa": "Who wrote Romeo and Juliet?",
        "natural_questions": "What is the capital of France?",
    }
    query = demo_queries.get(args.dataset, "What is this about?")
    print(f"\n--- Demo Query: '{query}' ---\n")
    results = retriever.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r['rank']}] (score={r['score']:.4f}) {r['document']['text'][:120]}...")

    print(f"\n✅ Done! Your teammates can now load this index with:")
    print(f"   retriever = Retriever(index_path='{args.index_path}')")
    print(f"   retriever.load()")


if __name__ == "__main__":
    main()
