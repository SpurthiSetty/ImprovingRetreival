"""
Baseline RAG experiment: retrieve top-k chunks via dense retrieval, then answer with GPT-4o.

Run with:
    python -m src.base_rag
"""
import os

import pandas as pd
from chromadb.utils import embedding_functions
from datasets import load_dataset

from src.evaluation import evaluate_all
from src.utils import (
    create_chroma_vectordb_from_pdf,
    download_pdf,
    get_assistant_response,
    load_api_key,
)

TOP_K = 5
SYSTEM_PROMPT = (
    "You are a financial chatbot trained to answer questions based on the information "
    "provided in 10-K documents. Your responses should be directly sourced from the "
    "content of these documents."
)


def query_with_rag(query: str, collection, openai_api_key: str, top_k: int = TOP_K) -> str:
    """Retrieves top-k chunks from Chroma and answers query with GPT-4o."""
    results = collection.query(query_texts=[query], n_results=top_k)
    context = "\n".join(
        metadata['sentence']
        for metadata_pair in results['metadatas']
        for metadata in metadata_pair
    )
    print("Context retrieved:", context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"\nContext:\n{context}\n\nQuery: {query}\n\nAnswer:"},
    ]
    return get_assistant_response(messages, openai_api_key)


def main():
    openai_api_key = load_api_key()
    dataset = load_dataset("PatronusAI/financebench")
    test = pd.DataFrame(dataset['train'])[:5]
    results_list = []

    for _, row in test.iterrows():
        download_dir = "pdf_documents"
        os.makedirs(download_dir, exist_ok=True)
        doc_path = os.path.join(download_dir, f"{row['doc_name']}.pdf")

        download_pdf(row['doc_link'], doc_path)
        collection = create_chroma_vectordb_from_pdf(doc_path, openai_api_key)
        print("Querying model...")
        model_answer = query_with_rag(row['question'], collection, openai_api_key)
        print(model_answer)

        metrics = evaluate_all(row['question'], model_answer, row['answer'], openai_api_key)
        results_list.append({
            'doc_name': row['doc_name'],
            'question': row['question'],
            'ref_answer': row['answer'],
            'model_answer': model_answer,
            **metrics,
        })

    pd.DataFrame(results_list).to_csv('results/base_rag_results.csv', index=False)
    print("Results saved to results/base_rag_results.csv")


if __name__ == "__main__":
    main()
