"""
Reranker RAG experiment: retrieve an initial candidate set, rerank with a CrossEncoder,
then answer using the top reranked chunks.

Run with:
    python -m src.reranker
"""
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import CrossEncoder

from src.evaluation import evaluate_all
from src.utils import (
    CHAT_MODEL,
    create_chroma_vectordb_from_pdf,
    download_pdf,
    get_assistant_response,
    load_api_key,
)

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
INITIAL_TOP_K = 10
FINAL_TOP_K = 3
SYSTEM_PROMPT = (
    "You are a financial chatbot trained to answer questions based on the information provided "
    "in 10-K documents. Your responses should be directly sourced from the content of these documents."
)


def rerank_and_create_context(query: str, collection, initial_top_k: int = INITIAL_TOP_K,
                               final_top_k: int = FINAL_TOP_K) -> str:
    """
    Retrieves initial_top_k chunks, reranks with CrossEncoder, returns top final_top_k as context.
    """
    results = collection.query(query_texts=[query], n_results=initial_top_k)
    initial_chunks = [
        metadata['sentence']
        for metadata_pair in results['metadatas']
        for metadata in metadata_pair
    ]

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512, device="cpu")
    scores = cross_encoder.predict([(query, chunk) for chunk in initial_chunks])
    top_indices = np.argsort(scores)[::-1][:final_top_k]
    context = " ".join(initial_chunks[idx] for idx in top_indices)
    print("Reranked context:", context)
    return context


def query_with_reranked_context(query: str, context: str, openai_api_key: str) -> str:
    """Answers a query using pre-built reranked context."""
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
        context = rerank_and_create_context(row['question'], collection)
        print("Querying model...")
        model_answer = query_with_reranked_context(row['question'], context, openai_api_key)
        print(model_answer)

        metrics = evaluate_all(row['question'], model_answer, row['answer'], openai_api_key)
        results_list.append({
            'doc_name': row['doc_name'],
            'question': row['question'],
            'ref_answer': row['answer'],
            'model_answer': model_answer,
            **metrics,
        })

    pd.DataFrame(results_list).to_csv('results/reranker_results.csv', index=False)
    print("Results saved to results/reranker_results.csv")


if __name__ == "__main__":
    main()
