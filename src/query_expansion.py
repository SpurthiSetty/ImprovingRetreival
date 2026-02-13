"""
Query Expansion RAG experiment: generate an initial LLM response without context,
then refine using retrieved chunks that match both the original query and initial response.

Run with:
    python -m src.query_expansion
"""
import os

import openai
import pandas as pd
from datasets import load_dataset

from src.evaluation import evaluate_all
from src.utils import (
    CHAT_MODEL,
    create_chroma_vectordb_from_pdf,
    download_pdf,
    load_api_key,
)

NO_CONTEXT_SYSTEM_PROMPT = (
    "You are a financial chatbot trained to answer financial questions to the absolute best "
    "of your ability. Your primary focus should be on accuracy, specificity, and correctness, "
    "particularly relating to financial statements, company performance, and market position. "
    "Please answer each question with total accuracy, performing all necessary calculations "
    "without skipping or simplifying any steps along the way. If you do not have enough "
    "information to answer a question, please make whatever reasonable assumptions are "
    "necessary and provide a full and complete answer."
)
RAG_SYSTEM_PROMPT = (
    "You are a financial chatbot trained to answer questions based on the information provided "
    "in 10-K documents. Your responses should be directly sourced from the content of these documents."
)


def no_context_response(query: str, openai_api_key: str) -> str:
    """Step 1: Generate an initial answer without any retrieved context."""
    client = openai.OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": NO_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}"},
        ],
    )
    return completion.choices[0].message.content


def query_expansion_response(query: str, initial_response: str, collection,
                             openai_api_key: str, top_k: int = 5) -> str:
    """Step 2: Retrieve chunks using both query + initial response, then answer."""
    rag_results = collection.query(query_texts=[query, initial_response], n_results=top_k)
    context = "\n".join(
        metadata['sentence']
        for metadata_pair in rag_results['metadatas']
        for metadata in metadata_pair
    )
    print("Context retrieved:", context)
    client = openai.OpenAI(api_key=openai_api_key)
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}. Relevant document information: {context}"},
        ],
    )
    return completion.choices[0].message.content


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
        initial = no_context_response(row['question'], openai_api_key)
        model_answer = query_expansion_response(row['question'], initial, collection, openai_api_key)
        print(model_answer)

        metrics = evaluate_all(row['question'], model_answer, row['answer'], openai_api_key)
        results_list.append({
            'doc_name': row['doc_name'],
            'question': row['question'],
            'ref_answer': row['answer'],
            'model_answer': model_answer,
            **metrics,
        })

    pd.DataFrame(results_list).to_csv('results/query_expansion_results.csv', index=False)
    print("Results saved to results/query_expansion_results.csv")


if __name__ == "__main__":
    main()
