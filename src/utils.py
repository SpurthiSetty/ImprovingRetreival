"""
Shared utilities: PDF handling, text extraction, and Chroma vector DB creation.
"""
import base64
import os
from urllib.parse import parse_qs, urlparse

import chromadb
import fitz  # PyMuPDF
import openai
import requests
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4o"
COLLECTION_NAME = "Finance_bench_documents"


def extract_pdf_url(url: str) -> str:
    """Extracts the actual PDF URL, decoding base64 encoding if necessary."""
    if url.lower().endswith('.pdf'):
        return url
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    pdf_target = query_params.get('pdfTarget', [None])[0]
    if pdf_target:
        return base64.b64decode(pdf_target).decode('utf-8')
    raise ValueError("No valid PDF URL found in the provided URL")


def download_pdf(url: str, save_path: str) -> None:
    """Downloads a PDF from a URL, skipping if the file already exists."""
    if os.path.exists(save_path):
        print(f"PDF already exists, skipping download: {save_path}")
        return
    try:
        pdf_url = extract_pdf_url(url)
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded PDF: {pdf_url} -> {save_path}")
    except Exception as e:
        print(f"Error downloading PDF: {e}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file using PyMuPDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc.load_page(page_num).get_text()
    return text


def create_chroma_vectordb_from_pdf(pdf_path: str, openai_api_key: str, batch_size: int = 100):
    """
    Creates an in-memory Chroma vector database from a PDF file.
    Returns the Chroma collection.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        sentences = [s.strip() for s in text.split('\n') if s.strip()]

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=EMBEDDING_MODEL,
        )

        vectors = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            if batch:
                vectors.extend(openai_ef(batch))

        client = chromadb.Client(Settings())
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef,
        )
        for i, (sentence, vector) in enumerate(zip(sentences, vectors)):
            collection.upsert(f"id_{i}", vector, {"sentence": sentence})

        print(f"Stored {len(sentences)} vectors in Chroma.")
        return collection
    except Exception as e:
        print(f"Error creating Chroma vector database: {e}")


def get_assistant_response(messages: list, openai_api_key: str) -> str:
    """Calls the OpenAI chat completion API and returns the assistant reply."""
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    return response.choices[0].message.content


def load_api_key() -> str:
    """Loads the OpenAI API key from the environment."""
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable not set. "
                         "Copy .env.example to .env and fill in your key.")
    return key
