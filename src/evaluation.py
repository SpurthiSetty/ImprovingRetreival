"""
Shared evaluation metrics: cosine similarity, BERTScore, and LLM-based evaluation.
"""
import re

from bert_score import score as bert_score_fn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from src.utils import get_assistant_response


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Returns TF-IDF cosine similarity between two texts."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return sk_cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


def calculate_bertscore(candidate: str, reference: str) -> float:
    """Returns BERTScore precision for a candidate against a reference."""
    P, R, F1 = bert_score_fn([candidate], [reference], lang="en", verbose=True)
    return P.mean().item()


def evaluate_llm_responses(question: str, model_answer: str, reference_answer: str,
                            openai_api_key: str) -> float:
    """
    Uses GPT-4o to assign a 0-1 correctness score for model_answer vs reference_answer.
    Returns the numeric score (0.0 on parse failure).
    """
    evaluation_prompt = f"""
    Evaluate the following response against the reference answer. Assign a score between 0 and 1 based on correctness and provide a brief justification.

    Question: {question}
    Response: {model_answer}
    Reference Answer: {reference_answer}

    Score (0 to 1):
    Justification:
    """
    messages = [
        {"role": "system", "content": "You are an evaluator that scores responses based on correctness."},
        {"role": "user", "content": evaluation_prompt},
    ]
    evaluation_text = get_assistant_response(messages, openai_api_key).strip()
    print(f"Evaluation Response: {evaluation_text}")

    match = re.search(r'Score\s*\(\s*0\s*to\s*1\s*\)\s*:\s*([0-9]*\.?[0-9]+)', evaluation_text)
    if match:
        result = float(match.group(1))
    else:
        print("Score not found in evaluation response; defaulting to 0.0")
        result = 0.0

    print(f"Correctness Score: {result:.2f}")
    return result


def evaluate_all(question: str, model_answer: str, ref_answer: str,
                 openai_api_key: str) -> dict:
    """Runs all three evaluation metrics and returns a result dict."""
    return {
        'cosine_similarity': calculate_cosine_similarity(model_answer, ref_answer),
        'bert_score': calculate_bertscore(model_answer, ref_answer),
        'llm_eval': evaluate_llm_responses(question, model_answer, ref_answer, openai_api_key),
    }
