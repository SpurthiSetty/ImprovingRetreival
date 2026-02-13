#!/usr/bin/env bash
# Run all three RAG experiments sequentially.
# Usage: bash run_experiments.sh
set -euo pipefail

if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and set OPENAI_API_KEY."
    exit 1
fi

mkdir -p results

echo "=== Running Base RAG ==="
python -m src.base_rag

echo "=== Running Query Expansion RAG ==="
python -m src.query_expansion

echo "=== Running Reranker RAG ==="
python -m src.reranker

echo ""
echo "All experiments complete. Results saved to results/"
ls -lh results/*.csv
