#!/usr/bin/env bash
# Phase d'installation de la finale : seule phase où le réseau est disponible.
# Installe les dépendances et télécharge TOUS les poids utilisés par le pipeline dans le cache Hugging Face.
set -euo pipefail
cd "$(dirname "$0")"
(apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-fra) > /dev/null 2>&1 || \
  (sudo apt-get -qq update && sudo apt-get -qq install -y tesseract-ocr tesseract-ocr-fra) > /dev/null 2>&1 || \
  echo "ATTENTION : tesseract (OCR des pages numérisées) non installé"
pip install -q -r requirements.txt sentence-transformers PyStemmer pytesseract "mistral-common[hf-hub]" huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "mistralai/Ministral-3-3B-Instruct-2512",   # générateur officiel
    "BAAI/bge-m3",                              # embeddings (recherche dense, graphe des éditions)
    "BAAI/bge-reranker-v2-m3",                  # reranker (cross-encoder)
]:
    snapshot_download(repo)
PY
python -c "import Stemmer, pytesseract, pymupdf, sentence_transformers; print('setup OK')"
