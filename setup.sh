#!/usr/bin/env bash
# Phase d'installation de la finale : seule phase où le réseau est disponible.
# Installe les dépendances et télécharge TOUS les poids utilisés par le pipeline dans le cache Hugging Face.
set -euo pipefail
cd "$(dirname "$0")"
export DEBIAN_FRONTEND=noninteractive
# 1. dépendances Python d'abord (indispensables)
pip install -q -r requirements.txt sentence-transformers PyStemmer pytesseract "mistral-common[hf-hub]" huggingface_hub < /dev/null
# 2. Tesseract (OCR des pages numérisées), sans aucune question interactive et avec une limite de temps
apt_install() {
  timeout --foreground 600 $1 apt-get -qq -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold \
    install -y --no-install-recommends tesseract-ocr tesseract-ocr-fra < /dev/null > /dev/null 2>&1
}
command -v tesseract > /dev/null || { timeout --foreground 300 apt-get -qq update < /dev/null > /dev/null 2>&1; apt_install ""; } || \
  { timeout --foreground 300 sudo apt-get -qq update < /dev/null > /dev/null 2>&1; apt_install sudo; } || true
command -v tesseract > /dev/null && echo "tesseract OK" || echo "ATTENTION : tesseract non installé (OCR des pages numérisées impossible)"
# Téléchargement classique (le nouveau protocole « xet » est resté bloqué une nuit sur Kaggle, le 2/11),
# et seulement les fichiers utilisés : pas les doublons « consolidated » (format mistral-inference) ni ONNX.
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
python - <<'PY'
import time
from huggingface_hub import snapshot_download
IGNORE = ["consolidated*", "*.gguf", "onnx/*", "*.onnx", "*.msgpack", "*.h5", "colbert*", "sparse*"]
for repo in [
    "mistralai/Ministral-3-3B-Instruct-2512",   # générateur officiel
    "BAAI/bge-m3",                              # embeddings (recherche dense, graphe des éditions)
    "BAAI/bge-reranker-v2-m3",                  # reranker (cross-encoder)
]:
    for attempt in range(3):
        try:
            t = time.time()
            snapshot_download(repo, ignore_patterns=IGNORE)
            print(f"{repo} : {time.time() - t:.0f} s", flush=True)
            break
        except Exception as e:                  # coupure réseau : on reprend (les fichiers déjà là sont gardés)
            print(f"{repo} : essai {attempt + 1} échoué ({e})", flush=True)
PY
python -c "import Stemmer, pytesseract, pymupdf, sentence_transformers; print('setup OK')"
