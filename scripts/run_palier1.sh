#!/usr/bin/env bash
# Palier 1 : résultats du code from scratch (my_rag/) sur validation_v1, sur GPU T4. ~40 min.
#   Kaggle (Save & Run All, GPU T4 x2, Internet On) :
#     !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_palier1.sh
# Résultat : palier1.zip et palier1.txt dans /kaggle/working.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
BACKEND="${BACKEND:-hf}"; JUDGE="${JUDGE:-hf}"; SKIP_DENSE="${SKIP_DENSE:-0}"; RAG_ONLY="${RAG_ONLY:-0}"
mkdir -p runs
{
echo "=== Installation"
pip install -q -r requirements.txt sentence-transformers "mistral-common[hf-hub]" huggingface_hub 2>&1 | grep -v -i "warning\|dependency" || true
python -c "import huggingface_hub, transformers; print('huggingface_hub', huggingface_hub.__version__, '| transformers', transformers.__version__)"
echo "=== Corpus"
python corpus/download_corpus.py | tail -n 2

if [ "$RAG_ONLY" != 1 ]; then
echo "=== Vérification du BM25 maison contre bm25s, et des métriques contre l'évaluateur officiel"
python -m my_rag.check_bm25

echo "=== Recherche : BM25 maison seul"
python -m my_rag.eval_retrieval --method bm25
fi
if [ "$SKIP_DENSE" != 1 ] && [ "$RAG_ONLY" != 1 ]; then
  echo "=== Recherche : dense maison seul (multilingual-e5-small, cosinus en NumPy)"
  python -m my_rag.eval_retrieval --method dense
fi

echo "=== RAG de base (BM25 maison + générateur officiel, 1 appel) sur les 100 questions"
python -m my_rag.rag --questions data/validation_v1.json --out runs/rag_val1.json --backend "$BACKEND" --method bm25
echo "=== Notation par le juge officiel"
python -m groundtruth.evaluate --pred runs/rag_val1.json --gold data/validation_v1.json --judge "$JUDGE" \
  --out runs/rag_val1_report.json
} 2>&1 | grep -v "Loading weights\|Fetching\|downloading bytes\|reconstructing\|it/s\]" | tee "$OUT/palier1.txt"

python - "$OUT/palier1.zip" <<'PY'
import glob, os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/retrieval_*.json") + ["runs/rag_val1.json", "runs/rag_val1_report.json"]:
        if os.path.exists(f):
            z.write(f)
PY
echo "Fichiers : $OUT/palier1.zip et $OUT/palier1.txt"
