#!/usr/bin/env bash
# Palier 2 de bout en bout, sur un GPU T4 (Kaggle ou Colab). Relançable : chaque étape déjà faite est sautée.
#
#   Kaggle (tourne en arrière-plan, même page fermée) :
#     !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_palier2.sh
#
# À la fin : resultats.zip et resume.txt dans $OUT (par défaut /kaggle/working, sinon le dossier du dépôt).
set -euo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
BACKEND="${BACKEND:-hf}"; JUDGE="${JUDGE:-hf}"          # BACKEND=echo JUDGE=none : test sans GPU
EXTRA_INDEX="${EXTRA_INDEX:-}"; ABL_MODES="${ABL_MODES:-bm25,dense,hybrid,hybrid_rerank}"; MODE_ARG="${MODE_ARG:-}"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }

step "GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || [ "$BACKEND" != hf ] || { echo "Pas de GPU : active un GPU T4."; exit 1; }

step "Installation"
(apt-get -qq update && apt-get -qq install -y tesseract-ocr tesseract-ocr-fra) > /dev/null 2>&1 || \
  echo "tesseract non installé (OCR déjà en cache, pas bloquant)"
pip install -q -r requirements.txt sentence-transformers PyStemmer pytesseract 2>&1 | grep -v -i "warning\|dependency" || true
python -c "import transformers, sentence_transformers; print('transformers', transformers.__version__, '| sentence-transformers', sentence_transformers.__version__)"

step "Corpus"
python corpus/download_corpus.py | tail -n 2

step "1. Indexation (OCR en cache, 4 découpages, embeddings bge-m3)"
mkdir -p index2 runs
[ -f index2/pages.json ] || gunzip -c cache/pages_v1.json.gz > index2/pages.json
python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir index2 $EXTRA_INDEX

step "2. Ablation de la recherche"
[ -f runs/ablation_retrieval.json ] || python -m rag2.ablation --index-dir index2 --questions data/validation_v1.json --modes "$ABL_MODES"
cat index2/best.json; echo

step "3. Validation : 100 questions"
python -m rag2.pipeline answer --index-dir index2 --questions data/validation_v1.json --out runs/val_raw.json --backend "$BACKEND" $MODE_ARG

step "   Notation par le juge officiel"
[ -f runs/val_report.json ] || python -m groundtruth.evaluate --pred runs/val_raw.json --gold data/validation_v1.json \
  --judge "$JUDGE" --out runs/val_report.json

step "   Calibration de l'abstention"
python -m rag2.calibrate fit --raw runs/val_raw.json --report runs/val_report.json --gold data/validation_v1.json \
  --out index2/abstention.json
python -m rag2.calibrate apply --raw runs/val_raw.json --rule index2/abstention.json --out runs/val_final.json

step "4. Leaderboard v1 : 200 questions"
python -m rag2.pipeline answer --index-dir index2 --questions data/classement_v1_questions.json --out runs/cls_raw.json --backend "$BACKEND" $MODE_ARG
python -m rag2.calibrate apply --raw runs/cls_raw.json --rule index2/abstention.json --out runs/classement_v1.json
ls submissions/v1/ 2>/dev/null | grep -q . || python -m groundtruth.submit --pred runs/classement_v1.json --phase v1

step "5. Résumé"
python scripts/summary.py | tee "$OUT/resume.txt"
python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
files = ["runs/ablation_retrieval.json", "runs/val_report.json", "runs/val_raw.json", "runs/cls_raw.json",
         "runs/classement_v1.json", "index2/abstention.json", "index2/best.json"] + glob.glob("submissions/v1/*")
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
PY
echo "Fichiers : $OUT/resultats.zip et $OUT/resume.txt"
