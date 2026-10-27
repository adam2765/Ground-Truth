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
# L'ablation complète (4 découpages x 4 recherches) a été faite au run 1 (results/run1/) : sauf FULL_ABLATION=1,
# on réutilise sa meilleure configuration et on n'indexe que ce découpage (~15 min gagnées).
if [ "${FULL_ABLATION:-0}" != 1 ] && [ -f results/run1/best.json ]; then
  cp results/run1/best.json index2/best.json
  cp results/run1/ablation_retrieval.json runs/ablation_retrieval.json
  STRATS=$(python -c "import json; print(json.load(open('index2/best.json'))['strategy'])")
  # run 4 : le même découpage, avec et sans la description des graphiques (cache/charts_v1.json.gz)
  [ -f cache/charts_v1.json.gz ] && STRATS="$STRATS,$STRATS+graph"
  python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir index2 --strategies "$STRATS" $EXTRA_INDEX
else
  python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir index2 $EXTRA_INDEX
  step "2. Ablation de la recherche"
  [ -f runs/ablation_retrieval.json ] || python -m rag2.ablation --index-dir index2 --questions data/validation_v1.json --modes "$ABL_MODES"
fi
cat index2/best.json; echo

step "3. Validation : chaque configuration (index x variante de génération)"
VARIANTS="${VARIANTS:-duoL}"
CONFIGS=""
for S in ${STRATS//,/ }; do
  python -m rag2.pipeline retrieve --index-dir index2 --questions data/validation_v1.json --strategy "$S" \
    --out "runs/val_retrieval_$S.json" $MODE_ARG
  for V in $VARIANTS; do CONFIGS="$CONFIGS $S@$V"; done
done
for C in $CONFIGS; do
  S=${C%@*}; V=${C#*@}
  step "   $C : 100 questions"
  [ -f "runs/val_raw_$C.json" ] || python -m rag2.pipeline generate --index-dir index2 --retrieval "runs/val_retrieval_$S.json" \
    --questions data/validation_v1.json --out "runs/val_raw_$C.json" --backend "$BACKEND" --variant "$V"
  step "   $C : notation par le juge officiel"
  [ -f "runs/val_report_$C.json" ] || python -m groundtruth.evaluate --pred "runs/val_raw_$C.json" \
    --gold data/validation_v1.json --judge "$JUDGE" --out "runs/val_report_$C.json"
  python -c "import json; r=json.load(open('runs/val_report_$C.json')); print('score brut', r['score_confiance'], r['issues'])"
  python -m rag2.calibrate fit --raw "runs/val_raw_$C.json" --report "runs/val_report_$C.json" \
    --gold data/validation_v1.json --out "index2/abstention_$C.json"
done
BEST=$(python - $CONFIGS <<'PY'
import json, sys
scores = {c: json.load(open(f"index2/abstention_{c}.json"))["validation_score_calibre"] for c in sys.argv[1:]}
for c, s in scores.items():
    print(f"{c}: {s:+.3f}", file=sys.stderr)
print(max(scores, key=scores.get))
PY
)
BS=${BEST%@*}; BV=${BEST#*@}
echo "Configuration retenue : $BEST"
cp "index2/abstention_$BEST.json" index2/abstention.json
cp "runs/val_raw_$BEST.json" runs/val_raw.json
cp "runs/val_report_$BEST.json" runs/val_report.json
python -m rag2.calibrate apply --raw runs/val_raw.json --rule index2/abstention.json --out runs/val_final.json

step "4. Leaderboard v1 : 200 questions ($BEST)"
python -m rag2.pipeline retrieve --index-dir index2 --questions data/classement_v1_questions.json --strategy "$BS" \
  --out runs/cls_retrieval.json $MODE_ARG
python -m rag2.pipeline generate --index-dir index2 --retrieval runs/cls_retrieval.json \
  --questions data/classement_v1_questions.json --out runs/cls_raw.json --backend "$BACKEND" --variant "$BV"
python -m rag2.calibrate apply --raw runs/cls_raw.json --rule index2/abstention.json --out runs/classement_v1.json
python -m groundtruth.submit --pred runs/classement_v1.json --phase v1 || \
  echo "Soumission non créée (quota de 2 par jour atteint ?) : runs/classement_v1.json est dans resultats.zip"

step "5. Résumé"
python scripts/summary.py | tee "$OUT/resume.txt"
python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
files = ["runs/ablation_retrieval.json", "runs/val_report.json", "runs/val_raw.json", "runs/cls_raw.json",
         "runs/classement_v1.json", "index2/abstention.json", "index2/best.json"] + \
        glob.glob("runs/val_raw_*.json") + glob.glob("runs/val_report_*.json") + glob.glob("index2/abstention_*.json") + \
        sorted(glob.glob("submissions/v1/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
PY
echo "Fichiers : $OUT/resultats.zip et $OUT/resume.txt"
