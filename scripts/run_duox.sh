#!/usr/bin/env bash
# Run 7 : variante duoX (1re réponse avec les tableaux structurés, 2e réponse sans : deux lectures différentes
# du contexte, dont les erreurs coïncident moins) + recherche par sous-question.
# Simulation à partir des runs 4 et 6 : +0,505 sur la validation (results/run6/abstention_duoX.json).
#
#   Kaggle (GPU T4, Internet activé, Save Version -> Save & Run All) :
#     !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_duox.sh
#
# Validation réelle (100 questions, juge officiel), comparaison au run 4b (+0,465), puis leaderboard
# (200 questions) et soumission seulement si c'est mieux.
set -euo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
BACKEND="${BACKEND:-hf}"; JUDGE="${JUDGE:-hf}"          # BACKEND=echo JUDGE=none : test sans GPU
REF_FILE="${REF_FILE:-results/run4b/abstention.json}"
EXTRA_INDEX="${EXTRA_INDEX:-}"; MODE_ARG="${MODE_ARG:-}"   # test sans GPU : EXTRA_INDEX=--no-dense MODE_ARG="--mode bm25"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
VAL=data/validation_v1.json; V=duoX

step "GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || [ "$BACKEND" != hf ] || { echo "Pas de GPU : active un GPU T4."; exit 1; }

step "Installation"
[ "$BACKEND" = hf ] && pip install -q -r requirements.txt sentence-transformers PyStemmer "mistral-common[hf-hub]" huggingface_hub 2>&1 | grep -v -i "warning\|dependency" || true

step "Corpus"
python corpus/download_corpus.py | tail -n 2

step "Indexation (OCR et tableaux en cache, embeddings bge-m3)"
mkdir -p index2 runs
[ -f index2/pages.json ] || gunzip -c cache/pages_v1.json.gz > index2/pages.json
cp results/run1/best.json index2/best.json
S=$(python -c "import json; print(json.load(open('index2/best.json'))['strategy'])")
python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir index2 --strategies "$S" $EXTRA_INDEX

step "Validation : recherche par sous-question"
python -m rag2.pipeline retrieve --index-dir index2 --questions $VAL --strategy "$S" --out runs/val_ret_split.json --split $MODE_ARG
step "Validation : génération $V (100 questions)"
[ -f runs/val_raw_X.json ] || python -m rag2.pipeline generate --index-dir index2 --retrieval runs/val_ret_split.json \
  --questions $VAL --out runs/val_raw_X.json --backend "$BACKEND" --variant $V
step "Validation : juge officiel"
[ -f runs/val_report_X.json ] || python -m groundtruth.evaluate --pred runs/val_raw_X.json \
  --gold $VAL --judge "$JUDGE" --out runs/val_report_X.json
python -c "import json; r=json.load(open('runs/val_report_X.json')); print('score brut', r['score_confiance'], r['issues'], '| recherche', r.get('retrieval'))"
python -m rag2.calibrate fit --raw runs/val_raw_X.json --report runs/val_report_X.json --gold $VAL --out index2/abstention_X.json

BETTER=$(python - "$REF_FILE" <<'PY'
import json, sys
ref = json.load(open(sys.argv[1]))["validation_score_calibre"]
x = json.load(open("index2/abstention_X.json"))["validation_score_calibre"]
print(f"  référence 4b {ref:+.3f} | duoX {x:+.3f}", file=sys.stderr)
print(int(x > ref))
PY
)

step "Leaderboard v1 : 200 questions"
if [ "$BETTER" != 1 ]; then
  echo "Pas mieux que le run 4b : pas de génération du leaderboard, aucune soumission utilisée."
else
  cp index2/abstention_X.json index2/abstention.json
  python -m rag2.pipeline retrieve --index-dir index2 --questions data/classement_v1_questions.json --strategy "$S" \
    --out runs/cls_ret.json --split $MODE_ARG
  python -m rag2.pipeline generate --index-dir index2 --retrieval runs/cls_ret.json \
    --questions data/classement_v1_questions.json --out runs/cls_raw.json --backend "$BACKEND" --variant $V
  python -m rag2.calibrate apply --raw runs/cls_raw.json --rule index2/abstention.json --out runs/classement_v1.json
  python -m groundtruth.submit --pred runs/classement_v1.json --phase v1 || \
    echo "Soumission non créée (quota ?) : runs/classement_v1.json est dans resultats.zip"
fi

step "Résumé"
python - "$OUT/resultats.zip" <<'PY'
import glob, json, os, sys, zipfile
raw = json.load(open("runs/val_raw_X.json"))["predictions"]
secs = [p["_meta"].get("seconds", 0) for p in raw]
print(f"temps moyen : {sum(secs) / len(secs):.1f} s/question, max {max(secs):.1f} s")
files = ["runs/val_raw_X.json", "runs/val_report_X.json", "index2/abstention_X.json", "runs/val_ret_split.json",
         "runs/cls_raw.json", "runs/classement_v1.json"] + sorted(glob.glob("submissions/v1/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
print("->", sys.argv[1])
PY
