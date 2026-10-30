#!/usr/bin/env bash
# Run 12 : « repêchage » des questions où le modèle a répondu INTROUVABLE alors qu'un tableau est dans le contexte.
# On repose la question sur les 6 rangées de tableau les plus proches, puis on vérifie par une 2e lecture
# indépendante (format citation, extraits sans tableaux). Appliqué aux réponses du run 9 (~45 min).
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_rescue.sh
#
# Validation (juge officiel) comparée au run 9 (+0,505) ; leaderboard soumis seulement si c'est mieux.
set -euo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
BACKEND="${BACKEND:-hf}"; JUDGE="${JUDGE:-hf}"; EXTRA_INDEX="${EXTRA_INDEX:-}"
REF_FILE="${REF_FILE:-results/run9/abstention_duoXR_rows.json}"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }

step "Installation"
[ "$BACKEND" = hf ] && pip install -q -r requirements.txt sentence-transformers PyStemmer "mistral-common[hf-hub]" huggingface_hub 2>&1 | grep -v -i "warning\|dependency" || true
step "Corpus et index"
python corpus/download_corpus.py | tail -n 2
mkdir -p index2 runs
[ -f index2/pages.json ] || gunzip -c cache/pages_v1.json.gz > index2/pages.json
cp results/run1/best.json index2/best.json
S=$(python -c "import json; print(json.load(open('index2/best.json'))['strategy'])")
python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir index2 --strategies "$S" $EXTRA_INDEX

for P in "val:data/validation_v1.json:results/run9/val_raw_duoXR.json" "cls:data/classement_v1_questions.json:results/run9/cls_raw_duoXR.json"; do
  IFS=: read -r N Q R <<< "$P"
  step "$N : rangées de tableau (reranker)"
  python -m rag2.recheck rescue_rows --index-dir index2 --raw "$R" --questions "$Q" --out "runs/rescue_$N.json"
  step "$N : repêchage"
  python -m rag2.recheck rescue --index-dir index2 --raw "$R" --focus "runs/rescue_$N.json" --questions "$Q" \
    --out "runs/${N}_raw_R.json" --backend "$BACKEND"
done

step "Validation : juge officiel"
python -m groundtruth.evaluate --pred runs/val_raw_R.json --gold data/validation_v1.json --judge "$JUDGE" --out runs/val_report_R.json
python -c "import json; r=json.load(open('runs/val_report_R.json')); print('score brut', r['score_confiance'], r['issues'])"
python -m rag2.calibrate fit --raw runs/val_raw_R.json --report runs/val_report_R.json --gold data/validation_v1.json \
  --out index2/abstention_R.json
BETTER=$(python - "$REF_FILE" <<'PY'
import json, sys
ref = json.load(open(sys.argv[1]))["validation_score_calibre"]
x = json.load(open("index2/abstention_R.json"))["validation_score_calibre"]
print(f"  référence run 9 {ref:+.3f} | repêchage {x:+.3f}", file=sys.stderr)
print(int(x > ref))
PY
)
step "Leaderboard"
python -m rag2.calibrate apply --raw runs/cls_raw_R.json --rule index2/abstention_R.json --out runs/classement_v1.json
if [ "$BETTER" = 1 ]; then
  python -m groundtruth.submit --pred runs/classement_v1.json --phase v1 || echo "Soumission non créée (quota ?)"
else
  echo "Pas mieux que le run 9 : aucune soumission."
fi
python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
files = glob.glob("runs/rescue_*.json") + ["runs/val_raw_R.json", "runs/cls_raw_R.json", "runs/val_report_R.json",
         "index2/abstention_R.json", "runs/classement_v1.json"] + sorted(glob.glob("submissions/v1/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
print("->", sys.argv[1])
PY
