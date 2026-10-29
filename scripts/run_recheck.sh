#!/usr/bin/env bash
# Run 10 : 3e appel ciblé (6 phrases ou rangées de tableau les plus proches) ajouté aux réponses du run 9,
# pour les questions où les deux premières réponses divergent. ~20 min sur T4.
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_recheck.sh
#
# Les deux premières réponses étant déjà générées et jugées (results/run9/), seuls les 3e appels tournent ici :
# le score de validation se recalcule ensuite sans le juge (rag2.calibrate fit). Rien n'est soumis.
set -euo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
BACKEND="${BACKEND:-hf}"; EXTRA_INDEX="${EXTRA_INDEX:-}"
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
  step "$N : lignes les plus proches (reranker)"
  python -m rag2.recheck focus --index-dir index2 --raw "$R" --questions "$Q" --out "runs/focus_$N.json"
  step "$N : 3e appels"
  python -m rag2.recheck ask --raw "$R" --focus "runs/focus_$N.json" --questions "$Q" --out "runs/${N}_raw_F.json" --backend "$BACKEND"
done

python - "$OUT/resultats.zip" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in ["runs/focus_val.json", "runs/focus_cls.json", "runs/val_raw_F.json", "runs/cls_raw_F.json",
              "runs/generator_log.jsonl"]:
        z.write(f)
print("->", sys.argv[1])
PY
