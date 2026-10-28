#!/usr/bin/env bash
# Run 8 : deux variantes testées contre le run 7 (duoX, +0,475) :
#   duoXH : duoX + consigne de lecture des tableaux (rangée, puis colonne) pour la 1re réponse ;
#   duoC  : 2e réponse avec les mêmes tableaux présentés colonne par colonne (au lieu de sans tableau).
#
#   Kaggle (GPU T4, Internet activé, Save Version -> Save & Run All) :
#     !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_opti.sh
#
# Validation réelle (100 questions, juge officiel), comparaison au run 7 (+0,475), puis leaderboard
# (200 questions) et soumission seulement si l'une fait mieux que le run 7.
set -euo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
BACKEND="${BACKEND:-hf}"; JUDGE="${JUDGE:-hf}"          # BACKEND=echo JUDGE=none : test sans GPU
REF_FILE="${REF_FILE:-results/run7/abstention_X.json}"
EXTRA_INDEX="${EXTRA_INDEX:-}"; MODE_ARG="${MODE_ARG:-}"   # test sans GPU : EXTRA_INDEX=--no-dense MODE_ARG="--mode bm25"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
VAL=data/validation_v1.json; VARIANTS="${VARIANTS:-duoXH duoC}"

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
for C in $VARIANTS; do
  step "Validation : génération $C (100 questions)"
  [ -f "runs/val_raw_$C.json" ] || python -m rag2.pipeline generate --index-dir index2 --retrieval runs/val_ret_split.json \
    --questions $VAL --out "runs/val_raw_$C.json" --backend "$BACKEND" --variant "$C"
  step "Validation : juge officiel ($C)"
  [ -f "runs/val_report_$C.json" ] || python -m groundtruth.evaluate --pred "runs/val_raw_$C.json" \
    --gold $VAL --judge "$JUDGE" --out "runs/val_report_$C.json"
  python -c "import json; r=json.load(open('runs/val_report_$C.json')); print('score brut', r['score_confiance'], r['issues'])"
  python -m rag2.calibrate fit --raw "runs/val_raw_$C.json" --report "runs/val_report_$C.json" --gold $VAL \
    --out "index2/abstention_$C.json"
done

V=$(python - "$REF_FILE" $VARIANTS <<'PY2'
import json, sys
ref = json.load(open(sys.argv[1]))["validation_score_calibre"]
sc = {c: json.load(open(f"index2/abstention_{c}.json"))["validation_score_calibre"] for c in sys.argv[2:]}
print(f"  référence run 7 {ref:+.3f} | " + " | ".join(f"{c} {s:+.3f}" for c, s in sc.items()), file=sys.stderr)
best = max(sc, key=sc.get)
print(best if sc[best] > ref else "REF")
PY2
)
echo "Retenu : $V"

step "Leaderboard v1 : 200 questions"
if [ "$V" = REF ]; then
  echo "Pas mieux que le run 7 : pas de génération du leaderboard, aucune soumission utilisée."
else
  cp "index2/abstention_$V.json" index2/abstention.json
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
for f in sorted(glob.glob("runs/val_raw_*.json")):
    raw = json.load(open(f))["predictions"]
    secs = [p["_meta"].get("seconds", 0) for p in raw]
    print(f"{f} : temps moyen {sum(secs) / len(secs):.1f} s/question, max {max(secs):.1f} s")
files = glob.glob("runs/val_raw_*.json") + glob.glob("runs/val_report_*.json") + glob.glob("index2/abstention_*.json") \
    + ["runs/val_ret_split.json", "runs/cls_raw.json", "runs/classement_v1.json"] + sorted(glob.glob("submissions/v1/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
print("->", sys.argv[1])
PY
