#!/usr/bin/env bash
# Run 6 : tableaux structurés (rag2/tables.py) + recherche par sous-question (rag2/search.py, search_multi).
#
#   Kaggle (GPU T4, Internet activé, Save Version -> Save & Run All) :
#     !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_tables.sh
#
# Deux configurations sur la validation (100 questions, juge officiel, abstention recalibrée) :
#   T  : variante duoT (duoL + tableaux structurés), recherche normale
#   TS : duoT + recherche par sous-question. Seules les questions dont la recherche change sont régénérées.
# Référence : run 4b (+0,465). Le leaderboard (200 questions) n'est généré et soumis que si on fait mieux.
set -euo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
BACKEND="${BACKEND:-hf}"; JUDGE="${JUDGE:-hf}"          # BACKEND=echo JUDGE=none : test sans GPU
REF_FILE="${REF_FILE:-results/run4b/abstention.json}"
EXTRA_INDEX="${EXTRA_INDEX:-}"; MODE_ARG="${MODE_ARG:-}"   # test sans GPU : EXTRA_INDEX=--no-dense MODE_ARG="--mode bm25"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
VAL=data/validation_v1.json

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

step "Recherche sur la validation : normale et par sous-question"
python -m rag2.pipeline retrieve --index-dir index2 --questions $VAL --strategy "$S" --out runs/val_ret.json $MODE_ARG
python -m rag2.pipeline retrieve --index-dir index2 --questions $VAL --strategy "$S" --out runs/val_ret_split.json --split $MODE_ARG

run_config() {   # $1 = nom, $2 = fichier de recherche
  step "   $1 : génération (100 questions)"
  [ -f "runs/val_raw_$1.json" ] || python -m rag2.pipeline generate --index-dir index2 --retrieval "$2" \
    --questions $VAL --out "runs/val_raw_$1.json" --backend "$BACKEND" --variant duoT
  step "   $1 : juge officiel"
  [ -f "runs/val_report_$1.json" ] || python -m groundtruth.evaluate --pred "runs/val_raw_$1.json" \
    --gold $VAL --judge "$JUDGE" --out "runs/val_report_$1.json"
  python -c "import json; r=json.load(open('runs/val_report_$1.json')); print('score brut', r['score_confiance'], r['issues'], '| recherche', r.get('retrieval'))"
  python -m rag2.calibrate fit --raw "runs/val_raw_$1.json" --report "runs/val_report_$1.json" \
    --gold $VAL --out "index2/abstention_$1.json"
}

run_config T runs/val_ret.json

# TS : on reprend les réponses de T pour les questions dont la recherche n'a pas changé
python - <<'PY'
import json
a = {it["qid"]: it for it in json.load(open("runs/val_ret.json"))["items"]}
b = json.load(open("runs/val_ret_split.json"))["items"]
done = {p["qid"]: p for p in json.load(open("runs/val_raw_T.json"))["predictions"]}
same = [it["qid"] for it in b if [h["idx"] for h in it["hits"]] == [h["idx"] for h in a[it["qid"]]["hits"]]]
with open("runs/val_raw_TS.json.partial.jsonl", "w", encoding="utf-8") as f:
    for q in same:
        f.write(json.dumps(done[q], ensure_ascii=False) + "\n")
print(f"recherche par sous-question : {len(b) - len(same)} questions changent sur {len(b)}")
PY
[ -f runs/val_raw_TS.json ] && rm runs/val_raw_TS.json.partial.jsonl
run_config TS runs/val_ret_split.json

step "Comparaison avec la référence (run 4b)"
BEST=$(python - "$REF_FILE" <<'PY'
import json, sys
ref = json.load(open(sys.argv[1]))["validation_score_calibre"]
sc = {c: json.load(open(f"index2/abstention_{c}.json"))["validation_score_calibre"] for c in ("T", "TS")}
print(f"  référence 4b {ref:+.3f} | T {sc['T']:+.3f} | TS {sc['TS']:+.3f}", file=sys.stderr)
best = max(sc, key=sc.get)
print(best if sc[best] > ref else "REF")
PY
)
echo "Retenu : $BEST"

step "Leaderboard v1 : 200 questions"
if [ "$BEST" = REF ]; then
  echo "Pas mieux que le run 4b : pas de génération du leaderboard, aucune soumission utilisée."
else
  SPLIT=""; [ "$BEST" = TS ] && SPLIT="--split"
  cp "index2/abstention_$BEST.json" index2/abstention.json
  python -m rag2.pipeline retrieve --index-dir index2 --questions data/classement_v1_questions.json --strategy "$S" \
    --out runs/cls_ret.json $SPLIT $MODE_ARG
  python -m rag2.pipeline generate --index-dir index2 --retrieval runs/cls_ret.json \
    --questions data/classement_v1_questions.json --out runs/cls_raw.json --backend "$BACKEND" --variant duoT
  python -m rag2.calibrate apply --raw runs/cls_raw.json --rule index2/abstention.json --out runs/classement_v1.json
  python -m groundtruth.submit --pred runs/classement_v1.json --phase v1 || \
    echo "Soumission non créée (quota ?) : runs/classement_v1.json est dans resultats.zip"
fi

step "Résumé"
python - "$OUT/resultats.zip" <<'PY'
import glob, json, os, sys, zipfile
raw = json.load(open("runs/val_raw_T.json"))["predictions"]
secs = [p["_meta"].get("seconds", 0) for p in raw]
print(f"temps moyen (T) : {sum(secs) / len(secs):.1f} s/question, max {max(secs):.1f} s")
print("extraits « tableau » ajoutés :", sum(p["_meta"].get("n_tables", 0) > 0 for p in raw), "questions")
files = glob.glob("runs/val_raw_*.json") + glob.glob("runs/val_report_*.json") + glob.glob("index2/abstention*.json") \
    + ["runs/cls_raw.json", "runs/classement_v1.json", "runs/val_ret.json", "runs/val_ret_split.json"] \
    + sorted(glob.glob("submissions/v1/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
print("->", sys.argv[1])
PY
