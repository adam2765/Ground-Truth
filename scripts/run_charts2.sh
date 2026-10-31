#!/usr/bin/env bash
# Run 14 (~3 h, Kaggle T4 x2) : graphiques découpés et décrits par deux petits modèles de vision, puis validation.
#   1. Description des ~874 graphiques du corpus, en parallèle : Qwen3-VL-2B (GPU 0, lecture A) et Qwen3-VL-4B
#      (GPU 1, lecture B). Les temps mesurés disent si l'indexation tient dans 2 h sur un seul T4.
#   2. Validation avec la variante duoXRSG (1re réponse : lecture A ; 2e réponse : lecture B ; concordance).
#   3. Comparaison au run 12 (+0,525) ; leaderboard généré et soumis seulement si c'est mieux.
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_charts2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
JUDGE="${JUDGE:-hf}"; REF_FILE="${REF_FILE:-results/run12/abstention_R2.json}"; V=duoXRSG
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
step "Installation"
pip install -q -r requirements.txt sentence-transformers PyStemmer "mistral-common[hf-hub]" huggingface_hub 2>&1 | grep -v -i "warning\|dependency" || true
python corpus/download_corpus.py | tail -n 1
mkdir -p index2 runs

step "1. Description des graphiques (2B sur GPU 0, 4B sur GPU 1)"
CUDA_VISIBLE_DEVICES=0 python -m rag2.charts_vlm --model Qwen/Qwen3-VL-2B-Instruct --out runs/charts_2b.jsonl > runs/charts_2b.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m rag2.charts_vlm --model Qwen/Qwen3-VL-4B-Instruct --batch 4 --out runs/charts_4b.jsonl > runs/charts_4b.log 2>&1 &
wait
tail -n 2 runs/charts_2b.log runs/charts_4b.log
python -m rag2.charts_vlm --merge runs/charts_2b.jsonl --out index2/charts_a.json.gz
python -m rag2.charts_vlm --merge runs/charts_4b.jsonl --out index2/charts_b.json.gz
cp index2/charts_a.json.gz runs/charts_a.json.gz; cp index2/charts_b.json.gz runs/charts_b.json.gz

set -e
step "2. Index et recherche (validation)"
[ -f index2/pages.json ] || gunzip -c cache/pages_v1.json.gz > index2/pages.json
cp results/run1/best.json index2/best.json
S=$(python -c "import json; print(json.load(open('index2/best.json'))['strategy'])")
python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir index2 --strategies "$S"
python -m rag2.pipeline retrieve --index-dir index2 --questions data/validation_v1.json --strategy "$S" --out runs/val_ret.json --split
step "   génération $V (100 questions)"
python -m rag2.pipeline generate --index-dir index2 --retrieval runs/val_ret.json --questions data/validation_v1.json \
  --out runs/val_raw_G.json --variant $V
step "   juge officiel"
python -m groundtruth.evaluate --pred runs/val_raw_G.json --gold data/validation_v1.json --judge "$JUDGE" --out runs/val_report_G.json
python -c "import json; r=json.load(open('runs/val_report_G.json')); print('score brut', r['score_confiance'], r['issues']); print({t: v['score'] for t, v in r['par_type'].items()})"
python -m rag2.calibrate fit --raw runs/val_raw_G.json --report runs/val_report_G.json --gold data/validation_v1.json --out index2/abstention_G.json
BETTER=$(python - "$REF_FILE" <<'PY'
import json, sys
ref = json.load(open(sys.argv[1]))["validation_score_calibre"]
x = json.load(open("index2/abstention_G.json"))["validation_score_calibre"]
print(f"  référence run 12 {ref:+.3f} | graphiques {x:+.3f}", file=sys.stderr)
print(int(x > ref))
PY
)
step "3. Leaderboard"
if [ "$BETTER" = 1 ]; then
  python -m rag2.pipeline retrieve --index-dir index2 --questions data/classement_v1_questions.json --strategy "$S" --out runs/cls_ret.json --split
  python -m rag2.pipeline generate --index-dir index2 --retrieval runs/cls_ret.json --questions data/classement_v1_questions.json \
    --out runs/cls_raw_G.json --variant $V
  python -m rag2.calibrate apply --raw runs/cls_raw_G.json --rule index2/abstention_G.json --out runs/classement_v1.json
  python -m groundtruth.submit --pred runs/classement_v1.json --phase v1 || echo "Soumission non créée (quota ?)"
else
  echo "Pas mieux que le run 12 : aucune soumission."
fi
python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
files = glob.glob("runs/charts_*") + glob.glob("runs/val_*_G.json") + ["index2/abstention_G.json", "runs/cls_raw_G.json",
         "runs/classement_v1.json"] + sorted(glob.glob("submissions/v1/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f)
print("->", sys.argv[1])
PY
