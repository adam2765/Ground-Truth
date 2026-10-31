#!/usr/bin/env bash
# Run 13 (~40 min, Kaggle T4 x2) : les pistes de la littérature récente, en parallèle. Rien n'est soumis.
#   GPU 0 : IBM Granite 4.0 3B Vision (ChartNet) sur les pages des questions « graphique » (sonde vision n° 2)
#   GPU 1 : vérificateur NLI indépendant (mDeBERTa-v3 XNLI) sur les réponses du run 12 (validation + leaderboard)
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_tests.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
pip install -q -r requirements.txt sentence-transformers PyStemmer sentencepiece protobuf 2>&1 | grep -v -i "warning\|dependency" || true
python corpus/download_corpus.py | tail -n 1
mkdir -p index2 runs
[ -f index2/pages.json ] || gunzip -c cache/pages_v1.json.gz > index2/pages.json
cp results/run1/best.json index2/best.json
S=$(python -c "import json; print(json.load(open('index2/best.json'))['strategy'])")

# GPU 1 : index puis NLI (dans l'environnement actuel)
( export CUDA_VISIBLE_DEVICES=1
  python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir index2 --strategies "$S" &&
  python -m rag2.nli --index-dir index2 --raw results/run12/val_raw_R2.json --questions data/validation_v1.json --out runs/nli_val.json &&
  python -m rag2.nli --index-dir index2 --raw results/run12/cls_raw_R2.json --questions data/classement_v1_questions.json --out runs/nli_cls.json
) > runs/nli.log 2>&1 &

# GPU 0 : Granite Vision, dans un environnement séparé (il demande transformers 4.57.6)
( export CUDA_VISIBLE_DEVICES=0
  python -m venv --system-site-packages /tmp/venv_granite && . /tmp/venv_granite/bin/activate &&
  pip install -q "transformers==4.57.6" "peft==0.18.1" 2>&1 | grep -v -i "warning\|dependency"
  python -m rag2.vision_probe --model ibm-granite/granite-4.0-3b-vision --out runs/vision_granite.jsonl
) > runs/vision_granite.log 2>&1 &

wait
echo "=== NLI"; tail -n 4 runs/nli.log
echo "=== Granite"; grep -E "chargé|Error|Traceback" runs/vision_granite.log | head -5
python -m rag2.vision_probe --report runs/vision_granite.jsonl | tee runs/vision_report.txt
python - "$OUT/resultats.zip" <<'PY'
import glob, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/nli*") + glob.glob("runs/vision_*"):
        z.write(f)
print("->", sys.argv[1])
PY
