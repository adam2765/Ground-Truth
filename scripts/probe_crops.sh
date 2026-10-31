#!/usr/bin/env bash
# Sonde vision n° 3 (~30 min, Kaggle T4 x2) : les graphiques sont d'abord DÉCOUPÉS (rag2/chart_crops.py), puis lus
#   GPU 0 : IBM Granite 4.0 3B Vision (<chart2csv>, conçu pour un graphique seul)
#   GPU 1 : Qwen3-VL-8B (4 bits), meilleur modèle de la sonde n° 1 sur pages entières (4/7)
# Si les deux lisent bien, on pourra les faire concorder (deux lectures indépendantes d'un même graphique).
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/probe_crops.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
pip install -q -r requirements.txt 2>&1 | grep -v -i "warning\|dependency" || true
pip install -q -U bitsandbytes accelerate 2>&1 | grep -v -i "warning\|dependency" || true
python corpus/download_corpus.py | tail -n 1
mkdir -p runs
( export CUDA_VISIBLE_DEVICES=0
  pip install -q --target /tmp/granite_libs "transformers==4.57.6" "peft==0.18.1" "tokenizers==0.22.2" "huggingface_hub>=0.34,<1.0" --no-deps 2>&1 | grep -v -i "warning\|dependency"
  PYTHONPATH=/tmp/granite_libs python -m rag2.vision_probe --model ibm-granite/granite-4.0-3b-vision --crops --out runs/vision_granite_crops.jsonl
) > runs/vision_granite_crops.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1
  python -m rag2.vision_probe --model Qwen/Qwen3-VL-8B-Instruct --crops --out runs/vision_qwen3_crops.jsonl
) > runs/vision_qwen3_crops.log 2>&1 &
wait
grep -h -E "chargé|Error|Traceback" runs/vision_*_crops.log | head
python -m rag2.vision_probe --report runs/vision_*_crops.jsonl | tee runs/vision_report_crops.txt
python - "$OUT/resultats.zip" <<'PY'
import glob, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/vision_*crops*"):
        z.write(f)
print("->", sys.argv[1])
PY
