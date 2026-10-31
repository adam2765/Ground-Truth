#!/usr/bin/env bash
# Sonde vision n° 4 (~30 min, Kaggle T4 x2) : peut-on décrire les ~874 graphiques du corpus en moins de 2 h ?
# Graphiques découpés, modèles plus petits de la famille Qwen3-VL, en float16, réponses plus courtes (350 tokens) :
#   GPU 0 : Qwen3-VL-4B-Instruct        GPU 1 : Qwen3-VL-2B-Instruct
# Objectif : <= 7 s par graphique avec la précision du 8B (6/6 valeurs chiffrées sur les pages découpées).
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/probe_small.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
pip install -q -r requirements.txt 2>&1 | grep -v -i "warning\|dependency" || true
pip install -q -U bitsandbytes accelerate 2>&1 | grep -v -i "warning\|dependency" || true
python corpus/download_corpus.py | tail -n 1
mkdir -p runs
M1="${M1:-Qwen/Qwen3-VL-4B-Instruct}"; M2="${M2:-Qwen/Qwen3-VL-2B-Instruct}"
CUDA_VISIBLE_DEVICES=0 python -m rag2.vision_probe --model "$M1" --crops --no-quant --max-new 350 --dpi 130 \
  --out runs/vision_small_1.jsonl > runs/vision_small_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m rag2.vision_probe --model "$M2" --crops --no-quant --max-new 350 --dpi 130 \
  --out runs/vision_small_2.jsonl > runs/vision_small_2.log 2>&1 &
wait
grep -h -E "chargé|Error|Traceback" runs/vision_small_*.log | head
python -m rag2.vision_probe --report runs/vision_small_*.jsonl | tee runs/vision_report_small.txt
python - "$OUT/resultats.zip" <<'PY'
import glob, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/vision_*small*"):
        z.write(f)
print("->", sys.argv[1])
PY
