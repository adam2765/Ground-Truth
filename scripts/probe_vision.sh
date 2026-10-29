#!/usr/bin/env bash
# Sonde vision (~30 min sur Kaggle T4 x2, ~1 h sur Colab T4) : deux modèles de vision plus forts que celui du run 4, en 4 bits, un par
# GPU, sur les pages des 12 questions « graphique » de la validation. Rien n'est soumis.
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/probe_vision.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
pip install -q -U bitsandbytes accelerate 2>&1 | grep -v -i "warning\|dependency" || true
python corpus/download_corpus.py | tail -n 1
mkdir -p runs
M1="${M1:-Qwen/Qwen2.5-VL-7B-Instruct}"; M2="${M2:-Qwen/Qwen3-VL-8B-Instruct}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NGPU" -ge 2 ]; then                 # Kaggle T4 x2 : un modèle par GPU, en parallèle
  CUDA_VISIBLE_DEVICES=0 python -m rag2.vision_probe --model "$M1" --out runs/vision_1.jsonl > runs/vision_1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -m rag2.vision_probe --model "$M2" --out runs/vision_2.jsonl > runs/vision_2.log 2>&1 &
  wait
else                                       # Colab (1 GPU) : l'un après l'autre
  python -m rag2.vision_probe --model "$M1" --out runs/vision_1.jsonl 2>&1 | tee runs/vision_1.log | grep -E "chargé|=====|Error"
  python -m rag2.vision_probe --model "$M2" --out runs/vision_2.jsonl 2>&1 | tee runs/vision_2.log | grep -E "chargé|=====|Error"
fi
tail -n 5 runs/vision_1.log runs/vision_2.log
python -m rag2.vision_probe --report runs/vision_1.jsonl runs/vision_2.jsonl | tee runs/vision_report.txt
python - "$OUT/resultats.zip" <<'PY'
import glob, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/vision_*"):
        z.write(f)
print("->", sys.argv[1])
PY
