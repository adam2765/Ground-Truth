#!/usr/bin/env bash
# Description des graphiques par un modèle de vision (Qwen2.5-VL-3B), sur les 2 GPU T4 de Kaggle. ~3-4 h.
#   Kaggle (Save & Run All, GPU T4 x2, Internet On) :
#     !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_charts.sh
# Résultat : charts_v1.json.gz dans /kaggle/working (à envoyer à Claude), plus les journaux.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }

step "GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

step "Installation"
pip install -q -r requirements.txt 2>&1 | grep -v -i "warning\|dependency" || true
python -c "import transformers; print('transformers', transformers.__version__)"

step "Corpus"
python corpus/download_corpus.py | tail -n 2
mkdir -p index2 runs
[ -f index2/pages.json ] || gunzip -c cache/pages_v1.json.gz > index2/pages.json
python -m rag2.describe_charts select --index-dir index2 --out runs/chart_pages.json

step "Description ($NGPU GPU en parallèle)"
PIDS=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m rag2.describe_charts run --pages runs/chart_pages.json \
    --shard "$i/$NGPU" --out "$OUT/charts_$i.jsonl" > "$OUT/charts_$i.log" 2>&1 &
  PIDS+=($!)
done
# avancement toutes les 5 min, tant qu'un des processus tourne
while kill -0 "${PIDS[@]}" 2>/dev/null || pgrep -f "rag2.describe_charts run" > /dev/null; do
  sleep 300; step "avancement"; for i in $(seq 0 $((NGPU - 1))); do tail -n 1 "$OUT/charts_$i.log" || true; done
done
wait || true
for i in $(seq 0 $((NGPU - 1))); do echo "--- GPU $i"; tail -n 3 "$OUT/charts_$i.log"; done

step "Fusion"
python -m rag2.describe_charts merge --inputs "$OUT"/charts_*.jsonl --out "$OUT/charts_v1.json.gz"
echo "Fichier à envoyer à Claude : $OUT/charts_v1.json.gz"
