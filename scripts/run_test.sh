#!/usr/bin/env bash
# Run final officiel (14 nov. 18 h -> 15 nov. 23 h 59), en une seule commande (faisable depuis un téléphone).
# Code : le commit tagué « final » ; questions : publiées par le mentor sur main après le tag.
# Produit /kaggle/working/final.zip (final/predictions_test.json, final/predictions_redteam.json,
# final/generator_log.jsonl, log) à envoyer pour qu'il soit poussé dans final/.
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_test.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
export CUDA_VISIBLE_DEVICES=0
exec > >(tee "$OUT/log_test.txt") 2>&1
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
git fetch -q --tags origin
MAIN=$(pwd)
# le code du tag « final » dans un dossier à part (ce script continue de tourner depuis main)
rm -rf /tmp/gt_final; git worktree add -q /tmp/gt_final final || { echo "tag « final » introuvable"; exit 1; }
cd /tmp/gt_final
for f in data/test_questions.json data/redteam_questions.json; do     # publiés par le mentor sur main après le tag
  git checkout -q origin/main -- "$f" 2>/dev/null || true
done
for f in data/test_questions.json data/redteam_questions.json; do
  [ -f "$f" ] && echo "$f : $(python -c "import json; d=json.load(open('$f')); print(len(d['questions'] if isinstance(d, dict) else d))") questions" \
    || echo "ATTENTION : $f absent (pas encore publié ?)"
done
echo "code : $(git rev-parse --short HEAD) (tag final)"
step "setup.sh"
timeout --foreground 2400 bash setup.sh < /dev/null 2>&1 | grep -v -E "Downloading|Reconstructing|Fetching|it/s" | tail -n 6
python corpus/download_corpus.py | tail -n 1
rm -f runs/generator_log.jsonl; mkdir -p final
step "run.sh index (v1)"
bash run.sh index corpus/pdf /tmp/index 2>&1 | grep -E "terminée|tableaux des scans|transcription des scans|Error|Traceback"
step "run.sh update (v2)"
bash run.sh update /tmp/index corpus/pdf_v2 2>&1 | grep -E "terminée|nouveaux documents|Error|Traceback"
for N in test redteam; do
  [ -f data/${N}_questions.json ] || continue
  step "run.sh answer : $N"
  bash run.sh answer /tmp/index data/${N}_questions.json final/predictions_$N.json 2>&1 | grep -E "réponses|Error|Traceback"
done
cp runs/generator_log.jsonl final/generator_log.jsonl
python - <<'PY'
import collections, json
c = collections.Counter(json.loads(l)["qid"] for l in open("final/generator_log.jsonl"))
print(f"journal du générateur : {sum(c.values())} appels, {len(c)} questions, max {max(c.values())} appels/question")
PY
step "terminé"
cp "$OUT/log_test.txt" final/log_test.txt
python - "$OUT/final.zip" <<'PY'
import glob, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("final/*"):
        z.write(f)
print("->", sys.argv[1])
PY
