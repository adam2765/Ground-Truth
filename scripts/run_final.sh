#!/usr/bin/env bash
# Répétition de la finale avec le contrat officiel (setup.sh puis run.sh), sur un T4 (Kaggle T4 x2 : un seul GPU utilisé).
#   - avant le 2 novembre : index v1 chronométré + validation v1 (juge officiel)
#   - à partir du 2 novembre : en plus, update v2 chronométré, validation v2, leaderboard v2 et soumission
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_final.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
export CUDA_VISIBLE_DEVICES=0
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
chrono() { local t=$(date +%s); "$@"; local rc=$?; echo "  -> $(( ($(date +%s) - t) / 60 )) min $(( ($(date +%s) - t) % 60 )) s"; return $rc; }
V2=$(python -c "import datetime; print(int(datetime.date.today() >= datetime.date(2026, 11, 2)))")
mkdir -p runs final

step "setup.sh (réseau)"
export HF_HUB_DISABLE_XET=1
chrono timeout --foreground 2400 bash setup.sh < /dev/null 2>&1 | grep -v -E "Downloading|Reconstructing|Fetching|it/s" || true
python -c "import pymupdf, Stemmer, sentence_transformers" || { echo "setup.sh a échoué : arrêt (pas la peine d'occuper le GPU)"; exit 1; }
python corpus/download_corpus.py | tail -n 2

step "run.sh index (v1, limite 2 h)"
chrono timeout --foreground 9000 bash run.sh index corpus/pdf /tmp/index 2>&1 | grep -v -E "^  [0-9]*/|Batches|Loading weights|it/s"
[ -f /tmp/index/docs.json ] || { echo "run.sh index a échoué : arrêt"; exit 1; }
python -c "import json; d=json.load(open('/tmp/index/docs.json')); [print('  ', k, '->', v['newer']) for k, v in d.items() if v['newer']]"

if [ "$V2" = 1 ]; then
  step "run.sh update (v2, limite 20 min)"
  chrono timeout --foreground 2400 bash run.sh update /tmp/index corpus/pdf_v2 2>&1 | grep -v -E "Batches|Loading weights|it/s"
  python -c "import json; d=json.load(open('/tmp/index/docs.json')); [print('  ', k, '->', v['newer']) for k, v in d.items() if v['newer']]"
fi

for S in v1 $( [ "$V2" = 1 ] && echo v2 ); do
  step "run.sh answer : validation $S"
  chrono bash run.sh answer /tmp/index data/validation_$S.json runs/val_$S.json | tail -n 2
  python -m groundtruth.evaluate --pred runs/val_${S}_raw.json --gold data/validation_$S.json --judge hf --out runs/val_${S}_report_raw.json > /dev/null
  python -m groundtruth.evaluate --pred runs/val_$S.json --gold data/validation_$S.json --judge hf --out runs/val_${S}_report.json > /dev/null
  python -c "
import json
r = json.load(open('runs/val_${S}_report.json')); b = json.load(open('runs/val_${S}_report_raw.json'))
print('VALIDATION $S : score', r['score_confiance'], '(brut', b['score_confiance'], ')', r['issues'])
print('  par type :', {t: v['score'] for t, v in r['par_type'].items()})"
done
python -c "
import json, statistics
s = [p['_meta']['seconds'] for p in json.load(open('runs/val_v1_raw.json'))['predictions']]
print(f'temps de génération : {statistics.mean(s):.1f} s/question (max {max(s):.0f} s)')"

if [ "$V2" = 1 ]; then
  step "Leaderboard v2 (130 questions)"
  chrono bash run.sh answer /tmp/index data/classement_v2_questions.json runs/classement_v2.json | tail -n 2
  python -m groundtruth.submit --pred runs/classement_v2.json --phase v2 || echo "Soumission v2 non créée"
fi
cp runs/generator_log.jsonl runs/generator_log_final.jsonl 2>/dev/null || true
python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
files = glob.glob("runs/val_*") + glob.glob("runs/classement_v2*") + ["/tmp/index/docs.json"] \
    + sorted(glob.glob("submissions/v2/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f, os.path.basename(f) if f.startswith("/tmp") else f)
print("->", sys.argv[1])
PY
