#!/usr/bin/env bash
# Palier 3 : optimisation v2 (~3 h 30, Kaggle T4 x2, un seul GPU utilisé).
#  - index v1 (caches OCR/tableaux pour aller vite) + run.sh-équivalent « update » v2 (calcul réel)
#  - recherche avec filtre VersionRAG (extraits des éditions remplacées retirés, sauf question d'évolution)
#  - 2 variantes : duoXRS (actuelle) et duoXRSf (2e lecture en format court : plus rapide)
#  - validation v1 + v2 (juge officiel), règle d'abstention choisie sur v1 + v2, leaderboard v2 et soumission
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_v2opti.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
VARIANTS="${VARIANTS:-duoXRS duoXRSf}"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
chrono() { local t=$(date +%s); "$@"; local rc=$?; echo "  -> $(( ($(date +%s) - t) / 60 )) min $(( ($(date +%s) - t) % 60 )) s"; return $rc; }
step "setup"
chrono timeout --foreground 2400 bash setup.sh < /dev/null 2>&1 | grep -v -E "Downloading|Reconstructing|Fetching|it/s" || true
python -c "import pymupdf, Stemmer, sentence_transformers" || { echo "setup.sh a échoué"; exit 1; }
python corpus/download_corpus.py | tail -n 1
set -e
I=/tmp/index; mkdir -p $I runs
step "index v1 (caches) + update v2"
gunzip -c cache/pages_v1.json.gz > $I/pages.json
echo '{"strategy": "naive_title", "mode": "hybrid_rerank"}' > $I/best.json
python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir $I --strategies naive_title 2>&1 | grep -v -E "Batches|it/s"
chrono python -m rag2.pipeline update --index-dir $I --pdf-dir corpus/pdf_v2 2>&1 | grep -v -E "Batches|it/s"
python -c "import json; d=json.load(open('$I/docs.json')); [print('  ', k, '->', v['newer']) for k, v in d.items() if v['newer']]"

for S in v1 v2; do
  python -m rag2.pipeline retrieve --index-dir $I --questions data/validation_$S.json --out runs/ret_$S.json --split
done
for V in $VARIANTS; do
  for S in v1 v2; do
    step "$V : validation $S"
    python -m rag2.pipeline generate --index-dir $I --retrieval runs/ret_$S.json --questions data/validation_$S.json \
      --out runs/raw_${V}_$S.json --variant $V
    python -m groundtruth.evaluate --pred runs/raw_${V}_$S.json --gold data/validation_$S.json --judge hf \
      --out runs/rep_${V}_$S.json > /dev/null
  done
done

python - $VARIANTS <<'EN'
import json, sys
g = json.load(open("data/validation_v2.json")); g = g["questions"] if isinstance(g, dict) else g
en = [q for q in g if q.get("lang") == "en"]
for V in sys.argv[1:]:
    r = {d["qid"]: d["points"] for d in json.load(open(f"runs/rep_{V}_v2.json"))["details"]}
    print(f"{V} : questions en anglais, score brut {sum(r[q['qid']] for q in en) / len(en):+.3f} sur {len(en)}")
EN
step "Choix de la variante et de la règle (validation v1 + v2)"
python - $VARIANTS <<'PY'
import json, statistics, sys
from rag2.calibrate import abstains, enrich
def load(V, S):
    g = json.load(open(f"data/validation_{S}.json")); g = g["questions"] if isinstance(g, dict) else g
    P = {p["qid"]: p for p in json.load(open(f"runs/raw_{V}_{S}.json"))["predictions"]}; enrich(P)
    r = {d["qid"]: d["points"] for d in json.load(open(f"runs/rep_{V}_{S}.json"))["details"]}
    return {q["qid"]: q["answerable"] for q in g}, P, r
def score(d, rule):
    ans, P, r = d
    return sum(((0 if ans[q] else 1) if abstains(P[q]["_meta"], rule) else r[q]) for q in r) / len(r)
best = None
for V in sys.argv[1:]:
    d = {S: load(V, S) for S in ("v1", "v2")}
    secs = [p["_meta"]["seconds"] for S in ("v1", "v2") for p in d[S][1].values()]
    print(f"{V} : temps moyen {statistics.mean(secs):.1f} s/question (max {max(secs):.0f} s), brut v1 "
          f"{sum(d['v1'][2].values()) / 100:+.3f}, brut v2 {sum(d['v2'][2].values()) / 40:+.3f}")
    for ag in ("souple_inclus", "strict_inclus", "souple", "off"):
        for u in (None, 0, 1):
            for th in (None, 0.5, 0.593, 0.7):
                for ov in (False, True):
                    rule = {"agreement": ag, "max_unsupported": u, "threshold": th, "rows_override_unsupported": ov}
                    a, b = score(d["v1"], rule), score(d["v2"], rule)
                    obj = (a + b) / 2
                    if best is None or obj > best[0] + 1e-9:
                        best = (obj, V, rule, a, b)
    ref = json.load(open("rag2/abstention_final.json"))
    print(f"   règle actuelle : v1 {score(d['v1'], ref):+.3f}, v2 {score(d['v2'], ref):+.3f}")
obj, V, rule, a, b = best
print(f"RETENU : {V}, règle {rule} -> v1 {a:+.3f}, v2 {b:+.3f}")
rule.update({"validation_score_calibre": round(obj, 4), "v1": a, "v2": b, "variant": V})
json.dump(rule, open("runs/abstention_v2.json", "w"), indent=1)
PY
V=$(python -c "import json; print(json.load(open('runs/abstention_v2.json'))['variant'])")

step "Leaderboard v2 ($V)"
python -m rag2.pipeline retrieve --index-dir $I --questions data/classement_v2_questions.json --out runs/ret_cls_v2.json --split
python -m rag2.pipeline generate --index-dir $I --retrieval runs/ret_cls_v2.json --questions data/classement_v2_questions.json \
  --out runs/raw_cls_v2.json --variant $V
python -m rag2.calibrate apply --raw runs/raw_cls_v2.json --rule runs/abstention_v2.json --out runs/classement_v2.json
python -m groundtruth.submit --pred runs/classement_v2.json --phase v2 || echo "Soumission v2 non créée (quota ?)"
python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
files = glob.glob("runs/raw_*") + glob.glob("runs/rep_*") + glob.glob("runs/ret_v*") + ["runs/abstention_v2.json",
         "runs/classement_v2.json", "/tmp/index/docs.json"] + sorted(glob.glob("submissions/v2/*"), key=os.path.getmtime)[-1:]
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        if os.path.exists(f):
            z.write(f, os.path.basename(f) if f.startswith("/tmp") else f)
print("->", sys.argv[1])
PY
