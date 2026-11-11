#!/usr/bin/env bash
# Test de la variante duoXRSt (12/11), ~1 h 30 sur Kaggle T4.
# Le palier 3 a ajouté à tous les extraits leur date de publication et la consigne « la publication la plus récente
# fait foi ». Sur la v1, ça coûte ~0,05 (+0,525 -> +0,475) : le modèle hésite davantage. duoXRSt ne les met que si
# deux éditions d'un même document sont dans le contexte, ou si la question porte sur une évolution / la donnée la
# plus récente (rag2/answer.py, temporal_needed). Sinon le prompt redevient celui du palier 2.
#
# Méthode : quand les dates restent actives, duoXRSt = duoXRS (même prompt) : on reprend la réponse de référence
# (répétition générale du 9/11 + correctif des questions datées, results/final2 et results/years). Seules les autres
# questions sont régénérées, puis juge officiel et même règle d'abstention.
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_temporal.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
exec > >(tee "$OUT/log_temporal.txt") 2>&1
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
step "setup"
timeout --foreground 2400 bash setup.sh < /dev/null 2>&1 | grep -v -E "Downloading|Reconstructing|Fetching|it/s" | tail -n 3
python -c "import pymupdf, Stemmer, sentence_transformers" || { echo "setup.sh a échoué"; exit 1; }
python corpus/download_corpus.py | tail -n 1
set -e
I=/tmp/index; mkdir -p $I runs
step "run.sh index + update (comme la référence, ~25 min)"
bash run.sh index corpus/pdf $I 2>&1 | grep -v -E "^  [0-9]*/|Batches|Loading weights|it/s" | tail -n 3
bash run.sh update $I corpus/pdf_v2 2>&1 | grep -v -E "Batches|Loading weights|it/s" | tail -n 3

for S in v1 v2; do
  step "recherche $S (comme run.sh : --split --year-keep) et questions à régénérer"
  python -m rag2.pipeline retrieve --index-dir $I --questions data/validation_$S.json --out runs/ret_$S.json --split --year-keep
  python - $S <<'PY'
import json, sys
import rag2.answer as a
from rag2 import docmeta
S = sys.argv[1]
meta = docmeta.load("/tmp/index")
a.DOC_EDITIONS.update({d: set(v.get("newer", [])) | set(v.get("older", [])) for d, v in meta.items()})
chunks = json.load(open("/tmp/index/naive_title/chunks.json"))
items = json.load(open(f"runs/ret_{S}.json"))["items"]
g = json.load(open(f"data/validation_{S}.json")); g = g["questions"] if isinstance(g, dict) else g
question = {x["qid"]: x["question"] for x in g}
ref = {p["qid"]: p for p in json.load(open(f"results/final2/val_{S}_raw.json"))["predictions"]}
kept = ["val1-011", "val1-039", "val1-051", "val1-055", "val1-063", "val1-072", "val1-083"]   # correctif du 11/11
for p in json.load(open(f"results/years/rawB_{S}.json"))["predictions"]:
    if p["qid"] in kept:
        ref[p["qid"]] = p
todo = []
with open(f"runs/raw_{S}.json.partial.jsonl", "w", encoding="utf-8") as f:
    for it in items:
        q = it["qid"]
        if a.temporal_needed(question[q], [chunks[h["idx"]]["doc_id"] for h in it["hits"]]):
            f.write(json.dumps(ref[q], ensure_ascii=False) + "\n")
        else:
            todo.append(q)
print(f"{S} : dates retirées pour {len(todo)} questions sur {len(items)} (régénérées)")
json.dump(todo, open(f"runs/todo_{S}.json", "w"))
json.dump({"predictions": list(ref.values())}, open(f"runs/ref_{S}_raw.json", "w"), ensure_ascii=False)
PY
  step "génération $S (duoXRSt, questions sans dates seulement)"
  python -m rag2.pipeline generate --index-dir $I --retrieval runs/ret_$S.json --questions data/validation_$S.json \
    --out runs/raw_$S.json --variant duoXRSt
  python -m rag2.calibrate apply --raw runs/raw_$S.json --rule rag2/abstention_final.json --out runs/val_$S.json
  python -m groundtruth.evaluate --pred runs/val_$S.json --gold data/validation_$S.json --judge hf --out runs/rep_$S.json > /dev/null
done

step "Comparaison (référence : 9/11 + correctif des questions datées = v1 +0,455, v2 +0,2375)"
python - <<'PY'
import json
gain = {}
for S, n in (("v1", 100), ("v2", 40)):
    A = {d["qid"]: d["points"] for d in json.load(open(f"results/final2/val_{S}_report.json"))["details"]}
    for d in (json.load(open(f"results/years/repB_{S}.json"))["details"]):
        if d["qid"] in ["val1-011", "val1-039", "val1-051", "val1-055", "val1-063", "val1-072", "val1-083"]:
            A[d["qid"]] = d["points"]
    B = {d["qid"]: d["points"] for d in json.load(open(f"runs/rep_{S}.json"))["details"]}
    todo = json.load(open(f"runs/todo_{S}.json"))
    print(f"{S} : référence {sum(A.values()) / n:+.4f} | duoXRSt {sum(B.values()) / n:+.4f}")
    for q in todo:
        if A[q] != B[q]:
            print(f"   {q} : {A[q]:+} -> {B[q]:+}")
    gain[S] = (sum(B.values()) - sum(A.values())) / n
    raw = json.load(open(f"runs/raw_{S}.json"))["predictions"]
    print(f"   temps moyen de génération (questions régénérées) : "
          f"{sum(p['_meta']['seconds'] for p in raw if p['qid'] in todo) / max(1, len(todo)):.1f} s")
print(f"ÉCART : v1 {gain['v1']:+.4f}, v2 {gain['v2']:+.4f} ->",
      "GARDER duoXRSt" if gain["v1"] + gain["v2"] > 0 and min(gain.values()) >= -0.0251 else "garder duoXRS")
PY
OUT="$OUT" python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/raw_v*") + glob.glob("runs/rep_v*") + glob.glob("runs/val_v*") + glob.glob("runs/todo_*") \
            + glob.glob("runs/ret_v*") + [os.environ.get("OUT", "") + "/log_temporal.txt"]:
        if os.path.exists(f):
            z.write(f, os.path.basename(f) if f.startswith("/") else f)
print("->", sys.argv[1])
PY
