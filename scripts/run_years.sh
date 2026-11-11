#!/usr/bin/env bash
# Test du correctif « questions datées » (11/11), ~1 h 15 sur Kaggle T4.
# Depuis le palier 3, les extraits d'une édition remplacée sont retirés, même quand la question porte sur l'année
# que seule cette édition couvre (« en 2023, combien gagne un enseignant ? » : l'édition 2026 donne 2024).
# Correctif (--year-keep) : on garde l'ancienne édition si la question est datée sans « dernier / plus récent ».
#
# Méthode : référence = répétition générale du 9/11 (results/final2, même code, juge officiel). Ici, on refait la
# recherche avec et sans correctif ; seules les questions dont la recherche change sont régénérées, les autres
# reprennent la réponse de la référence. Puis juge officiel et même règle d'abstention.
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_years.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
exec > >(tee "$OUT/log_years.txt") 2>&1
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
step "setup"
timeout --foreground 2400 bash setup.sh < /dev/null 2>&1 | grep -v -E "Downloading|Reconstructing|Fetching|it/s" | tail -n 3
python -c "import pymupdf, Stemmer, sentence_transformers" || { echo "setup.sh a échoué"; exit 1; }
python corpus/download_corpus.py | tail -n 1
set -e
I=/tmp/index; mkdir -p $I runs
step "run.sh index + update (comme la référence : tout recalculé, ~25 min)"
bash run.sh index corpus/pdf $I 2>&1 | grep -v -E "^  [0-9]*/|Batches|Loading weights|it/s" | tail -n 3
bash run.sh update $I corpus/pdf_v2 2>&1 | grep -v -E "Batches|Loading weights|it/s" | tail -n 3

for S in v1 v2; do
  step "recherche $S : actuelle (A) et avec correctif (B)"
  python -m rag2.pipeline retrieve --index-dir $I --questions data/validation_$S.json --out runs/retA_$S.json --split
  python -m rag2.pipeline retrieve --index-dir $I --questions data/validation_$S.json --out runs/retB_$S.json --split --year-keep
  python - $S <<'PY'
import json, sys
S = sys.argv[1]
a = {it["qid"]: [h["idx"] for h in it["hits"]] for it in json.load(open(f"runs/retA_{S}.json"))["items"]}
b = {it["qid"]: [h["idx"] for h in it["hits"]] for it in json.load(open(f"runs/retB_{S}.json"))["items"]}
ref = {p["qid"]: p for p in json.load(open(f"results/final2/val_{S}_raw.json"))["predictions"]}
changed = [q for q in b if a[q] != b[q]]
print(f"{S} : recherche modifiée pour {len(changed)} questions : {changed}")
with open(f"runs/rawB_{S}.json.partial.jsonl", "w", encoding="utf-8") as f:
    for q in b:
        if q not in changed:
            f.write(json.dumps(ref[q], ensure_ascii=False) + "\n")
json.dump(changed, open(f"runs/changed_{S}.json", "w"))
PY
  step "génération $S (questions modifiées seulement)"
  python -m rag2.pipeline generate --index-dir $I --retrieval runs/retB_$S.json --questions data/validation_$S.json \
    --out runs/rawB_$S.json --variant duoXRS
  python -m rag2.calibrate apply --raw runs/rawB_$S.json --rule rag2/abstention_final.json --out runs/valB_$S.json
  python -m groundtruth.evaluate --pred runs/valB_$S.json --gold data/validation_$S.json --judge hf --out runs/repB_$S.json > /dev/null
done

step "Comparaison (référence : répétition générale du 9/11)"
python - <<'PY'
import json
tot = {}
for S, n in (("v1", 100), ("v2", 40)):
    A = {d["qid"]: d["points"] for d in json.load(open(f"results/final2/val_{S}_report.json"))["details"]}
    B = {d["qid"]: d["points"] for d in json.load(open(f"runs/repB_{S}.json"))["details"]}
    ch = json.load(open(f"runs/changed_{S}.json"))
    print(f"{S} : actuel {sum(A.values()) / n:+.4f} | correctif {sum(B.values()) / n:+.4f}")
    for q in ch:
        print(f"   {q} : {A[q]:+} -> {B[q]:+}")
    tot[S] = (sum(B.values()) - sum(A.values())) / n
print(f"ÉCART : v1 {tot['v1']:+.4f}, v2 {tot['v2']:+.4f} ->", "GARDER le correctif" if tot["v1"] + tot["v2"] > 0 and min(tot.values()) >= -0.0251 else "NE PAS garder")
PY
OUT="$OUT" python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/*B_*") + glob.glob("runs/changed_*") + glob.glob("runs/retA_*") + [os.environ.get("OUT", "") + "/log_years.txt"]:
        if os.path.exists(f):
            z.write(f, os.path.basename(f) if f.startswith("/") else f)
print("->", sys.argv[1])
PY
