#!/usr/bin/env bash
# Test de la transcription des pages numérisées par Qwen3-VL-4B (12/11), ~1 h 45 sur Kaggle T4.
# Tesseract abîme les tableaux scannés (les 9 questions « scan » de la v1 rapportent +0,11 en moyenne).
# rag2/scan_vlm.py fait transcrire les 104 pages numérisées (97 v1 + 7 v2) par un modèle de vision, tableaux en
# Markdown réécrits au format « rangée | en-tête : valeur ». Mesure aussi le temps (limite : index 2 h, update 20 min).
#
# Méthode : référence = répétition générale du 9/11 + correctif des questions datées (v1 +0,455, v2 +0,2375).
# Seules les questions dont la recherche touche un document numérisé (avant ou après) sont régénérées.
#
#   !git clone -q https://github.com/adam2765/Ground-Truth.git /tmp/gt; bash /tmp/gt/scripts/run_scans.sh
set -uo pipefail
cd "$(dirname "$0")/.."
git pull -q || true
OUT="${OUT:-$( [ -d /kaggle/working ] && echo /kaggle/working || pwd )}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
VLM="${VLM:-Qwen/Qwen3-VL-4B-Instruct}"
exec > >(tee "$OUT/log_scans.txt") 2>&1
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $* ==="; }
chrono() { local t=$(date +%s); "$@"; local rc=$?; echo "  -> $(( ($(date +%s) - t) / 60 )) min $(( ($(date +%s) - t) % 60 )) s"; return $rc; }
step "setup + $VLM"
timeout --foreground 2400 bash setup.sh < /dev/null 2>&1 | grep -v -E "Downloading|Reconstructing|Fetching|it/s" | tail -n 3
python -c "from huggingface_hub import snapshot_download as s; s('$VLM')" 2>&1 | tail -n 1
python -c "import pymupdf, Stemmer, sentence_transformers" || { echo "setup.sh a échoué"; exit 1; }
python corpus/download_corpus.py | tail -n 1
set -e
I=/tmp/index; mkdir -p $I runs
step "transcription des 97 pages numérisées v1 (OCR Tesseract en cache pour le reste)"
gunzip -c cache/pages_v1.json.gz > $I/pages.json
export RAG_SCAN_VLM=$VLM
chrono python -m rag2.scan_vlm --index-dir $I --pdf-dir corpus/pdf --log runs/scan_v1.jsonl
step "index + update v2 (avec transcription des scans v2)"
echo '{"strategy": "naive_title", "mode": "hybrid_rerank"}' > $I/best.json
python -m rag2.pipeline index --pdf-dir corpus/pdf --index-dir $I --strategies naive_title 2>&1 | grep -v -E "Batches|Loading weights|it/s" | tail -n 3
RAG_SCAN_LOG=runs/scan_v2.jsonl chrono python -m rag2.pipeline update --index-dir $I --pdf-dir corpus/pdf_v2 2>&1 | grep -v -E "Batches|Loading weights|it/s" | tail -n 8
unset RAG_SCAN_VLM

for S in v1 v2; do
  step "recherche $S (comme run.sh : --split --year-keep) et questions à régénérer"
  python -m rag2.pipeline retrieve --index-dir $I --questions data/validation_$S.json --out runs/ret_$S.json --split --year-keep
  python - $S <<'PY'
import json, sys
pages = json.load(open("/tmp/index/pages.json"))
scanned = {p["doc_id"] for p in pages if p.get("source") == "vlm"}   # pages réellement transcrites
S = sys.argv[1]
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
        new_docs = {chunks[h["idx"]]["doc_id"] for h in it["hits"]}
        old_docs = {r["doc_id"] for r in ref[q].get("retrieved", [])}
        if not ((new_docs | old_docs) & scanned):
            f.write(json.dumps(ref[q], ensure_ascii=False) + "\n")
        else:
            todo.append(q)
print(f"{S} : {len(todo)} questions sur {len(items)} touchent un document numérisé (régénérées) ; numérisés : {sorted(scanned)}")
json.dump(todo, open(f"runs/todo_{S}.json", "w"))
json.dump({"predictions": list(ref.values())}, open(f"runs/ref_{S}_raw.json", "w"), ensure_ascii=False)
PY
  step "génération $S (duoXRS, questions touchant un scan)"
  python -m rag2.pipeline generate --index-dir $I --retrieval runs/ret_$S.json --questions data/validation_$S.json \
    --out runs/raw_$S.json --variant duoXRS
  python -m rag2.calibrate apply --raw runs/raw_$S.json --rule rag2/abstention_final.json --out runs/val_$S.json
  python -m groundtruth.evaluate --pred runs/val_$S.json --gold data/validation_$S.json --judge hf --out runs/rep_$S.json > /dev/null
done

step "Comparaison (référence : 9/11 + correctif des questions datées = v1 +0,455, v2 +0,2375)"
python - <<'PY'
import json
for S in ('v1', 'v2'):
    rows = [json.loads(l) for l in open(f'runs/scan_{S}.jsonl')] if __import__('os').path.exists(f'runs/scan_{S}.jsonl') else []
    print(f'transcriptions {S} : {sum(r["ok"] for r in rows)} gardées sur {len(rows)}')
PY
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
    print(f"{S} : référence {sum(A.values()) / n:+.4f} | scans transcrits {sum(B.values()) / n:+.4f}")
    for q in todo:
        if A[q] != B[q]:
            print(f"   {q} : {A[q]:+} -> {B[q]:+}")
    gain[S] = (sum(B.values()) - sum(A.values())) / n
    raw = json.load(open(f"runs/raw_{S}.json"))["predictions"]
    print(f"   temps moyen de génération (questions régénérées) : "
          f"{sum(p['_meta']['seconds'] for p in raw if p['qid'] in todo) / max(1, len(todo)):.1f} s")
print(f"ÉCART : v1 {gain['v1']:+.4f}, v2 {gain['v2']:+.4f} ->",
      "GARDER la transcription" if gain["v1"] + gain["v2"] > 0 and min(gain.values()) >= -0.0251 else "NE PAS garder")
PY
OUT="$OUT" python - "$OUT/resultats.zip" <<'PY'
import glob, os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("runs/raw_v*") + glob.glob("runs/rep_v*") + glob.glob("runs/val_v*") + glob.glob("runs/todo_*") + glob.glob("runs/scan_*") \
            + glob.glob("runs/ret_v*") + [os.environ.get("OUT", "") + "/log_scans.txt"]:
        if os.path.exists(f):
            z.write(f, os.path.basename(f) if f.startswith("/") else f)
print("->", sys.argv[1])
PY
