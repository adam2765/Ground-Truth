"""Sonde : un modèle de vision plus fort lit-il correctement les graphiques ? (avant de l'intégrer)

Le run 4 (Qwen2.5-VL-3B) a fait baisser le score : lectures de valeurs trop souvent fausses. On mesure ici,
sur les pages de preuve des 12 questions « graphique » de la validation, ce que lisent des modèles plus gros
chargés en 4 bits (bitsandbytes) pour tenir sur un T4 : la valeur attendue (key_values) apparaît-elle dans la
description, et la réponse attendue (catégorie la plus haute / la plus basse) est-elle lisible ?

    CUDA_VISIBLE_DEVICES=0 python -m rag2.vision_probe --model Qwen/Qwen2.5-VL-7B-Instruct --out runs/vision_7b.jsonl
    python -m rag2.vision_probe --report runs/vision_*.jsonl
"""
import argparse
import glob
import io
import json
import time
from pathlib import Path

import pymupdf

from rag2.describe_charts import PROMPT

DPI = 110


def graph_pages(gold="data/validation_v1.json") -> list[dict]:
    g = json.load(open(gold))
    g = g["questions"] if isinstance(g, dict) else g
    out = []
    for q in g:
        if q["type"] == "graphique":
            for grp in q["evidence"]:
                for e in grp:
                    out.append({"qid": q["qid"], "doc_id": e["doc_id"], "page": e["page"]})
    return out


def run(model_id: str, out_path: str, pdf_dir: str = "corpus/pdf") -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    t0 = time.time()
    granite = "granite" in model_id.lower()
    if granite:          # IBM Granite 4.0 3B Vision (entraîné sur ChartNet) : 4 Md, tient en float16 sur T4
        model = AutoModelForImageTextToText.from_pretrained(model_id, trust_remote_code=True, dtype=torch.float16,
                                                            device_map={"": 0}).eval()
        if hasattr(model, "merge_lora_adapters"):
            model.merge_lora_adapters()
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    else:
        q4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForImageTextToText.from_pretrained(model_id, quantization_config=q4, device_map={"": 0},
                                                            dtype=torch.float16).eval()
        proc = AutoProcessor.from_pretrained(model_id)
    print(f"{model_id} chargé en {time.time() - t0:.0f} s, mémoire GPU {torch.cuda.memory_allocated() / 1e9:.1f} Go",
          flush=True)
    seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for p in graph_pages():
            key = (p["doc_id"], p["page"])
            if key in seen:
                continue
            seen.add(key)
            if not (Path(pdf_dir) / f"{p['doc_id']}.pdf").exists():
                continue                               # (document hors corpus v1)
            with pymupdf.open(Path(pdf_dir) / f"{p['doc_id']}.pdf") as doc:
                img = Image.open(io.BytesIO(doc[p["page"] - 1].get_pixmap(dpi=DPI).tobytes("png"))).convert("RGB")
            if granite:      # balise de tâche : graphique -> CSV
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<chart2csv>"}]}]
                text = proc.apply_chat_template(msgs, add_generation_prompt=True)
                inputs = proc(text=text, images=img, return_tensors="pt", padding=True).to(model.device)
            else:
                msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
                inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                  return_tensors="pt").to(model.device)
            t = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=600, do_sample=False)
            text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            dt = time.time() - t
            f.write(json.dumps({**p, "model": model_id, "text": text, "seconds": round(dt, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\n===== {p['qid']} {p['doc_id']} p.{p['page']} ({dt:.0f} s) =====\n{text}", flush=True)


def report(files: list[str], gold="data/validation_v1.json") -> None:
    from groundtruth.normalize import _close, extract_numbers
    g = json.load(open(gold))
    g = {q["qid"]: q for q in (g["questions"] if isinstance(g, dict) else g)}
    for path in files:
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        if not rows:
            continue
        by_page = {(r["doc_id"], r["page"]): r["text"] for r in rows}
        print(f"\n### {rows[0]['model']} : {sum(r['seconds'] for r in rows) / len(rows):.0f} s/page")
        for qid in dict.fromkeys(r["qid"] for r in rows):
            q = g[qid]
            txt = " ".join(by_page.get((e["doc_id"], e["page"]), "") for grp in q["evidence"] for e in grp)
            nums = extract_numbers(txt)
            hits = [any(_close(a, b) or _close(sa, sb) for a, sa in extract_numbers(v) for b, sb in nums)
                    for v in q.get("key_values", [])]
            print(f"  {qid} valeurs attendues {q.get('key_values')} -> {hits} | attendu : {q['answer'][:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--report", nargs="*")
    a = ap.parse_args()
    if a.report is not None:
        report([f for pat in a.report for f in glob.glob(pat)])
    else:
        run(a.model, a.out)


if __name__ == "__main__":
    main()
