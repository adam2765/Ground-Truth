"""Description des graphiques du corpus par un petit modèle de vision, à l'indexation (autorisé par le README).

Sondes du 31/10 (results/vision/) : sur les pages entières, les modèles de vision lisent mal ; sur les graphiques
DÉCOUPÉS (rag2/chart_crops.py), Qwen3-VL-2B et -4B retrouvent 5 à 6 des 6 valeurs chiffrées attendues. Il reste à
tenir dans le budget d'indexation (2 h sur un T4) : ~874 graphiques, décrits par lots (batch) avec des réponses
courtes.

    CUDA_VISIBLE_DEVICES=0 python -m rag2.charts_vlm --model Qwen/Qwen3-VL-2B-Instruct --pdf-dir corpus/pdf --out runs/charts_2b.jsonl
    python -m rag2.charts_vlm --merge runs/charts_2b.jsonl --out cache/charts_2b.json.gz
"""
import argparse
import gzip
import io
import json
import time
from pathlib import Path

import pymupdf

PROMPT = (
    "This image is a chart from a French public report. Write its title, then list every data value as "
    "« catégorie (année) : valeur unité », one per line. If a value is not printed, estimate it from the axis and "
    "write « environ ». Finish with one line « Plus haut : ... ; plus bas : ... ». Answer in French, no other text."
)


def crops_of(pdf_dir: str, dpi: int = 130):
    """[(doc_id, page, n° du graphique, image PIL)] pour toutes les pages à couche texte."""
    from PIL import Image

    from rag2.chart_crops import chart_regions, crop_png
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for pno, page in enumerate(doc):
                if len(page.get_text().strip()) < 50:
                    continue
                for i, r in enumerate(chart_regions(page)):
                    yield pdf.stem, pno + 1, i, Image.open(io.BytesIO(crop_png(page, r, dpi))).convert("RGB")


def run(model_id: str, pdf_dir: str, out: str, batch: int, max_new: int, limit: int = 0) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    done = set()
    if Path(out).exists():
        done = {(r["doc_id"], r["page"], r["i"]) for r in map(json.loads, Path(out).read_text().splitlines()) if r}
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16, device_map={"": 0}).eval()
    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = "left"
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print(f"{model_id} chargé en {time.time() - t0:.0f} s", flush=True)
    t0, n, buf = time.time(), 0, []

    def flush(f):
        nonlocal n
        inputs = proc(text=[prompt] * len(buf), images=[b[3] for b in buf], padding=True,
                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        texts = proc.batch_decode(o[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for (d, p, i, _), t in zip(buf, texts):
            f.write(json.dumps({"doc_id": d, "page": p, "i": i, "text": t.strip()}, ensure_ascii=False) + "\n")
        f.flush()
        n += len(buf)
        buf.clear()
        if n % (batch * 10) == 0:
            print(f"  {n} graphiques, {(time.time() - t0) / n:.1f} s/graphique", flush=True)

    with open(out, "a", encoding="utf-8") as f:
        for item in crops_of(pdf_dir):
            if item[:3] in done:
                continue
            buf.append(item)
            if len(buf) == batch:
                flush(f)
            if limit and n >= limit:
                break
        if buf:
            flush(f)
    dt = time.time() - t0
    print(f"{n} graphiques décrits en {dt / 60:.1f} min ({dt / max(n, 1):.1f} s/graphique)", flush=True)


def merge(paths: list[str], out: str) -> None:
    """{"doc_id|page": "description du graphique 1\\n\\ndescription du graphique 2"}"""
    by_page = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                by_page.setdefault(f"{r['doc_id']}|{r['page']}", {})[r["i"]] = r["text"]
    res = {k: "\n\n".join(v[i] for i in sorted(v)) for k, v in by_page.items()}
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"{len(res)} pages -> {out}")


def load(path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--pdf-dir", default="corpus/pdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", nargs="*")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge, a.out)
    else:
        run(a.model, a.pdf_dir, a.out, a.batch, a.max_new, a.limit)


if __name__ == "__main__":
    main()
