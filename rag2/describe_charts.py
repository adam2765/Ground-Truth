"""Description des graphiques par un modèle de vision open weights, à l'indexation (autorisé par le README).

Problème (annotation des 30 pages, runs 1 à 3) : sur les graphiques, les valeurs sont dessinées (barres,
courbes, camemberts) et absentes du texte ; le générateur, qui ne voit que du texte, devine.
Solution : Qwen2.5-VL-3B-Instruct regarde chaque page qui contient un graphique et écrit en texte le titre
et les valeurs lisibles. Ce texte est ajouté à la page (« [Description des graphiques] ... ») puis indexé.

Calcul unique, mis en cache (cache/charts_v1.json.gz), comme l'OCR :
    python -m rag2.describe_charts select --index-dir index2 --out runs/chart_pages.json
    CUDA_VISIBLE_DEVICES=0 python -m rag2.describe_charts run --pages runs/chart_pages.json --shard 0/2 --out runs/charts_0.jsonl
    CUDA_VISIBLE_DEVICES=1 python -m rag2.describe_charts run --pages runs/chart_pages.json --shard 1/2 --out runs/charts_1.jsonl
    python -m rag2.describe_charts merge --inputs runs/charts_*.jsonl --out cache/charts_v1.json.gz
"""
import argparse
import glob
import gzip
import io
import json
import re
import time
from pathlib import Path

import pymupdf

VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
DPI = 110                      # page A4 ~ 900 x 1290 px
MAX_NEW = 320

PROMPT = (
    "Cette image est une page d'un rapport public français (Insee ou Cour des comptes). "
    "Pour chaque graphique, carte ou tableau présenté sous forme d'image de la page, écris :\n"
    "- son titre (ou son numéro, par exemple « Figure 2 » ou « Graphique n° 7 ») ;\n"
    "- toutes les valeurs que l'on peut lire, une par ligne, sous la forme « catégorie ou année : valeur unité ».\n"
    "Si une valeur n'est pas écrite sur le graphique, donne une estimation lue sur l'axe, précédée de « environ ». "
    "Indique aussi la catégorie la plus élevée et la plus faible. N'invente aucune catégorie. "
    "Ignore le texte courant de la page. Écris en français. "
    "S'il n'y a aucun graphique ni tableau en image, réponds seulement : AUCUN"
)

CAPTION = re.compile(r"(Graphique\s*n\s*°|Figure\s*\d|figure\s*\d|►\s*\d+\.|Carte\s*n\s*°|Schéma\s*n\s*°)", re.I)


def select_pages(index_dir: str, pdf_dir: str = "corpus/pdf") -> list[list]:
    """Pages à décrire : légende de graphique, beaucoup de tracés vectoriels, ou image de graphique."""
    pages = json.loads((Path(index_dir) / "pages.json").read_text(encoding="utf-8"))
    sel, docs = [], {}
    for p in pages:
        if p["source"] == "ocr":
            if CAPTION.search(p["text"]):
                sel.append([p["doc_id"], p["page"]])
            continue
        doc = docs.setdefault(p["doc_id"], pymupdf.open(Path(pdf_dir) / f"{p['doc_id']}.pdf"))
        page = doc[p["page"] - 1]
        big_img = any(b.startswith("[Texte du graphique]") for b in p["blocks"])
        if CAPTION.search(p["text"]) or len(page.get_drawings()) >= 30 or big_img:
            sel.append([p["doc_id"], p["page"]])
    return sel


def _looks_broken(text: str) -> bool:
    t = text.strip()
    return not t or (len(set(t)) <= 3 and len(t) > 10)


def run(pages: list[list], out_path: str, pdf_dir: str = "corpus/pdf") -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    done = set()
    if Path(out_path).exists():                      # reprise
        for line in Path(out_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["doc_id"], r["page"]))
    todo = [p for p in pages if tuple(p) not in done]
    print(f"{len(done)} déjà faites, {len(todo)} à faire", flush=True)
    if not todo:
        return

    dtype = torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM, dtype=dtype, device_map={"": 0}).eval()
    processor = AutoProcessor.from_pretrained(VLM, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    t0, docs, broken = time.time(), {}, 0
    with open(out_path, "a", encoding="utf-8") as f:
        for n, (doc_id, page_no) in enumerate(todo, 1):
            doc = docs.setdefault(doc_id, pymupdf.open(Path(pdf_dir) / f"{doc_id}.pdf"))
            png = doc[page_no - 1].get_pixmap(dpi=DPI).tobytes("png")
            img = Image.open(io.BytesIO(png)).convert("RGB")
            inputs = processor(text=[prompt_text], images=[img], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
            text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            ok = not _looks_broken(text)
            broken = broken + (not ok) if n <= 5 else broken
            if n == 5 and broken == 5:
                raise SystemExit("Les 5 premières descriptions sont illisibles (float16 ?) : arrêt pour ne pas "
                                 "gaspiller le GPU. Envoie le log à Claude.")
            f.write(json.dumps({"doc_id": doc_id, "page": page_no, "text": text, "ok": ok}, ensure_ascii=False) + "\n")
            f.flush()
            if n <= 3 or n % 25 == 0:
                print(f"  {n}/{len(todo)} ({(time.time() - t0) / n:.1f} s/page) {doc_id} p.{page_no}: "
                      f"{' '.join(text.split())[:160]}", flush=True)


def merge(inputs: list[str], out_path: str) -> None:
    rows = {}
    for path in inputs:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["doc_id"], r["page"])] = r
    kept = [r for r in rows.values() if r["ok"] and not re.match(r"^\W*AUCUN\W*$", r["text"], re.I)]
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
    print(f"{len(rows)} pages décrites, {len(kept)} avec un graphique -> {out_path}")


def add_descriptions(pages: list[dict], cache_path: str = "cache/charts_v1.json.gz") -> int:
    """Ajoute les descriptions en cache aux pages (idempotent). Retourne le nombre de pages enrichies."""
    if not Path(cache_path).exists():
        return 0
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        desc = {(r["doc_id"], r["page"]): r["text"] for r in json.load(f)}
    n = 0
    for p in pages:
        t = desc.get((p["doc_id"], p["page"]))
        if t and not any(b.startswith("[Description des graphiques]") for b in p["blocks"]):
            p["blocks"].append("[Description des graphiques]\n" + t)
            p["text"] = "\n".join(p["blocks"])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--index-dir", default="index2")
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pages", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "select":
        sel = select_pages(args.index_dir)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(sel))
        print(f"{len(sel)} pages à décrire -> {args.out}")
    elif args.cmd == "run":
        i, n = map(int, args.shard.split("/"))
        pages = json.loads(Path(args.pages).read_text())[i::n]
        if args.limit:
            pages = pages[: args.limit]
        run(pages, args.out)
    else:
        files = [f for pat in args.inputs for f in glob.glob(pat)]
        merge(files, args.out)


if __name__ == "__main__":
    main()
