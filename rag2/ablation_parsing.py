"""Comparaison des approches de parsing, mesurée automatiquement.

Pour chaque question de validation qui a une réponse, on lit les pages de preuve avec chaque approche et
on regarde quelle part des valeurs clés (key_values) se retrouve dans le texte extrait. Une valeur
absente du texte ne pourra jamais être retrouvée ni citée par le LLM.

    python -m rag2.ablation_parsing            (CPU, ~5-10 min à cause de l'OCR et de pymupdf4llm)

Approches :
  raw     : texte PyMuPDF seul (palier 1)
  tables  : raw + tableaux détectés par find_tables(), ajoutés en Markdown
  md      : pymupdf4llm (Markdown, mise en page, OCR intégré)
  ocr     : raw + OCR Tesseract des pages sans texte et des images de graphiques (approche retenue)

Limite : certaines valeurs clés sont des calculs (un écart entre deux chiffres, « 32 milliers » écrit
32 000) et n'apparaissent dans aucun texte : la couverture maximale n'est donc pas 100 %.
"""
import io
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pymupdf

from groundtruth.io import load_questions
from groundtruth.normalize import value_present
from rag2.parse import IMG_MAX_FRAC, IMG_MIN_FRAC, MIN_TEXT_CHARS, OCR_DPI, _ocr_image

APPROACHES = ["raw", "tables", "md", "ocr"]


def page_text(doc, pno: int, approach: str) -> str:
    page = doc[pno]
    raw = page.get_text()
    if approach == "raw":
        return raw
    if approach == "tables":
        out = raw
        try:
            for tb in page.find_tables().tables:
                out += "\n" + tb.to_markdown()
        except Exception:
            pass
        return out
    if approach == "md":
        import pymupdf4llm
        return pymupdf4llm.to_markdown(doc, pages=[pno], page_chunks=True, show_progress=False)[0]["text"]
    if approach == "ocr":
        if len(raw.strip()) < MIN_TEXT_CHARS:
            return _ocr_image(page.get_pixmap(dpi=OCR_DPI).tobytes("png"))
        out = raw
        area = page.rect.width * page.rect.height
        for info in page.get_image_info():
            r = pymupdf.Rect(info["bbox"]) & page.rect
            if IMG_MIN_FRAC <= r.width * r.height / area <= IMG_MAX_FRAC:
                out += "\n" + _ocr_image(page.get_pixmap(dpi=OCR_DPI, clip=r).tobytes("png"))
        return out
    raise ValueError(approach)


def main():
    qs = [q for q in load_questions("data/validation_v1.json") if q["answerable"]]
    docs = {}
    results = {}
    for approach in APPROACHES:
        t0 = time.time()
        by_type = defaultdict(list)
        for q in qs:
            texts = []
            for g in q["evidence"]:
                for e in g:
                    path = f"corpus/pdf/{e['doc_id']}.pdf"
                    if not os.path.exists(path):        # page de preuve alternative dans le corpus v2
                        continue
                    docs.setdefault(path, pymupdf.open(path))
                    texts.append(page_text(docs[path], e["page"] - 1, approach))
            txt = "\n".join(texts)
            kv = q.get("key_values") or []
            frac = sum(value_present(v, txt) for v in kv) / len(kv) if kv else 1.0
            by_type[q["type"]].append(frac)
            by_type["TOUT"].append(frac)
        results[approach] = {t: round(sum(v) / len(v), 3) for t, v in sorted(by_type.items())}
        results[approach]["secondes"] = round(time.time() - t0)
        print(approach, results[approach], flush=True)

    types = sorted(k for k in results["raw"] if k != "secondes")
    print("\n| Approche | " + " | ".join(types) + " | temps |")
    print("|---" * (len(types) + 2) + "|")
    for a in APPROACHES:
        print(f"| {a} | " + " | ".join(f"{results[a][t]:.2f}" for t in types) + f" | {results[a]['secondes']} s |")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/ablation_parsing.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
