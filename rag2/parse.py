"""Lecture des PDF, avec OCR là où il n'y a pas de couche texte.

Approche retenue (« texte + OCR ») :
  - page avec couche texte : texte PyMuPDF, découpé en blocs (≈ paragraphes) ;
  - page sans texte (les 10 documents numérisés) : OCR Tesseract (français) de la page rendue à 300 dpi ;
  - image de taille significative dans une page texte (graphiques bitmap) : OCR de l'image, ajouté à la page
    comme bloc « [Texte du graphique] ... » (récupère les étiquettes et valeurs écrites dans le graphique).

Approches comparées (voir rag2/ablation_parsing.py) : texte brut seul, texte + tableaux (find_tables),
pymupdf4llm (Markdown), texte + OCR.

L'OCR coûte ~1 à 10 s par page/image : il est fait une fois à l'indexation, en parallèle, quand le
générateur n'est pas chargé.
"""
import io
import re
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

MIN_TEXT_CHARS = 50          # en dessous : page considérée comme numérisée
IMG_MIN_FRAC = 0.05          # image >= 5 % de la page : on l'OCRise
IMG_MAX_FRAC = 0.90          # au-delà : fond de page, pas un graphique
OCR_DPI = 300


def _ocr_image(png_bytes: bytes) -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(png_bytes)).convert("L")       # niveaux de gris
    return pytesseract.image_to_string(img, lang="fra", config="--psm 3")


def _ocr_job(job):
    """job = (index de la page, "page" | "img", chemin du pdf, numéro de page 0-based, bbox ou None)."""
    idx, kind, pdf_path, pno, bbox = job
    with pymupdf.open(pdf_path) as doc:
        page = doc[pno]
        clip = pymupdf.Rect(bbox) if bbox else None
        png = page.get_pixmap(dpi=OCR_DPI, clip=clip).tobytes("png")
    return idx, kind, _ocr_image(png)


def _clean(text: str) -> str:
    return " ".join(text.split())


_NUMERIC_LINE = re.compile(r"^[-−–+]?\s*[\d\s.,%€$]+\s*(%|€|M€|Md€|pts?|points?)?$")


def block_text(raw: str) -> str:
    """Texte d'un bloc PyMuPDF. Si le bloc ressemble à une rangée de tableau (une cellule par ligne,
    surtout des nombres), on sépare les cellules par « | » au lieu d'un espace :
    'Zone urbaine | 91 567 | 44 243 740 | 21 | 19 | 4,7 | 72' plutôt que 'Zone urbaine 91 567 44 243 740 21 19 4,7 72',
    où le modèle ne peut plus savoir où finit un nombre. Les paragraphes de texte restent joints par des espaces.
    """
    lines = [_clean(l) for l in raw.split("\n") if _clean(l)]
    if len(lines) >= 3:
        numeric = sum(bool(_NUMERIC_LINE.match(l)) for l in lines)
        if numeric / len(lines) >= 0.5:
            return " | ".join(lines)
    return " ".join(lines)


def read_pdf_dir(pdf_dir: str, ocr: bool = True, ocr_images: bool = True, workers: int | None = None) -> list[dict]:
    """Retourne [{doc_id, page, text, blocks, source}] pour tous les PDF du dossier.

    blocks : liste de paragraphes (chaînes) ; text : blocs joints par des sauts de ligne.
    source : "text" (couche texte), "ocr" (page numérisée).
    """
    pages, jobs = [], []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        with pymupdf.open(pdf_path) as doc:
            for pno, page in enumerate(doc):
                raw = page.get_text()
                rec = {"doc_id": pdf_path.stem, "page": pno + 1, "blocks": [], "source": "text"}
                if len(raw.strip()) < MIN_TEXT_CHARS:
                    rec["source"] = "ocr"
                    if ocr:
                        jobs.append((len(pages), "page", str(pdf_path), pno, None))
                else:
                    # blocs de texte dans l'ordre natif du PDF (type 0 = texte). Pas de sort=True : sur les pages
                    # en colonnes, le tri par position verticale mélange les paragraphes des colonnes.
                    for b in page.get_text("blocks"):
                        if b[6] == 0 and _clean(b[4]):
                            rec["blocks"].append(block_text(b[4]))
                    if ocr and ocr_images:
                        area = page.rect.width * page.rect.height
                        for info in page.get_image_info():
                            r = pymupdf.Rect(info["bbox"]) & page.rect
                            frac = r.width * r.height / area if area else 0
                            if IMG_MIN_FRAC <= frac <= IMG_MAX_FRAC:
                                jobs.append((len(pages), "img", str(pdf_path), pno, tuple(r)))
                pages.append(rec)

    if jobs:
        workers = workers or os.cpu_count() or 2
        print(f"OCR : {sum(j[1] == 'page' for j in jobs)} pages numérisées, "
              f"{sum(j[1] == 'img' for j in jobs)} images, {workers} processus", flush=True)
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, kind, text in ex.map(_ocr_job, jobs, chunksize=4):
                rec = pages[idx]
                if kind == "page":
                    paras = [_clean(p) for p in text.split("\n\n") if _clean(p)]
                    rec["blocks"].extend(paras)
                else:
                    t = _clean(text)
                    if len(t) >= 20:
                        rec["blocks"].append("[Texte du graphique] " + t)
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"\r  {done}/{len(jobs)}", end="", flush=True)
        print()

    for rec in pages:
        rec["text"] = "\n".join(rec["blocks"])
    return pages


def doc_titles(pages: list[dict], n_words: int = 25) -> dict[str, str]:
    """Titre approximatif de chaque document : les premiers mots de sa première page non vide."""
    titles = {}
    for p in pages:
        if p["doc_id"] not in titles and p["text"].strip():
            titles[p["doc_id"]] = " ".join(p["text"].split()[:n_words])
    return titles
