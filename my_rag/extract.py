"""Étape 1 : lire les PDF page par page."""
from pathlib import Path

import pymupdf


def extract_pages(pdf_dir: str) -> list[dict]:
    """Retourne une liste de {"doc_id": ..., "page": ..., "text": ...}.

    Les pages sont numérotées à partir de 1 (index physique dans le PDF), comme le kit l'exige.
    Les PDF numérisés n'ont pas de couche texte : leur "text" est vide (il faudra de l'OCR).
    """
    pages = []
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        doc_id = pdf_path.stem
        doc = pymupdf.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append({"doc_id": doc_id, "page": i + 1, "text": text})
        doc.close()
    return pages


if __name__ == "__main__":
    pages = extract_pages("corpus/pdf")
    print(len(pages), "pages")
    print(pages[0])
    vides = [p for p in pages if not p["text"].strip()]
    print(len(vides), "pages sans texte, ex.", sorted({p["doc_id"] for p in vides}))
