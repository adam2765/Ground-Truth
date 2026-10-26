"""Stratégies de découpage. Un chunk ne mélange jamais deux pages (la citation reste exacte).

  naive        : fenêtres de 200 mots, recouvrement 50 (le chunker du palier 1)
  naive_title  : naive + en-tête « titre du document » devant chaque chunk
  page         : une page = un chunk, avec en-tête
  struct_title : regroupe les paragraphes (blocs PyMuPDF / OCR) jusqu'à ~180 mots, sans couper une
                 phrase au milieu quand c'est possible, avec en-tête

Pourquoi l'en-tête : un passage comme « ils sont passés de 308 000 à 276 600 » ne dit pas de quoi il
parle ; le titre du rapport (« ... politique agricole commune ... ») donne ce contexte à BM25, à
l'embedding et au reranker. Le champ "body" garde le texte sans en-tête (c'est lui qu'on montre au LLM).
"""
import re

from my_rag.chunk import chunk_pages as naive_chunks
from rag2.parse import doc_titles

STRATEGIES = ["naive", "naive_title", "page", "struct_title"]


def _header(title: str, page: int) -> str:
    return f"{title} (p.{page})\n"


def _split_long(text: str, max_words: int) -> list[str]:
    """Coupe un bloc trop long en morceaux de phrases d'au plus max_words mots."""
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    out, cur = [], []
    for s in sentences:
        words = s.split()
        if len(words) > max_words:                      # phrase énorme (tableau aplati) : coupe brute
            if cur:
                out.append(" ".join(cur)); cur = []
            for i in range(0, len(words), max_words):
                out.append(" ".join(words[i:i + max_words]))
            continue
        if cur and len(cur) + len(words) > max_words:
            out.append(" ".join(cur)); cur = []
        cur.extend(words)
    if cur:
        out.append(" ".join(cur))
    return out


def _struct(page: dict, target: int = 180, max_words: int = 260) -> list[str]:
    pieces = []
    for b in page["blocks"]:
        pieces.extend(_split_long(b, max_words) if len(b.split()) > max_words else [b])
    chunks, cur = [], []
    for piece in pieces:
        n = len(piece.split())
        if cur and sum(len(x.split()) for x in cur) + n > max_words:
            chunks.append(" ".join(cur)); cur = []
        cur.append(piece)
        if sum(len(x.split()) for x in cur) >= target:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        # dernier morceau trop court : on le rattache au précédent
        if chunks and sum(len(x.split()) for x in cur) < 40:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


def make_chunks(pages: list[dict], strategy: str) -> list[dict]:
    """Retourne [{chunk_id, doc_id, page, body, text}] ; text = ce qu'on indexe (en-tête + body)."""
    titles = doc_titles(pages)
    out = []

    def add(p, i, body, with_title):
        head = _header(titles.get(p["doc_id"], p["doc_id"]), p["page"]) if with_title else ""
        out.append({"chunk_id": f'{p["doc_id"]}_p{p["page"]}_{i}', "doc_id": p["doc_id"], "page": p["page"],
                    "body": body, "text": head + body})

    if strategy in ("naive", "naive_title"):
        by_key = {(p["doc_id"], p["page"]): p for p in pages}
        for i, c in enumerate(naive_chunks(pages)):
            add(by_key[(c["doc_id"], c["page"])], i, c["text"], strategy == "naive_title")
    elif strategy == "page":
        for p in pages:
            if p["text"].strip():
                add(p, 0, " ".join(p["text"].split()), True)
    elif strategy == "struct_title":
        for p in pages:
            for i, body in enumerate(_struct(p)):
                add(p, i, body, True)
    else:
        raise ValueError(f"stratégie inconnue : {strategy}")
    return out
