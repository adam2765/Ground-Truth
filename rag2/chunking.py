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


def _naive_keep_lines(text: str, size: int = 200, overlap: int = 50) -> list[str]:
    """Mêmes fenêtres que le chunker naïf du palier 1 (200 mots, recouvrement 50), mais on garde la mise en
    forme d'origine (retours à la ligne) : sinon la fin d'une rangée de tableau se colle au début de la
    suivante (« ... | 72 Grands centres urbains | 46 742 ... ») et le modèle se trompe de ligne."""
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    out, step = [], size - overlap
    for start in range(0, len(spans), step):
        piece = spans[start:start + size]
        out.append(text[piece[0][0]:piece[-1][1]])
        if start + size >= len(spans):
            break
    return out


def make_chunks(pages: list[dict], strategy: str) -> list[dict]:
    """Retourne [{chunk_id, doc_id, page, body, text}] ; text = ce qu'on indexe (en-tête + body)."""
    titles = doc_titles(pages)
    out = []

    def add(p, i, body, with_title):
        head = _header(titles.get(p["doc_id"], p["doc_id"]), p["page"]) if with_title else ""
        out.append({"chunk_id": f'{p["doc_id"]}_p{p["page"]}_{i}', "doc_id": p["doc_id"], "page": p["page"],
                    "body": body, "text": head + body})

    if strategy in ("naive", "naive_title"):
        for p in pages:
            for i, body in enumerate(_naive_keep_lines(p["text"])):
                add(p, i, body, strategy == "naive_title")
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
