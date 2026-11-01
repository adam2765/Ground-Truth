"""Métadonnées des documents et graphe des éditions (palier 3 : « quand deux éditions se contredisent, la plus
récente fait foi »).

Version légère de GraphRAG : un nœud par document, des arêtes « édition plus récente » entre documents d'une même
collection qui traitent du même sujet. Pas de modèle de langage : la date vient du manifeste du corpus
(corpus/manifest.json ; à défaut, la première date trouvée en page 1), le sujet du titre (début de la page 1),
comparé par embeddings bge-m3. Construit en quelques secondes, donc compatible avec les 20 min de `run.sh update`.

    docs.json : {doc_id: {"date": "2026-05-06", "collection": "...", "title": "...", "newer": [...], "older": [...]}}
"""
import json
import re
from pathlib import Path

import numpy as np

MANIFEST = Path(__file__).resolve().parents[1] / "corpus" / "manifest.json"
SAME_TOPIC = 0.80          # cosinus minimal entre titres (bge-m3) pour deux éditions d'un même sujet
_DATE = re.compile(r"\b(?:(\d{1,2})[/ .-](\d{1,2}|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|"
                   r"novembre|décembre)[/ .-])?((?:19|20)\d\d)\b", re.I)


def _family(collection: str) -> str:
    return "insee" if "insee" in (collection or "").lower() else "cc"


def build(pages: list[dict], encoder=None, old: dict | None = None) -> dict:
    """Métadonnées de tous les documents de `pages` (en gardant celles de `old`), puis arêtes d'éditions."""
    man = {}
    if MANIFEST.exists():
        man = {d["doc_id"]: d for d in json.loads(MANIFEST.read_text(encoding="utf-8"))["documents"]}
    docs = dict(old or {})
    first = {}
    for p in sorted(pages, key=lambda p: (p["doc_id"], p["page"])):
        if p["doc_id"] not in first and p["text"].strip():
            first[p["doc_id"]] = p["text"]
    for d, text in first.items():
        if d in docs:
            continue
        m = man.get(d, {})
        date = m.get("date")
        if not date:                                   # document hors manifeste : année trouvée en page 1
            years = [int(y.group(3)) for y in _DATE.finditer(text[:3000]) if 1990 <= int(y.group(3)) <= 2030]
            date = f"{max(years)}-01-01" if years else ""
        coll = m.get("collection") or ("Insee Première" if d.startswith("insee") else "Rapport")
        docs[d] = {"date": date, "collection": coll, "title": " ".join(text.split()[:30]), "newer": [], "older": []}
    link(docs, encoder)
    return docs


def link(docs: dict, encoder=None) -> None:
    """Arêtes « édition plus récente » : même famille (Insee / Cour des comptes), titres proches, dates différentes.
    Un rapport et sa synthèse (même date) ne sont pas des éditions."""
    ids = sorted(docs)
    for d in ids:
        docs[d]["newer"], docs[d]["older"] = [], []
    if len(ids) < 2:
        return
    if encoder is not None:
        v = encoder.encode([docs[d]["title"] for d in ids], progress=False).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        sim, thr = v @ v.T, SAME_TOPIC
    else:                                              # sans modèle : mots communs des titres (racinisés)
        from rag2.search import tokenize
        toks = [set(tokenize(docs[d]["title"])) for d in ids]
        sim = np.array([[len(a & b) / max(1, len(a | b)) for b in toks] for a in toks])
        thr = 0.5
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            A, B = docs[a], docs[b]
            if (i == j or _family(A["collection"]) != _family(B["collection"]) or not A["date"] or not B["date"]
                    or A["date"] >= B["date"] or sim[i, j] < thr):
                continue
            A["newer"].append(b)
            B["older"].append(a)


def save(docs: dict, index_dir) -> None:
    (Path(index_dir) / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")


def load(index_dir) -> dict:
    p = Path(index_dir) / "docs.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
