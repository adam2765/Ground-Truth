"""Recherche hybride : BM25 (bm25s, avec racinisation française) + dense (bge-m3), fusion RRF, puis
reranking (bge-reranker-v2-m3) des meilleurs candidats.

Modes (pour l'ablation) :
  bm25           : BM25 seul
  dense          : embeddings seuls
  hybrid         : fusion RRF des deux classements
  hybrid_rerank  : hybrid, puis le cross-encoder relit les 40 premiers avec la question

RRF (Reciprocal Rank Fusion) : score(chunk) = somme sur les classements de 1 / (60 + rang).
Pas besoin de rendre comparables un score BM25 (0 à 40) et un cosinus (0 à 1) : seuls les rangs comptent.
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

DENSE_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RRF_K = 60

STOPWORDS = set("""
a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me meme mes moi mon ne
nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une vos votre vous c d j l
m n s t y ete etre est sont ont a ai as avait avaient cette cet celle celui ceux dont lors entre plus moins
quel quelle quels quelles combien comment quoi apres avant depuis selon
the of and in to is are what how many which was were for on by with from
""".split())

_stemmer = None


def _stem(words):
    global _stemmer
    if _stemmer is None:
        import Stemmer
        _stemmer = Stemmer.Stemmer("french")
    return _stemmer.stemWords(words)


def _strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def tokenize(text: str) -> list[str]:
    """Minuscules, mots, sans mots vides, racinisés (« exploitants » -> « exploit »), sans accents."""
    words = re.findall(r"\w+", text.lower())
    words = [w for w in words if _strip_accents(w) not in STOPWORDS]
    return [_strip_accents(w) for w in _stem(words)]


class BM25Index:
    def __init__(self, texts: list[str]):
        import bm25s

        self.bm25 = bm25s.BM25(k1=1.5, b=0.75)
        self.bm25.index([tokenize(t) for t in texts], show_progress=False)
        self.n = len(texts)

    def search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseEncoder:
    def __init__(self, model_id: str = DENSE_MODEL):
        from sentence_transformers import SentenceTransformer

        dev = _device()
        self.model = SentenceTransformer(model_id, device=dev)
        if dev == "cuda":
            self.model.half()
        self.model.max_seq_length = 512

    def encode(self, texts: list[str], batch_size: int = 32, progress: bool = True) -> np.ndarray:
        v = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=progress)
        return v.astype(np.float16)


class Reranker:
    def __init__(self, model_id: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        dev = _device()
        self.model = CrossEncoder(model_id, max_length=512, device=dev)
        if dev == "cuda":
            self.model.half()              # CrossEncoder est un nn.Module : tout passe en float16

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        s = self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        return [float(x) for x in np.asarray(s).reshape(-1)]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


class Retriever:
    """Charge un index (une stratégie de chunking) et cherche dans tous les modes."""

    def __init__(self, index_dir: str, strategy: str, use_dense: bool = True, use_rerank: bool = True,
                 encoder: "DenseEncoder | None" = None, reranker: "Reranker | None" = None):
        d = Path(index_dir) / strategy
        self.chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.bm25 = BM25Index([c["text"] for c in self.chunks])
        self.vectors = None
        self.encoder = encoder
        self.reranker = reranker
        if use_dense:
            self.vectors = np.load(d / "dense.npy").astype(np.float32)
            self.encoder = self.encoder or DenseEncoder()
        if use_rerank:
            self.reranker = self.reranker or Reranker()

    def dense_search(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        q = self.encoder.encode([query], progress=False)[0].astype(np.float32)
        sims = self.vectors @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 20, n_rerank: int = 40) -> list[dict]:
        """Retourne les k meilleurs chunks : [{idx, score, bm25, dense, rerank}] (score = critère du mode)."""
        bm = self.bm25.search(query, 100)
        if mode == "bm25":
            return [{"idx": i, "score": s, "bm25": s} for i, s in bm[:k]]
        de = self.dense_search(query, 100)
        if mode == "dense":
            return [{"idx": i, "score": s, "dense": s} for i, s in de[:k]]
        bm_s, de_s = dict(bm), dict(de)
        if is_english(query):
            # question en anglais sur un corpus français : le BM25 (mots racinisés en français) ne trouve presque
            # rien d'utile et ses résultats pollueraient la fusion. On reranke surtout les résultats denses
            # (bge-m3 est multilingue), plus les 5 premiers BM25 (noms propres, sigles, chiffres communs).
            seen, cand_en = set(), []
            for i in [i for i, _ in de[: n_rerank - 5]] + [i for i, _ in bm[:5]]:
                if i not in seen:
                    seen.add(i)
                    cand_en.append(i)
            fused = [(i, 1.0 / (60 + r)) for r, i in enumerate(cand_en)]
        else:
            fused = rrf([[i for i, _ in bm], [i for i, _ in de]])
        if mode == "hybrid":
            return [{"idx": i, "score": s, "bm25": bm_s.get(i), "dense": de_s.get(i)} for i, s in fused[:k]]
        cand = [i for i, _ in fused[:n_rerank]]
        rs = self.reranker.score(query, [self.chunks[i]["text"] for i in cand])
        order = sorted(zip(cand, rs), key=lambda x: -x[1])
        return [{"idx": i, "score": s, "rerank": s, "bm25": bm_s.get(i), "dense": de_s.get(i)} for i, s in order[:k]]


_SPLIT = re.compile(r"\s*(?:,\s*(?:et\s+|puis\s+)?|\s(?:et|puis)\s+)(?=(?:combien|quel|quelle|quels|quelles|que|qu['’]|à combien"
                    r"|de combien|selon|d['’]après|dans|pour)\b)|\s*;\s*|\?\s+(?=\S)", re.I)
_ASK = re.compile(r"\b(combien|quel|quelle|quels|quelles|où|comment|pourquoi|que|qu['’])\b", re.I)


def split_question(question: str, min_words: int = 6) -> list[str]:
    """Questions en plusieurs parties (« combien ... selon le rapport X, et quelle ... selon le rapport Y ? ») :
    une sous-question par partie. Les parties trop courtes (« combien de points les séparent ») sont ignorées.
    Retourne [question] si la question n'a qu'une partie."""
    parts = [p.strip(" ,?") for p in _SPLIT.split(question)]
    # une vraie sous-question contient son propre mot interrogatif (sinon c'est un complément :
    # « Parmi les immigrés entrés en 2014, quelle proportion... » reste une seule question)
    def asks(p):                                       # mot interrogatif dans les 3 premiers mots
        p = re.sub(r"^(selon|d['’]après)[^,]*,\s*", "", p, flags=re.I)
        return bool(_ASK.search(" ".join(p.split()[:3])))
    parts = [p for p in parts if len(p.split()) >= min_words and asks(p)]
    return parts if len(parts) >= 2 else [question]


def search_multi(retriever: "Retriever", question: str, mode: str = "hybrid_rerank", k: int = 20,
                 per_part: int = 2) -> list[dict]:
    """Recherche « multi-parties » (run 6). Une seule recherche sur une question qui porte sur deux rapports
    ramène surtout le rapport le plus proche des mots de la question : l'autre partie manque au contexte.
    On cherche aussi chaque partie séparément et on place ses `per_part` meilleurs extraits juste après le
    meilleur extrait de la question entière (qui reste en tête : son score sert au seuil d'abstention)."""
    main = retriever.search(question, mode=mode, k=k)
    parts = split_question(question)
    if len(parts) < 2 or not main:
        return main
    extra = [retriever.search(p, mode=mode, k=per_part) for p in parts]
    out, seen = [main[0]], {main[0]["idx"]}
    for r in range(per_part):
        for hits in extra:
            if r < len(hits) and hits[r]["idx"] not in seen:
                seen.add(hits[r]["idx"])
                out.append({**hits[r], "part": True})
    out += [h for h in main[1:] if h["idx"] not in seen]
    return out[:k]


def add_newer_editions(r: "Retriever", query: str, hits: list[dict], docs: dict, top_hits: int = 5,
                       max_add: int = 2, min_score: float = 0.3, drop_old: bool = True,
                       year_keep: bool = False) -> list[dict]:
    """Palier 3 (graphe des éditions) : si un des meilleurs extraits vient d'un document dont une édition plus
    récente est dans l'index, on ajoute juste après lui le meilleur extrait de cette édition récente (choisi par
    le cross-encoder parmi ses 5 chunks les plus proches en embeddings). Le modèle voit ainsi les deux dates et
    applique la consigne « la publication la plus récente fait foi »."""
    if not hasattr(r, "_by_doc"):
        r._by_doc = {}
        for i, c in enumerate(r.chunks):
            r._by_doc.setdefault(c["doc_id"], []).append(i)
    present = {r.chunks[h["idx"]]["doc_id"] for h in hits}
    out, added, qv, replaced = list(hits), 0, None, set()
    for h in hits[:top_hits]:
        for nd in docs.get(r.chunks[h["idx"]]["doc_id"], {}).get("newer", []):
            if nd in present:                           # l'édition récente est déjà dans les résultats
                replaced.add(r.chunks[h["idx"]]["doc_id"])
                continue
            if nd not in r._by_doc or added >= max_add:
                continue
            idxs = r._by_doc[nd]
            if qv is None:
                qv = r.encoder.encode([query], progress=False)[0].astype(np.float32)
            near = sorted(idxs, key=lambda i: -float(r.vectors[i] @ qv))[:5]
            sc = r.reranker.score(query, [r.chunks[i]["text"] for i in near])
            best = max(range(len(near)), key=lambda k: sc[k])
            if sc[best] < min_score:                    # l'édition récente ne parle pas de la question
                continue
            out.insert(out.index(h) + 1, {"idx": near[best], "score": sc[best], "rerank": sc[best], "edition": True})
            present.add(nd)
            added += 1
            replaced.add(r.chunks[h["idx"]]["doc_id"])
    if year_keep and not _RECENT.search(query):
        # Question datée sans « dernier / plus récent » : on garde l'édition remplacée si elle est la seule à porter
        # sur l'année demandée (« en 2023, combien gagne un enseignant ? » : l'édition 2025 porte sur 2023, la
        # nouvelle sur 2024). Si la nouvelle édition porte sur la même année (compte prévisionnel 2025 -> provisoire
        # 2025), l'ancienne est bien remplacée.
        years = {int(y) for y in _YEAR.findall(query)}
        replaced = {d for d in replaced if not (ref_year(docs.get(d, {})) in years and ref_year(docs.get(d, {}))
                    < min((ref_year(docs.get(n, {})) or 9999) for n in docs.get(d, {}).get("newer", [])))}
    if replaced and drop_old and not _EVOLUTION.search(query):
        # VersionRAG (2025) : pour une question sur le contenu (pas sur une évolution d'une édition à l'autre),
        # seule l'édition la plus récente fait foi -> on retire les extraits des éditions remplacées, pour que le
        # modèle 3B n'ait pas deux chiffres contradictoires sous les yeux.
        out = [x for x in out if r.chunks[x["idx"]]["doc_id"] not in replaced]
    return out[: len(hits)]


_EVOLUTION = re.compile(
    r"évolu|révis|revu|d['’]une édition|édition précédente|précédent|successi|initial|auparavant|annonçait|"
    r"puis |depuis|entre .* et|changé|changer|revised|previous|earlier|initially|successive|trace|changed|since|"
    r"first estimate|from .* to", re.I)


_YEAR = re.compile(r"\b(?:19|20)\d\d\b")


def ref_year(doc: dict) -> int | None:
    """Année sur laquelle porte un document : la plus récente citée dans son titre (« LES COMPTES DE LA NATION EN
    2025 », « En 2023, un agent… »)."""
    ys = [int(y) for y in _YEAR.findall(doc.get("title", ""))]
    return max(ys) if ys else None
_RECENT = re.compile(r"derni|plus récent|actuel|aujourd|latest|most recent|current|now\b", re.I)


_EN_WORDS = re.compile(r"\b(the|what|which|how|many|much|in|of|did|does|was|were|is|are|according|between)\b", re.I)
_FR_WORDS = re.compile(r"\b(le|la|les|des|du|quel|quelle|combien|en|est|sont|selon|entre|dans)\b", re.I)


def is_english(question: str) -> bool:
    """Palier 3 : questions en anglais sur un corpus en français."""
    return len(_EN_WORDS.findall(question)) > len(_FR_WORDS.findall(question))


def split_units(body: str, max_words: int = 60) -> list[str]:
    """Découpe un extrait en « lignes » : rangées de tableau, puis phrases pour les longs paragraphes."""
    units = []
    for line in body.split("\n"):
        line = " ".join(line.split())
        if len(line.split()) < 3:
            continue
        if len(line.split()) > max_words:
            units.extend(x for x in re.split(r"(?<=[.!?;])\s+", line) if len(x.split()) >= 3)
        else:
            units.append(line)
    return units


def key_lines(reranker: "Reranker", query: str, chunks: list[dict], hits: list[dict], n_chunks: int = 8,
              top: int = 3) -> list[dict]:
    """Les lignes (phrases, rangées de tableau) des meilleurs extraits les plus proches de la question,
    notées par le cross-encoder. Elles sont placées en tête du prompt pour guider le petit modèle."""
    cand, seen = [], set()
    for h in hits[:n_chunks]:
        c = chunks[h["idx"]]
        for u in split_units(c["body"]):
            if u not in seen:
                seen.add(u)
                cand.append({"doc_id": c["doc_id"], "page": c["page"], "text": u})
    if not cand:
        return []
    scores = reranker.score(query, [x["text"] for x in cand])
    for x, sc in zip(cand, scores):
        x["score"] = sc
    return sorted(cand, key=lambda x: -x["score"])[:top]


def table_rows(reranker: "Reranker", query: str, chunks: list[dict], hits: list[dict], tables: dict,
               n_chunks: int = 5, top: int = 4) -> list[dict]:
    """Rangées des tableaux des meilleures pages, chacune écrite seule avec ses en-têtes
    (« Europe du Sud | 2006 : 11 | 2014 : 16 ... »), notées par le cross-encoder (run 9 : 3e appel ciblé)."""
    cand, seen = [], set()
    for h in hits[:n_chunks]:
        c = chunks[h["idx"]]
        key = f"{c['doc_id']}|{c['page']}"
        if key in seen or key not in tables:
            continue
        seen.add(key)
        for t in tables[key]:
            for label, vals in t["rows"]:
                txt = label + " | " + " | ".join(f"{t['heads'][j]} : {v}" for j, v in vals)
                cand.append({"doc_id": c["doc_id"], "page": c["page"], "text": txt})
    if not cand:
        return []
    for x, sc in zip(cand, reranker.score(query, [x["text"] for x in cand])):
        x["score"] = sc
    return sorted(cand, key=lambda x: -x["score"])[:top]


def focus_lines(reranker: "Reranker", query: str, chunks: list[dict], hits: list[dict], tables: dict,
                n_chunks: int = 8, top: int = 6) -> list[dict]:
    """Run 10 : les `top` unités les plus proches de la question, phrases (split_units) et rangées de tableau
    (avec leurs en-têtes) confondues, notées par le cross-encoder. Sert au 3e appel « ciblé »."""
    cand, seen = [], set()
    for h in hits[:n_chunks]:
        c = chunks[h["idx"]]
        for u in split_units(c["body"]):
            if u not in seen:
                seen.add(u)
                cand.append({"doc_id": c["doc_id"], "page": c["page"], "text": u})
    pages = []
    for h in hits[:5]:
        c = chunks[h["idx"]]
        key = f"{c['doc_id']}|{c['page']}"
        if key in tables and key not in pages:
            pages.append(key)
            for t in tables[key]:
                for label, vals in t["rows"]:
                    cand.append({"doc_id": c["doc_id"], "page": c["page"],
                                 "text": label + " | " + " | ".join(f"{t['heads'][j]} : {v}" for j, v in vals)})
    if not cand:
        return []
    for x, sc in zip(cand, reranker.score(query, [x["text"] for x in cand])):
        x["score"] = sc
    return sorted(cand, key=lambda x: -x["score"])[:top]


def ranked_pages(chunks: list[dict], hits: list[dict]) -> list[dict]:
    """Pages distinctes, dans l'ordre de leur meilleur chunk."""
    seen, out = set(), []
    for h in hits:
        c = chunks[h["idx"]]
        key = (c["doc_id"], c["page"])
        if key not in seen:
            seen.add(key)
            out.append({"doc_id": c["doc_id"], "page": c["page"]})
    return out
