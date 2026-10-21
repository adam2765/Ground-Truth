"""Étape 3 : BM25 (Okapi) codé à la main : tokenisation, index inversé, scoring.

Formule (variante « Lucene », celle de bm25s par défaut) pour une requête q et un chunk d :

    score(q, d) = somme sur les termes t de q de  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = log(1 + (N - df + 0.5) / (df + 0.5))

    tf    : nombre d'occurrences de t dans d
    df    : nombre de chunks qui contiennent t
    N     : nombre de chunks
    |d|   : longueur de d en tokens ; avgdl : longueur moyenne
    k1    : saturation de tf (1.5) ; b : normalisation par la longueur (0.75)
"""
import math
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np


def tokenize(text: str) -> list[str]:
    """Minuscules, accents retirés, on garde les suites de lettres/chiffres."""
    text = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", text)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def index(self, texts: list[str]) -> "BM25":
        docs = [tokenize(t) for t in texts]
        self.N = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = self.doc_len.mean()
        # index inversé : terme -> liste de (numéro du chunk, tf)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        # partie du dénominateur qui ne dépend que du chunk, calculée une fois
        self.norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        return self

    def scores(self, query: str) -> np.ndarray:
        """Score de la requête contre tous les chunks (0 pour ceux qui n'ont aucun terme)."""
        s = np.zeros(self.N)
        for term in tokenize(query):           # un terme répété dans la requête compte plusieurs fois
            if term not in self.postings:
                continue
            idx, tf = zip(*self.postings[term])
            idx, tf = np.array(idx), np.array(tf, dtype=np.float64)
            s[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm[idx])
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Les k meilleurs chunks : [(numéro du chunk, score)], du meilleur au moins bon."""
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]
