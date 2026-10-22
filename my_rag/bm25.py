"""Étape 3 : BM25 codé à la main.

Idée : on donne des points à chaque morceau pour chaque mot de la question qu'il contient.
  - un mot rare dans le corpus rapporte plus de points (idf) ;
  - un mot répété dans le morceau rapporte plus, mais de moins en moins (k1) ;
  - un morceau court a un léger avantage sur un morceau long (b).

Formule pour un mot t et un morceau d :
    points = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * longueur(d) / longueur_moyenne))
    idf(t) = log(1 + (N - df + 0.5) / (df + 0.5))
    tf : nombre de fois où t apparaît dans d ; df : nombre de morceaux qui contiennent t ; N : nombre de morceaux
"""
import math
import re
import unicodedata


def tokenize(text):
    """"Le Taux de Chômage !" -> ["le", "taux", "de", "chomage"]"""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()   # enlève les accents
    return re.findall(r"[a-z0-9]+", text)                                            # garde les mots


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1      # saturation : à partir de combien de répétitions un mot ne rapporte presque plus
        self.b = b        # importance de la longueur du morceau (0 = aucune, 1 = maximale)

    def index(self, texts):
        """Prépare la recherche : à faire une seule fois, sur tous les morceaux."""
        docs = [tokenize(t) for t in texts]
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.N

        # Index inversé : pour chaque mot, la liste des (numéro du morceau, nombre d'occurrences).
        # Ex. self.postings["chomage"] = [(12, 3), (845, 1)]
        self.postings = {}
        for num, words in enumerate(docs):
            counts = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            for w, tf in counts.items():
                self.postings.setdefault(w, []).append((num, tf))

        # Rareté de chaque mot : df = nombre de morceaux qui le contiennent
        self.idf = {}
        for w, plist in self.postings.items():
            df = len(plist)
            self.idf[w] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
        return self

    def scores(self, query):
        """Score de la question pour chacun des N morceaux (0 si aucun mot en commun)."""
        scores = [0.0] * self.N
        for w in tokenize(query):
            if w not in self.postings:            # mot absent de tout le corpus : il ne rapporte rien
                continue
            for num, tf in self.postings[w]:      # seulement les morceaux qui contiennent ce mot
                longueur = self.k1 * (1 - self.b + self.b * self.doc_len[num] / self.avgdl)
                points = self.idf[w] * tf * (self.k1 + 1) / (tf + longueur)
                scores[num] += points
        return scores

    def search(self, query, k=10):
        """Les k meilleurs morceaux : [(numéro du morceau, score)], du meilleur au moins bon."""
        scores = self.scores(query)
        ordre = sorted(range(self.N), key=lambda num: scores[num], reverse=True)
        return [(num, scores[num]) for num in ordre[:k] if scores[num] > 0]
