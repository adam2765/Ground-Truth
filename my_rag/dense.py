"""Étape 4 : recherche dense (par le sens).

Idée : un modèle transforme chaque texte en une liste de 384 nombres (un « embedding »).
Deux textes qui parlent de la même chose ont des listes proches. On mesure la proximité avec le
cosinus : 1 = même direction (même sens), 0 = sans rapport.

    cos(a, b) = (a · b) / (|a| * |b|)

Modèle : intfloat/multilingual-e5-small (open weights, multilingue, tourne sur CPU).
E5 attend "query: " devant la question et "passage: " devant les morceaux.
Les embeddings du corpus sont calculés une fois puis gardés dans index/ pour ne pas recommencer.
"""
from pathlib import Path

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"


class Embedder:
    """Transforme des textes en vecteurs de 384 nombres."""

    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)            # texte -> numéros de tokens
        self.model = AutoModel.from_pretrained(model_id).to(self.device)    # tokens -> vecteurs
        self.model.eval()

    def encode(self, texts, batch_size=32):
        vectors = []
        for start in range(0, len(texts), batch_size):                     # par paquets de 32 textes
            batch = texts[start:start + batch_size]
            tokens = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            tokens = tokens.to(self.device)
            with self.torch.no_grad():                                       # pas d'entraînement : pas de gradients
                out = self.model(**tokens).last_hidden_state                 # un vecteur de 384 nombres PAR token
            # On veut UN vecteur par texte : on fait la moyenne des vecteurs de ses tokens,
            # en ignorant les tokens de remplissage (padding, masque = 0).
            mask = tokens["attention_mask"].unsqueeze(-1)
            mean = (out * mask).sum(dim=1) / mask.sum(dim=1)
            vectors.append(mean.cpu().numpy())
            print(f"\r{min(start + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
        print()
        return np.concatenate(vectors)


def cosine(a, b):
    """Cosinus entre deux vecteurs."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def top_k(query_vec, vectors, k=10):
    """Les k vecteurs les plus proches de la question : [(numéro du morceau, cosinus)]."""
    sims = [cosine(query_vec, v) for v in vectors]
    ordre = sorted(range(len(sims)), key=lambda num: sims[num], reverse=True)
    return [(num, sims[num]) for num in ordre[:k]]


class DenseIndex:
    def __init__(self, chunks, cache_dir="index", model_id=MODEL_ID):
        self.embedder = Embedder(model_id)
        cache = Path(cache_dir) / f"dense_{model_id.split('/')[-1]}_{len(chunks)}.npy"
        if cache.exists():                                                   # déjà calculé : on recharge
            self.vectors = np.load(cache)
        else:                                                                # sinon on encode tout le corpus
            self.vectors = self.embedder.encode(["passage: " + c["text"] for c in chunks])
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.vectors)

    def search(self, query, k=10):
        q = self.embedder.encode(["query: " + query])[0]
        return top_k(q, self.vectors, k)


if __name__ == "__main__":
    # test sans modèle : le vecteur le plus proche de q doit être le n°42
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(1000, 384))
    q = vectors[42] + 0.01 * rng.normal(size=384)
    print(top_k(q, vectors, 3))
