"""Étape 4 : recherche dense. Embeddings d'un modèle open weights, cosinus et top-k en NumPy.

Modèle : intfloat/multilingual-e5-small (118 M paramètres, 384 dimensions, multilingue, tient
largement sur CPU). E5 attend les préfixes "query: " pour la question et "passage: " pour le texte.

L'encodage du corpus prend quelques minutes sur CPU, moins d'une minute sur GPU. Le résultat est
mis en cache dans index/dense_<modèle>.npy pour ne pas recommencer.
"""
from pathlib import Path

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**batch).last_hidden_state          # (batch, tokens, 384)
            mask = batch["attention_mask"].unsqueeze(-1).float()        # 1 = vrai token, 0 = padding
            mean = (hidden * mask).sum(1) / mask.sum(1)                 # moyenne sur les vrais tokens
            out.append(mean.cpu().numpy())
            print(f"\r{min(i + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
        print()
        return np.concatenate(out).astype(np.float32)


def normalize(m: np.ndarray) -> np.ndarray:
    """Divise chaque vecteur par sa norme : ensuite, cosinus = simple produit scalaire."""
    return m / np.linalg.norm(m, axis=-1, keepdims=True).clip(min=1e-12)


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
    """cos(q, d) = q·d / (|q| |d|), pour toutes les lignes d de la matrice, puis les k meilleures."""
    q = normalize(query_vec.reshape(1, -1))[0]
    sims = normalize(matrix) @ q                      # (N,) : un cosinus par chunk
    k = min(k, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]           # les k plus grands, dans le désordre (O(N))
    top = top[np.argsort(-sims[top])]                 # puis triés
    return [(int(i), float(sims[i])) for i in top]


class DenseIndex:
    def __init__(self, chunks: list[dict], cache_dir: str = "index", model_id: str = MODEL_ID):
        self.embedder = Embedder(model_id)
        cache = Path(cache_dir) / f"dense_{model_id.split('/')[-1]}_{len(chunks)}.npy"
        if cache.exists():
            self.matrix = np.load(cache)
        else:
            self.matrix = normalize(self.embedder.encode(["passage: " + c["text"] for c in chunks]))
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.matrix)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = self.embedder.encode(["query: " + query])[0]
        return cosine_topk(q, self.matrix, k)


if __name__ == "__main__":
    # test de cosine_topk sans modèle : le vecteur le plus proche de q doit sortir premier
    rng = np.random.default_rng(0)
    m = rng.normal(size=(1000, 384))
    q = m[42] + 0.01 * rng.normal(size=384)
    print(cosine_topk(q, m, 3))
