"""Vérifie notre BM25 contre bm25s (autorisé ici uniquement comme référence) et nos métriques
contre celles de l'évaluateur officiel."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import bm25s
import numpy as np

from groundtruth.evaluate import retrieval_metrics
from groundtruth.io import load_questions
from my_rag.bm25 import BM25, tokenize
from my_rag.chunk import chunk_pages
from my_rag.extract import extract_pages
from my_rag.metrics import all_metrics

chunks = chunk_pages(extract_pages("corpus/pdf"))
texts = [c["text"] for c in chunks]
ours = BM25().index(texts)
ref = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
ref.index([tokenize(t) for t in texts], show_progress=False)

questions = [q for q in load_questions("data/validation_v1.json") if q["answerable"]]
max_err, same_top10 = 0.0, 0
for q in questions:
    # bm25s ("lucene") omet le facteur constant (k1 + 1) du numérateur : on l'enlève pour comparer.
    # Un facteur constant ne change pas le classement.
    a = np.array(ours.scores(q["question"])) / (ours.k1 + 1)
    b = ref.get_scores(tokenize(q["question"]))
    max_err = max(max_err, float(np.abs(a - b).max()))
    same_top10 += set(np.argsort(-a)[:10]) == set(np.argsort(-b)[:10])
print(f"BM25 : écart max de score avec bm25s = {max_err:.2e}, top-10 identique sur {same_top10}/{len(questions)} questions")

# métriques : on compare nos fonctions à celles du kit sur les mêmes classements
diff = 0.0
for q in questions:
    ranked = [(chunks[i]["doc_id"], chunks[i]["page"]) for i, _ in ours.search(q["question"], k=50)]
    m_ours = all_metrics(ranked, q["evidence"])
    m_ref = retrieval_metrics([{"doc_id": d, "page": p} for d, p in ranked], q["evidence"])
    diff = max(diff, max(abs(m_ours[k] - m_ref[k]) for k in m_ref))
print(f"Métriques : écart max avec l'évaluateur officiel = {diff:.2e}")
