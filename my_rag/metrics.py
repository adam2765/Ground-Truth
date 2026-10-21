"""Étape 5 : métriques de recherche, au niveau page (doc_id, page).

`evidence` est une liste de groupes (un par document à croiser) ; dans un groupe, toutes les pages
listées sont acceptées. On suit exactement les définitions du kit (groundtruth/evaluate.py).
"""
import math


def dedup_pages(ranked: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Plusieurs chunks d'une même page : on ne garde la page qu'à sa première position."""
    seen, out = set(), []
    for r in ranked:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def recall_at_k(ranked, evidence, k):
    """Proportion des groupes de preuve dont au moins une page est dans le top k."""
    top = set(ranked[:k])
    return sum(bool(top & {(e["doc_id"], e["page"]) for e in g}) for g in evidence) / len(evidence)


def mrr(ranked, evidence):
    """1 / rang de la première page pertinente (0 si aucune)."""
    relevant = {(e["doc_id"], e["page"]) for g in evidence for e in g}
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            return 1 / i
    return 0.0


def ndcg_at_10(ranked, evidence):
    """DCG = somme des 1/log2(rang+1) des pages pertinentes du top 10, divisé par le DCG idéal."""
    relevant = {(e["doc_id"], e["page"]) for g in evidence for e in g}
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked[:10], 1) if r in relevant)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), 10) + 1))
    return dcg / idcg if idcg else 0.0


def all_metrics(ranked, evidence) -> dict:
    ranked = dedup_pages(ranked)
    return {"recall@5": recall_at_k(ranked, evidence, 5), "recall@10": recall_at_k(ranked, evidence, 10),
            "mrr": mrr(ranked, evidence), "ndcg@10": ndcg_at_10(ranked, evidence)}
