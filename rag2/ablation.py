"""Ablation de la recherche sur validation_v1 : stratégies de chunking × modes de recherche.

    python -m rag2.ablation --index-dir index2 --questions data/validation_v1.json
    python -m rag2.ablation --index-dir index2 --modes bm25            # sans GPU

Mesures au niveau page (Recall@5, Recall@10, MRR, nDCG@10) sur les questions qui ont une réponse,
avec les fonctions du palier 1 (identiques à l'évaluateur officiel). Écrit runs/ablation_retrieval.json
et <index>/best.json (meilleure combinaison selon Recall@5, puis MRR).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from groundtruth.io import load_questions
from my_rag.metrics import all_metrics
from rag2.search import Retriever

MODES = ["bm25", "dense", "hybrid", "hybrid_rerank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="index2")
    ap.add_argument("--questions", default="data/validation_v1.json")
    ap.add_argument("--strategies", default=None, help="par défaut : toutes celles présentes dans l'index")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--out", default="runs/ablation_retrieval.json")
    args = ap.parse_args()

    idx = Path(args.index_dir)
    strategies = args.strategies.split(",") if args.strategies else sorted(
        p.name for p in idx.iterdir() if (p / "chunks.json").exists())
    modes = args.modes.split(",")
    need_dense = any(m != "bm25" for m in modes)
    need_rerank = "hybrid_rerank" in modes
    qs = [q for q in load_questions(args.questions) if q["answerable"]]

    encoder = reranker = None
    rows = []
    for strat in strategies:
        if need_dense and not (idx / strat / "dense.npy").exists():
            print(f"[{strat}] pas d'embeddings, ignoré pour les modes denses")
        r = Retriever(args.index_dir, strat, use_dense=need_dense and (idx / strat / "dense.npy").exists(),
                      use_rerank=need_rerank, encoder=encoder, reranker=reranker)
        encoder, reranker = r.encoder, r.reranker
        for mode in modes:
            if mode != "bm25" and r.vectors is None:
                continue
            per_q, by_type = [], defaultdict(list)
            for q in qs:
                hits = r.search(q["question"], mode=mode, k=40)
                ranked = [(r.chunks[h["idx"]]["doc_id"], r.chunks[h["idx"]]["page"]) for h in hits]
                m = all_metrics(ranked, q["evidence"])
                per_q.append(m)
                by_type[q["type"]].append(m["recall@5"])
            row = {"strategy": strat, "mode": mode, "n_chunks": len(r.chunks),
                   **{k: round(float(np.mean([m[k] for m in per_q])), 3) for k in per_q[0]},
                   "recall@5_par_type": {t: round(float(np.mean(v)), 2) for t, v in sorted(by_type.items())}}
            rows.append(row)
            print(f"{strat:13s} {mode:14s} R@5 {row['recall@5']:.3f}  R@10 {row['recall@10']:.3f}  "
                  f"MRR {row['mrr']:.3f}  nDCG@10 {row['ndcg@10']:.3f}", flush=True)

    print("\n| Chunking | Recherche | Chunks | Recall@5 | Recall@10 | MRR | nDCG@10 |\n|---|---|---|---|---|---|---|")
    for r_ in rows:
        print(f"| {r_['strategy']} | {r_['mode']} | {r_['n_chunks']} | {r_['recall@5']:.3f} | {r_['recall@10']:.3f} "
              f"| {r_['mrr']:.3f} | {r_['ndcg@10']:.3f} |")
    best = max(rows, key=lambda r_: (r_["recall@5"], r_["mrr"]))
    print(f"\nMeilleure combinaison : {best['strategy']} / {best['mode']}")
    print("Recall@5 par type :", best["recall@5_par_type"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (idx / "best.json").write_text(json.dumps({"strategy": best["strategy"], "mode": best["mode"]}))


if __name__ == "__main__":
    main()
