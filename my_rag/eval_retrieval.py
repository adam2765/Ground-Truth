"""Mesure la recherche seule sur validation_v1 (questions avec réponse) : Recall@5/10, MRR, nDCG@10.

    python -m my_rag.eval_retrieval --method bm25
    python -m my_rag.eval_retrieval --method dense
    python -m my_rag.eval_retrieval --method bm25 --unit page      # une page = un chunk (comme la baseline)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from groundtruth.io import load_questions
from my_rag.chunk import chunk_pages
from my_rag.extract import extract_pages
from my_rag.metrics import all_metrics


def build_retriever(method: str, chunks: list[dict]):
    if method == "bm25":
        from my_rag.bm25 import BM25
        return BM25().index([c["text"] for c in chunks])
    from my_rag.dense import DenseIndex
    return DenseIndex(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--unit", choices=["chunk", "page"], default="chunk")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--questions", default="data/validation_v1.json")
    args = ap.parse_args()

    pages = extract_pages("corpus/pdf")
    if args.unit == "page":
        chunks = [p for p in pages if p["text"].strip()]
    else:
        chunks = chunk_pages(pages, args.size, args.overlap)
    retriever = build_retriever(args.method, chunks)

    questions = [q for q in load_questions(args.questions) if q["answerable"]]
    rows = []
    for q in questions:
        hits = retriever.search(q["question"], k=50)          # 50 chunks pour avoir ~10 pages distinctes
        ranked = [(chunks[i]["doc_id"], chunks[i]["page"]) for i, _ in hits]
        m = all_metrics(ranked, q["evidence"])
        rows.append({"qid": q["qid"], "type": q["type"], **m, "top3": ranked[:3], "evidence": q["evidence"]})

    print(f"{args.method} / {args.unit}" + (f" ({args.size} mots, recouvrement {args.overlap})" if args.unit == "chunk" else ""),
          f"- {len(chunks)} unités, {len(rows)} questions")
    for k in ["recall@5", "recall@10", "mrr", "ndcg@10"]:
        print(f"  {k:10s} {np.mean([r[k] for r in rows]):.3f}")
    print("  Recall@5 par type :")
    for t in sorted({r["type"] for r in rows}):
        sub = [r["recall@5"] for r in rows if r["type"] == t]
        print(f"    {t:10s} {np.mean(sub):.2f}  (n={len(sub)})")

    out = Path("runs") / f"retrieval_{args.method}_{args.unit}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  détail par question : {out}")


if __name__ == "__main__":
    main()
