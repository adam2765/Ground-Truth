"""Pipeline du palier 2.

    python -m rag2.pipeline index    --pdf-dir corpus/pdf --index-dir index2
    python -m rag2.pipeline answer   --index-dir index2 --questions data/validation_v1.json --out runs/val_raw.json
    (answer = retrieve puis generate, dans deux processus séparés pour libérer la VRAM entre les deux)

    python -m rag2.pipeline retrieve --index-dir index2 --questions Q.json --out runs/ret.json
    python -m rag2.pipeline generate --index-dir index2 --retrieval runs/ret.json --questions Q.json --out runs/raw.json

La stratégie de chunking et le mode de recherche par défaut sont lus dans <index>/best.json (écrit par
rag2.ablation), sinon struct_title + hybrid_rerank.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rag2.answer import DEFAULT_VARIANT, VARIANTS
from rag2.chunking import STRATEGIES, make_chunks

DEFAULT = {"strategy": "struct_title", "mode": "hybrid_rerank"}


def defaults(index_dir: str) -> dict:
    p = Path(index_dir) / "best.json"
    return {**DEFAULT, **(json.loads(p.read_text()) if p.exists() else {})}


def cmd_index(args):
    from rag2.parse import read_pdf_dir

    idx = Path(args.index_dir)
    idx.mkdir(parents=True, exist_ok=True)
    pages_path = idx / "pages.json"
    t0 = time.time()
    if pages_path.exists() and not args.reparse:
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
        print(f"{len(pages)} pages relues depuis {pages_path}")
    else:
        pages = read_pdf_dir(args.pdf_dir, ocr=not args.no_ocr, ocr_images=not args.no_ocr_images)
        pages_path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        print(f"{len(pages)} pages lues en {time.time() - t0:.0f} s")

    from rag2.describe_charts import add_descriptions
    n_desc = add_descriptions(pages)                   # descriptions de graphiques en cache (modèle de vision)
    if n_desc:
        print(f"{n_desc} pages enrichies par la description de leurs graphiques")

    encoder = None
    for strat in args.strategies.split(","):
        d = idx / strat
        d.mkdir(exist_ok=True)
        chunks = make_chunks(pages, strat)
        (d / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        print(f"[{strat}] {len(chunks)} chunks")
        if not args.no_dense and not (d / "dense.npy").exists():
            from rag2.search import DenseEncoder
            encoder = encoder or DenseEncoder()
            t = time.time()
            np.save(d / "dense.npy", encoder.encode([c["text"] for c in chunks]))
            print(f"[{strat}] embeddings en {time.time() - t:.0f} s")


def cmd_retrieve(args):
    from groundtruth.io import load_questions
    from rag2.search import Retriever, key_lines, ranked_pages

    cfg = defaults(args.index_dir)
    strategy, mode = args.strategy or cfg["strategy"], args.mode or cfg["mode"]
    qs = load_questions(args.questions)
    if Path(args.out).exists():                         # reprise après une coupure : déjà fait
        old = json.loads(Path(args.out).read_text(encoding="utf-8"))
        if old.get("strategy") == strategy and old.get("mode") == mode and len(old["items"]) == len(qs):
            print(f"recherche déjà faite -> {args.out}")
            return
    r = Retriever(args.index_dir, strategy, use_dense=mode != "bm25", use_rerank=mode == "hybrid_rerank")
    items, t0 = [], time.time()
    for n, q in enumerate(qs, 1):
        hits = r.search(q["question"], mode=mode, k=20)
        item = {"qid": q["qid"], "hits": hits, "retrieved": ranked_pages(r.chunks, hits)}
        if r.reranker is not None:
            item["key_lines"] = key_lines(r.reranker, q["question"], r.chunks, hits)
        items.append(item)
        if n % 20 == 0:
            print(f"  {n}/{len(qs)} ({(time.time() - t0) / n:.2f} s/question)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"strategy": strategy, "mode": mode, "items": items}, ensure_ascii=False),
                              encoding="utf-8")
    print(f"recherche : {strategy} / {mode}, {len(items)} questions -> {args.out}")


def cmd_generate(args):
    from groundtruth.generator import OfficialGenerator
    from groundtruth.io import load_questions, write_predictions
    from rag2.answer import answer_one

    ret = json.loads(Path(args.retrieval).read_text(encoding="utf-8"))
    chunks = json.loads((Path(args.index_dir) / ret["strategy"] / "chunks.json").read_text(encoding="utf-8"))
    by_qid = {it["qid"]: it for it in ret["items"]}
    qs = load_questions(args.questions)
    # Reprise après une coupure : chaque réponse est ajoutée à un fichier .partial.jsonl au fil de l'eau.
    partial = Path(str(args.out) + ".partial.jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = json.loads(line)
                done[p["qid"]] = p
        print(f"reprise : {len(done)}/{len(qs)} réponses déjà faites")
    preds, t0, n_new = [], time.time(), 0
    gen = OfficialGenerator(backend=args.backend) if len(done) < len(qs) else None
    for n, q in enumerate(qs, 1):
        if q["qid"] in done:
            preds.append(done[q["qid"]])
            continue
        it = by_qid[q["qid"]]
        p = answer_one(gen, q, chunks, it["hits"], variant=args.variant, key_lines=it.get("key_lines"))
        p["retrieved"] = it["retrieved"]
        preds.append(p)
        n_new += 1
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
        if n % 10 == 0 or n == len(qs):
            print(f"  {n}/{len(qs)} ({(time.time() - t0) / max(n_new, 1):.1f} s/question) {q['qid']}: {p['answer'][:70]}",
                  flush=True)
    write_predictions(args.out, preds, run_name=f"rag2-{ret['strategy']}-{ret['mode']}-{args.variant}")
    partial.unlink(missing_ok=True)
    print(f"réponses brutes (avant abstention calibrée) -> {args.out}")


def cmd_answer(args):
    if Path(args.out).exists() and not Path(str(args.out) + ".partial.jsonl").exists():
        print(f"déjà fait -> {args.out}")
        return
    ret_path = str(Path(args.out).with_suffix("")) + "_retrieval.json"
    base = [sys.executable, "-m", "rag2.pipeline"]
    r = ["retrieve", "--index-dir", args.index_dir, "--questions", args.questions, "--out", ret_path]
    if args.strategy:
        r += ["--strategy", args.strategy]
    if args.mode:
        r += ["--mode", args.mode]
    subprocess.run(base + r, check=True)
    subprocess.run(base + ["generate", "--index-dir", args.index_dir, "--retrieval", ret_path,
                           "--questions", args.questions, "--out", args.out, "--backend", args.backend,
                           "--variant", args.variant], check=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--pdf-dir", required=True)
    a.add_argument("--index-dir", required=True)
    a.add_argument("--strategies", default=",".join(STRATEGIES))
    a.add_argument("--no-dense", action="store_true")
    a.add_argument("--no-ocr", action="store_true")
    a.add_argument("--no-ocr-images", action="store_true")
    a.add_argument("--reparse", action="store_true")
    for name in ("retrieve", "answer"):
        b = sub.add_parser(name)
        b.add_argument("--index-dir", required=True)
        b.add_argument("--questions", required=True)
        b.add_argument("--out", required=True)
        b.add_argument("--strategy", default=None)
        b.add_argument("--mode", default=None, choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
        if name == "answer":
            b.add_argument("--backend", default="hf", choices=["hf", "openai", "echo"])
            b.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    g = sub.add_parser("generate")
    g.add_argument("--index-dir", required=True)
    g.add_argument("--retrieval", required=True)
    g.add_argument("--questions", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--backend", default="hf", choices=["hf", "openai", "echo"])
    g.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(VARIANTS))
    args = ap.parse_args()
    {"index": cmd_index, "retrieve": cmd_retrieve, "generate": cmd_generate, "answer": cmd_answer}[args.cmd](args)


if __name__ == "__main__":
    main()
