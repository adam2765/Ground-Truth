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

    # Un découpage suffixé « +graph » (ex. naive_title+graph) utilise les pages enrichies par la description de
    # leurs graphiques (modèle de vision, en cache) ; sans suffixe, les pages telles que lues (texte + OCR).
    import copy

    from rag2.describe_charts import add_descriptions
    pages_graph = None

    # Tableaux réécrits avec leurs en-têtes de colonnes (rag2/tables.py), pour la génération (variante duoT)
    tab_path = idx / "tables.json.gz"
    if not tab_path.exists():
        import shutil

        from rag2.tables import extract_dir, save
        cache = Path(__file__).resolve().parents[1] / "cache" / "tables_v1.json.gz"
        import os
        if (cache.exists() and not os.environ.get("RAG_NO_CACHE")
                and Path(args.pdf_dir).resolve() == (cache.parents[1] / "corpus" / "pdf").resolve()):
            shutil.copy(cache, tab_path)
        else:
            save(extract_dir(args.pdf_dir), tab_path)

    encoder = None
    for strat in args.strategies.split(","):
        d = idx / strat
        d.mkdir(exist_ok=True)
        base, _, extra = strat.partition("+")
        if extra == "graph":
            if pages_graph is None:
                pages_graph = copy.deepcopy(pages)
                print(f"{add_descriptions(pages_graph)} pages enrichies par la description de leurs graphiques")
            chunks = make_chunks(pages_graph, base)
        else:
            chunks = make_chunks(pages, strat)
        (d / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        print(f"[{strat}] {len(chunks)} chunks")
        if not args.no_dense and not (d / "dense.npy").exists():
            from rag2.search import DenseEncoder
            encoder = encoder or DenseEncoder()
            t = time.time()
            np.save(d / "dense.npy", encoder.encode([c["text"] for c in chunks]))
            print(f"[{strat}] embeddings en {time.time() - t:.0f} s")

    # palier 3 : dates et graphe des éditions (rag2/docmeta.py)
    from rag2 import docmeta
    if not args.no_dense:
        from rag2.search import DenseEncoder
        encoder = encoder or DenseEncoder()
    docs = docmeta.build(pages, encoder if not args.no_dense else None)
    docmeta.save(docs, idx)
    print(f"{len(docs)} documents, {sum(len(v['newer']) for v in docs.values())} liens « édition plus récente »")
    print(f"indexation terminée en {(time.time() - t0) / 60:.1f} min")


def cmd_update(args):
    """Palier 3 : ajoute les PDF d'un nouveau dossier à un index existant, sans rien recalculer pour l'ancien
    corpus. Seuls les nouveaux documents sont lus (OCR compris), découpés et encodés ; les vecteurs sont ajoutés à
    la fin de dense.npy (les numéros des anciens chunks ne changent pas) ; BM25 est reconstruit au chargement
    (quelques secondes) ; le graphe des éditions est recalculé (titres seulement)."""
    from rag2 import docmeta
    from rag2.parse import read_pdf_dir
    from rag2.search import DenseEncoder
    from rag2.tables import extract_dir, load as load_tables, save as save_tables

    t0 = time.time()
    idx = Path(args.index_dir)
    pages = json.loads((idx / "pages.json").read_text(encoding="utf-8"))
    known = {p["doc_id"] for p in pages}
    new_pdfs = [p for p in sorted(Path(args.pdf_dir).glob("*.pdf")) if p.stem not in known]
    print(f"{len(new_pdfs)} nouveaux documents (sur {len(list(Path(args.pdf_dir).glob('*.pdf')))})")
    if not new_pdfs:
        return
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    for p in new_pdfs:                                  # on ne lit que les nouveaux fichiers
        shutil.copy(p, tmp / p.name)
    new_pages = read_pdf_dir(str(tmp), ocr=True, ocr_images=True)
    print(f"  {len(new_pages)} pages lues en {time.time() - t0:.0f} s")
    pages += new_pages
    (idx / "pages.json").write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    tabs = load_tables(idx / "tables.json.gz")
    tabs.update(extract_dir(str(tmp)))
    save_tables(tabs, idx / "tables.json.gz")
    encoder = None
    for d in sorted(p for p in idx.iterdir() if (p / "chunks.json").exists()):
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        add = make_chunks(new_pages, d.name.partition("+")[0])
        (d / "chunks.json").write_text(json.dumps(chunks + add, ensure_ascii=False), encoding="utf-8")
        if (d / "dense.npy").exists():
            encoder = encoder or DenseEncoder()
            old = np.load(d / "dense.npy")
            np.save(d / "dense.npy", np.concatenate([old, encoder.encode([c["text"] for c in add])]))
        print(f"  [{d.name}] +{len(add)} chunks")
    docs = docmeta.build(pages, encoder, docmeta.load(idx))
    docmeta.save(docs, idx)
    print(f"  {sum(len(v['newer']) for v in docs.values())} liens « édition plus récente »")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"mise à jour terminée en {(time.time() - t0) / 60:.1f} min")


def cmd_retrieve(args):
    from groundtruth.io import load_questions
    from rag2 import docmeta
    from rag2.search import Retriever, add_newer_editions, key_lines, ranked_pages, search_multi, table_rows
    from rag2.tables import load as load_tables

    cfg = defaults(args.index_dir)
    strategy, mode = args.strategy or cfg["strategy"], args.mode or cfg["mode"]
    qs = load_questions(args.questions)
    if Path(args.out).exists():                         # reprise après une coupure : déjà fait
        old = json.loads(Path(args.out).read_text(encoding="utf-8"))
        if (old.get("strategy") == strategy and old.get("mode") == mode and len(old["items"]) == len(qs)
                and old.get("split", False) == args.split and "table_rows" in old["items"][0]):
            print(f"recherche déjà faite -> {args.out}")
            return
    r = Retriever(args.index_dir, strategy, use_dense=mode != "bm25", use_rerank=mode == "hybrid_rerank")
    tables = load_tables(Path(args.index_dir) / "tables.json.gz")
    docs = docmeta.load(args.index_dir)
    items, t0 = [], time.time()
    for n, q in enumerate(qs, 1):
        hits = search_multi(r, q["question"], mode=mode, k=20) if args.split else r.search(q["question"], mode=mode, k=20)
        if docs and r.reranker is not None:
            hits = add_newer_editions(r, q["question"], hits, docs, drop_old=not args.keep_old,
                                      year_keep=getattr(args, "year_keep", False))
        item = {"qid": q["qid"], "hits": hits, "retrieved": ranked_pages(r.chunks, hits)}
        if r.reranker is not None:
            item["key_lines"] = key_lines(r.reranker, q["question"], r.chunks, hits)
            if tables:
                item["table_rows"] = table_rows(r.reranker, q["question"], r.chunks, hits, tables)
        items.append(item)
        if n % 20 == 0:
            print(f"  {n}/{len(qs)} ({(time.time() - t0) / n:.2f} s/question)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"strategy": strategy, "mode": mode, "split": args.split, "items": items}, ensure_ascii=False),
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
    import rag2.answer as answer_mod
    from rag2 import docmeta
    meta = docmeta.load(args.index_dir)
    answer_mod.DOC_DATES.update({d: v["date"] for d, v in meta.items() if v.get("date")})
    answer_mod.DOC_EDITIONS.update({d: set(v.get("newer", [])) | set(v.get("older", [])) for d, v in meta.items()})
    tables = None
    if VARIANTS[args.variant].get("tables"):
        from rag2.tables import load
        tables = load(Path(args.index_dir) / "tables.json.gz")
        print(f"tableaux structurés : {len(tables)} pages")
    charts = None
    if VARIANTS[args.variant].get("charts"):
        from rag2.charts_vlm import load as load_charts
        charts = (load_charts(Path(args.index_dir) / "charts_a.json.gz"), load_charts(Path(args.index_dir) / "charts_b.json.gz"))
        print(f"graphiques décrits : {len(charts[0])} pages (lecture A), {len(charts[1])} (lecture B)")
    pages = None
    if VARIANTS[args.variant].get("second_pages"):
        pages = {f"{p['doc_id']}|{p['page']}": p["text"]
                 for p in json.loads((Path(args.index_dir) / "pages.json").read_text(encoding="utf-8"))}
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
        p = answer_one(gen, q, chunks, it["hits"], variant=args.variant, key_lines=it.get("key_lines"),
                       tables=tables, table_rows=it.get("table_rows"), pages=pages,
                       charts=charts)
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
    if args.split:
        r += ["--split"]
    if getattr(args, "keep_old", False):
        r += ["--keep-old"]
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
    u = sub.add_parser("update")
    u.add_argument("--index-dir", required=True)
    u.add_argument("--pdf-dir", required=True)
    for name in ("retrieve", "answer"):
        b = sub.add_parser(name)
        b.add_argument("--index-dir", required=True)
        b.add_argument("--questions", required=True)
        b.add_argument("--out", required=True)
        b.add_argument("--strategy", default=None)
        b.add_argument("--mode", default=None, choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
        b.add_argument("--split", action="store_true", help="questions en plusieurs parties : une recherche par partie")
        b.add_argument("--keep-old", action="store_true", help="garder les extraits des éditions remplacées")
        b.add_argument("--year-keep", action="store_true",
                       help="garder les éditions remplacées si la question est datée sans « dernier / plus récent »")
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
    {"index": cmd_index, "update": cmd_update, "retrieve": cmd_retrieve, "generate": cmd_generate,
     "answer": cmd_answer}[args.cmd](args)


if __name__ == "__main__":
    main()
