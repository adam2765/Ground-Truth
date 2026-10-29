"""Run 10 : 3e appel ciblé ajouté après coup à des réponses déjà générées (et déjà jugées).

Les deux premières réponses ne changent pas (décodage glouton, déterministe) : seul le verdict de concordance
change. Le score de validation se recalcule donc sans nouveau passage du juge.

    python -m rag2.recheck focus --index-dir index2 --raw R.json --questions Q.json --out F.json   # reranker
    python -m rag2.recheck ask   --raw R.json --focus F.json --questions Q.json --out R2.json      # générateur

Seules les questions où les deux réponses divergent (et où le 3e appel du run 9 n'a rien confirmé) sont
reprises. Le 3e appel reçoit les 6 lignes les plus proches de la question, phrases et rangées de tableau
confondues (rag2.search.focus_lines), au lieu des seules rangées de tableau du run 9.
"""
import argparse
import json
from pathlib import Path

from groundtruth.io import load_predictions, load_questions, write_predictions
from rag2.calibrate import enrich


def todo(preds: dict) -> list[str]:
    enrich(preds)
    return [q for q, p in preds.items() if not p["_meta"].get("llm_abstain")
            and p["_meta"].get("agree_inclus") is not True]


def cmd_focus(a):
    from rag2.pipeline import defaults
    from rag2.search import Retriever, focus_lines, search_multi
    from rag2.tables import load

    preds = load_predictions(a.raw)
    qs = {q["qid"]: q for q in load_questions(a.questions)}
    cfg = defaults(a.index_dir)
    r = Retriever(a.index_dir, cfg["strategy"])
    tables = load(Path(a.index_dir) / "tables.json.gz")
    out = {}
    for qid in todo(preds):
        hits = search_multi(r, qs[qid]["question"], k=20)       # même recherche que le run 9
        out[qid] = focus_lines(r.reranker, qs[qid]["question"], r.chunks, hits, tables)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} questions à revérifier -> {a.out}")


def cmd_ask(a):
    from groundtruth.generator import OfficialGenerator
    from rag2.answer import focus_check

    preds = load_predictions(a.raw)
    qs = {q["qid"]: q for q in load_questions(a.questions)}
    focus = json.loads(Path(a.focus).read_text(encoding="utf-8"))
    gen = OfficialGenerator(backend=a.backend)
    n_ok = 0
    for qid, lines in focus.items():
        if not lines:
            continue
        gen.start_question(qid + "#3")                 # compteur à part : en production c'est le 3e appel sur 4
        p = preds[qid]
        third, ok = focus_check(gen, qs[qid]["question"], p["answer"], lines)
        p["_meta"]["row_check"] = third
        p["_meta"]["agree_inclus"] = ok
        n_ok += ok
    write_predictions(a.out, list(preds.values()), run_name="rag2-recheck")
    print(f"{len(focus)} revérifiées, {n_ok} confirmées -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("focus")
    f.add_argument("--index-dir", required=True)
    for p in (f, sub.add_parser("ask")):
        p.add_argument("--raw", required=True)
        p.add_argument("--questions", required=True)
        p.add_argument("--out", required=True)
    sub.choices["ask"].add_argument("--focus", required=True)
    sub.choices["ask"].add_argument("--backend", default="hf")
    a = ap.parse_args()
    {"focus": cmd_focus, "ask": cmd_ask}[a.cmd](a)


if __name__ == "__main__":
    main()
