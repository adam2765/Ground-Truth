"""Vérificateur indépendant (NLI) : la réponse est-elle impliquée par les extraits ?

Idée tirée de la littérature récente (arXiv 2608.22228 : sur de petits modèles, l'abstention par le prompt échoue
face à un extrait trompeur ; il faut des signaux extérieurs au générateur). Un petit modèle d'inférence
(mDeBERTa-v3 multilingue, entraîné sur XNLI) lit chaque extrait du contexte (prémisse) et la réponse (hypothèse)
et donne P(implication). On garde le maximum sur les extraits. C'est un 3e signal, indépendant du générateur,
que la calibration peut combiner avec la concordance (rag2/calibrate.py, option min_entail).

    python -m rag2.nli --index-dir index2 --raw R.json --questions Q.json --out runs/nli_X.json
"""
import argparse
import json
from pathlib import Path

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from groundtruth.io import load_predictions, load_questions
    from rag2.answer import MAX_CHUNKS, diversify
    from rag2.pipeline import defaults
    from rag2.search import Retriever, search_multi
    from rag2.tables import load, render

    preds = load_predictions(a.raw)
    qs = {q["qid"]: q for q in load_questions(a.questions)}
    r = Retriever(a.index_dir, defaults(a.index_dir)["strategy"])
    tables = load(Path(a.index_dir) / "tables.json.gz")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(dev).eval()
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    i_ent, i_con = lab["entailment"], lab["contradiction"]

    out = {}
    for qid, p in preds.items():
        if p["answer"] == "INTROUVABLE":
            continue
        q = qs[qid]["question"]
        hits = search_multi(r, q, k=20)
        cand = [r.chunks[h["idx"]] for h in hits]
        prem = [c["body"] for c in diversify(cand, [h.get("score") for h in hits], MAX_CHUNKS)]
        for c in cand[:5]:
            for t in tables.get(f"{c['doc_id']}|{c['page']}", []):
                prem.append(render(t, q))
        res = p["_meta"].get("rescue")
        if res:
            prem.append(res.get("rows_answer", ""))
        hyp = p["answer"]
        enc = tok(prem, [hyp] * len(prem), truncation="only_first", max_length=512, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            prob = model(**enc).logits.float().softmax(-1).cpu()
        k = int(prob[:, i_ent].argmax())
        out[qid] = {"entail": float(prob[k, i_ent]), "contra": float(prob[:, i_con].max()), "n": len(prem)}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} réponses vérifiées -> {a.out}")


if __name__ == "__main__":
    main()
