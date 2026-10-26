"""Abstention calibrée.

Une réponse fausse coûte -1, « INTROUVABLE » coûte 0 (ou rapporte +1 si la question n'a pas de réponse).
On répond donc seulement quand on a de bonnes raisons de croire la réponse juste.

Règle : on remplace la réponse par INTROUVABLE si
    le modèle s'est abstenu, OU score du reranker du meilleur chunk < seuil,
    OU (option) la réponse contient des chiffres absents des extraits.

fit   : cherche le seuil et l'option qui maximisent le score de confiance sur validation, à partir des
        réponses brutes (_meta) et du rapport détaillé de l'évaluateur officiel (--details / --out).
apply : applique la règle à un fichier de réponses brutes et écrit le fichier au format officiel.

    python -m groundtruth.evaluate --pred runs/val_raw.json --gold data/validation_v1.json --judge hf --out runs/val_report.json
    python -m rag2.calibrate fit   --raw runs/val_raw.json --report runs/val_report.json --gold data/validation_v1.json --out index2/abstention.json
    python -m rag2.calibrate apply --raw runs/cls_raw.json --rule index2/abstention.json --out runs/classement_v1.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from groundtruth.io import load_predictions, load_questions, write_predictions

UNSUP_OPTIONS = [None, 0, 1]          # None = ignoré ; 0 = abstention dès 1 chiffre non retrouvé ; 1 = dès 2


def abstains(meta: dict, rule: dict) -> bool:
    if meta.get("llm_abstain"):
        return True
    ts = meta.get("top_score")
    if rule.get("threshold") is not None and ts is not None and ts < rule["threshold"]:
        return True
    u = rule.get("max_unsupported")
    return u is not None and meta.get("unsupported", 0) > u


def simulate(rows, metas, answerable, rule) -> float:
    pts = []
    for qid, points in rows.items():
        if abstains(metas[qid], rule):
            pts.append(1.0 if not answerable[qid] else 0.0)
        else:
            pts.append(points)
    return float(np.mean(pts))


def cmd_fit(args):
    preds = load_predictions(args.raw)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = {r["qid"]: r["points"] for r in report["details"]}
    answerable = {q["qid"]: q["answerable"] for q in load_questions(args.gold)}
    metas = {qid: preds[qid].get("_meta", {}) for qid in rows}

    base = simulate(rows, metas, answerable, {"threshold": None, "max_unsupported": None})
    scores = sorted({m["top_score"] for m in metas.values() if m.get("top_score") is not None})
    thresholds = [None] + scores
    results = []
    for u in UNSUP_OPTIONS:
        for t in thresholds:
            rule = {"threshold": t, "max_unsupported": u}
            results.append((simulate(rows, metas, answerable, rule), rule))
    best_score = max(s for s, _ in results)
    # parmi les règles quasi optimales, on prend le seuil médian : moins sensible au hasard des 100 questions
    near = [r for s, r in results if s >= best_score - 0.005]
    u_best = max(set(str(r["max_unsupported"]) for r in near),
                 key=lambda u: sum(str(r["max_unsupported"]) == u for r in near))
    cands = [r for r in near if str(r["max_unsupported"]) == u_best]
    ts = sorted(r["threshold"] for r in cands if r["threshold"] is not None)
    rule = cands[0] if not ts else {"threshold": ts[len(ts) // 2], "max_unsupported": cands[0]["max_unsupported"]}
    final = simulate(rows, metas, answerable, rule)
    rule["validation_score_brut"] = round(base, 4)
    rule["validation_score_calibre"] = round(final, 4)
    rule["taux_abstention"] = round(float(np.mean([abstains(metas[q], rule) for q in rows])), 3)
    print(f"score sans calibration : {base:+.3f}")
    print(f"score avec calibration : {final:+.3f}   (seuil reranker {rule['threshold']}, "
          f"chiffres non retrouvés tolérés : {rule['max_unsupported']}, abstention {rule['taux_abstention']:.0%})")
    for u in UNSUP_OPTIONS:
        sub = [(s, r) for s, r in results if r["max_unsupported"] == u]
        s, r = max(sub, key=lambda x: x[0])
        print(f"  option chiffres={u}: meilleur {s:+.3f} (seuil {r['threshold']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rule, indent=1))


def cmd_apply(args):
    rule = json.loads(Path(args.rule).read_text())
    preds = load_predictions(args.raw)
    out, n_abs = [], 0
    for qid, p in preds.items():
        meta = p.get("_meta", {})
        if abstains(meta, rule):
            n_abs += 1
            out.append({"qid": qid, "answer": "INTROUVABLE", "citations": [], "retrieved": p.get("retrieved", [])})
        else:
            out.append({"qid": qid, "answer": p["answer"], "citations": p["citations"],
                        "retrieved": p.get("retrieved", [])})
    write_predictions(args.out, out, run_name="rag2-calibre")
    print(f"{len(out)} réponses, {n_abs} INTROUVABLE ({n_abs / len(out):.0%}) -> {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--raw", required=True)
    f.add_argument("--report", required=True)
    f.add_argument("--gold", required=True)
    f.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--raw", required=True)
    a.add_argument("--rule", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"fit": cmd_fit, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
