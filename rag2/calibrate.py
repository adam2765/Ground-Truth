"""Abstention calibrée.

Une réponse fausse coûte -1, « INTROUVABLE » coûte 0 (ou rapporte +1 si la question n'a pas de réponse).
On répond donc seulement quand on a de bonnes raisons de croire la réponse juste.

Règle : on remplace la réponse par INTROUVABLE si
    le modèle s'est abstenu, OU score du reranker du meilleur chunk < seuil,
    OU (option) la réponse contient des chiffres absents des extraits,
    OU (option) l'auto-vérification du modèle a répondu NON.

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
    if rule.get("require_verified") and meta.get("verified") is False:
        return True
    if rule.get("require_quote") and meta.get("quote_ok") is False:
        return True
    ag = rule.get("agreement", "off")                  # concordance de deux réponses indépendantes
    key = "agree_inclus" if ag.endswith("_inclus") else "agree"
    if ag != "off" and key in meta and (meta[key] is False or (ag.startswith("strict") and meta[key] is None)):
        return True
    u = rule.get("max_unsupported")
    # run 9 : une réponse confirmée par la relecture des rangées du tableau (3e appel) peut contenir un
    # calcul (écart de deux valeurs) absent du texte : ce n'est pas une invention, le filtre ne s'applique pas
    if rule.get("rows_override_unsupported") and meta.get("row_check") is not None and meta.get("agree_inclus") is True:
        return False
    return u is not None and meta.get("unsupported", 0) > u


def enrich(preds: dict) -> None:
    """Ajoute la concordance « inclusion » aux réponses qui ont une 2e réponse (variantes duo / trio)."""
    from rag2.answer import agree_inclus
    for p in preds.values():
        m = p.get("_meta", {})
        if "second" in m and "agree_inclus" not in m and not m.get("llm_abstain"):
            m["agree_inclus"] = agree_inclus(p["answer"], m["second"])


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
    enrich(preds)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = {r["qid"]: r["points"] for r in report["details"]}
    answerable = {q["qid"]: q["answerable"] for q in load_questions(args.gold)}
    metas = {qid: preds[qid].get("_meta", {}) for qid in rows}

    base = simulate(rows, metas, answerable, {"threshold": None, "max_unsupported": None})
    scores = sorted({m["top_score"] for m in metas.values() if m.get("top_score") is not None})
    thresholds = [None] + scores
    results = []
    has_agree = any(m.get("agree") is not None for m in metas.values())
    for ag in (("off", "souple", "strict", "souple_inclus", "strict_inclus") if has_agree else ("off",)):
        for rq in (False, True):
            for rv in (False, True):
                for u in UNSUP_OPTIONS:
                    for t in thresholds:
                        rule = {"threshold": t, "max_unsupported": u, "require_verified": rv, "require_quote": rq,
                                "agreement": ag}
                        results.append((simulate(rows, metas, answerable, rule), rule))
    best_score = max(s for s, _ in results)
    # parmi les règles quasi optimales, on prend le seuil médian : moins sensible au hasard des 100 questions
    near = [r for s, r in results if s >= best_score - 0.005]
    opt = lambda r: (r["max_unsupported"], r["require_verified"], r["require_quote"], r["agreement"])
    best_opt = max({opt(r) for r in near}, key=lambda o: sum(opt(r) == o for r in near))
    cands = [r for r in near if opt(r) == best_opt]
    ts = sorted(r["threshold"] for r in cands if r["threshold"] is not None)
    rule = dict(cands[0]) if not ts else {"threshold": ts[len(ts) // 2], "max_unsupported": best_opt[0],
                                          "require_verified": best_opt[1], "require_quote": best_opt[2],
                                          "agreement": best_opt[3]}
    final = simulate(rows, metas, answerable, rule)
    rule["validation_score_brut"] = round(base, 4)
    rule["validation_score_calibre"] = round(final, 4)
    rule["taux_abstention"] = round(float(np.mean([abstains(metas[q], rule) for q in rows])), 3)
    print(f"score sans calibration : {base:+.3f}")
    print(f"score avec calibration : {final:+.3f}   (seuil reranker {rule['threshold']}, "
          f"chiffres non retrouvés tolérés : {rule['max_unsupported']}, auto-vérification exigée : "
          f"{rule['require_verified']}, citation vérifiée exigée : {rule['require_quote']}, "
          f"concordance : {rule.get('agreement')}, abstention {rule['taux_abstention']:.0%})")
    print(f"  citations recopiées : {sum(m.get('quote_ok') is not None for m in metas.values())}, "
          f"dont introuvables dans les extraits : {sum(m.get('quote_ok') is False for m in metas.values())}")
    n_ver = sum(m.get("verified") is not None for m in metas.values())
    print(f"  auto-vérification : {n_ver} verdicts, dont NON : {sum(m.get('verified') is False for m in metas.values())}")
    for ag in sorted({r["agreement"] for _, r in results}):
        for u in UNSUP_OPTIONS:
            sub = [(s, r) for s, r in results if r["max_unsupported"] == u and r["agreement"] == ag]
            s, r = max(sub, key=lambda x: x[0])
            print(f"  concordance={ag}, chiffres={u}: meilleur {s:+.3f} (seuil {r['threshold']}, "
                  f"vérif={r['require_verified']}, citation={r['require_quote']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rule, indent=1))


def cmd_apply(args):
    rule = json.loads(Path(args.rule).read_text())
    preds = load_predictions(args.raw)
    enrich(preds)
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
