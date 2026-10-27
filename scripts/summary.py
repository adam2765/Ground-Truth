"""Résumé des résultats du palier 2, à envoyer à Claude."""
import glob
import json
from collections import Counter

abl = json.load(open("runs/ablation_retrieval.json"))
print("RECHERCHE (validation, questions avec réponse)")
for r in abl:
    print(f"  {r['strategy']:13s} {r['mode']:14s} R@5 {r['recall@5']:.3f}  R@10 {r['recall@10']:.3f}  MRR {r['mrr']:.3f}")
print("  meilleure :", json.load(open("index2/best.json")))

print("\nVARIANTES DE GÉNÉRATION (validation, juge officiel)")
for f in sorted(glob.glob("index2/abstention_*.json")):
    v = f.split("abstention_")[1][:-5]
    a = json.load(open(f)); r = json.load(open(f"runs/val_report_{v}.json"))
    print(f"  {v:8s} brut {r['score_confiance']:+.3f}  calibré {a['validation_score_calibre']:+.3f}  "
          f"abstention {a['taux_abstention']:.0%}  issues {r['issues']}")

rep = json.load(open("runs/val_report.json"))
print("\nVALIDATION, réponses brutes (juge officiel) :", rep["score_confiance"], rep["issues"])
print("  par type :", {t: v["score"] for t, v in rep["par_type"].items()})
print("  retrieval :", rep.get("retrieval"))
print("ABSTENTION CALIBRÉE :", json.load(open("index2/abstention.json")))
fails = [d for d in rep["details"] if d["points"] < 0]
print("  réponses fausses avant calibration, par type :", dict(Counter(d["type"] for d in fails)))

cls = json.load(open("runs/classement_v1.json"))["predictions"]
n_abs = sum(p["answer"] == "INTROUVABLE" for p in cls)
print(f"\nLEADERBOARD v1 : {len(cls)} réponses, {n_abs} INTROUVABLE ({n_abs / len(cls):.0%})")
subs = sorted(glob.glob("submissions/v1/*"))
print("SOUMISSION :", subs[-1] if subs else "aucune")
