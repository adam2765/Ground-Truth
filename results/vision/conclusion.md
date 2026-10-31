# Sonde vision (30/10) : 7 premières questions « graphique » de la validation

| Question | Attendu | Qwen2.5-VL-3B (run 4) | Qwen2.5-VL-7B (4 bits) | Qwen3-VL-8B (4 bits) |
|---|---|---|---|---|
| val1-006 | année la plus basse : 2023 | non | non (2023 = 2024) | non (2023 = 2024) |
| val1-010 | 2,6 % | non | oui | oui |
| val1-022 | 133 M€, 19 % | oui | oui | oui |
| val1-027 | plus faible : agriculture | non | oui | oui |
| val1-041 | plus fort : pôles < 50 000 hab. | non | non | non (ordre inversé) |
| val1-042 | 322 Md€ | non | non | oui |
| val1-045 | 1er contributeur : hébergement-restauration | non | non | non |
| **Total** | | **1/7** | **3/7** | **4/7** |
| Temps par page (T4, 4 bits) | | ~3 s | 29 s | 35 à 55 s |

Arrêt de la sonde à la page insee-ip2112 p.2 (document absent du corpus v1).

Conclusion : les modèles plus gros lisent mieux (4/7 contre 1/7), mais se trompent encore une fois sur deux, avec
des valeurs inventées sûres d'elles (« Total : 149 Md€ »), et 630 pages à 30-50 s dépassent de loin les 2 h
d'indexation autorisées sur un T4. Piste abandonnée.

## Sonde n° 2 (31/10) : IBM Granite 4.0 3B Vision (ChartNet, balise <chart2csv>)

15 s/page (float16, T4). Sur la page entière (texte + graphique), le modèle produit souvent un CSV vide, recopie
du texte courant, ou décale les étiquettes d'une ligne (val1-060 : « 6 % Atteintes humaines » au lieu de 28 %).
Valeurs attendues retrouvées : 1 question sur 7 numériques (val1-055). Il est conçu pour un graphique recadré, en
anglais : sur nos pages françaises entières, il fait moins bien que Qwen3-VL-8B (4/7). Piste vision close.
