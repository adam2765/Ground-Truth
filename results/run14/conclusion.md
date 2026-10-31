# Run 14 : graphiques découpés décrits par Qwen3-VL-2B (lecture A) et -4B (lecture B)

- Description des 874 graphiques du corpus : 2B en 31,5 min (2,2 s/graphique, lots de 8), 4B en 75 min
  (5,2 s/graphique, lots de 4). Les deux à la suite sur un seul T4 : ~107 min, trop pour les 2 h d'indexation
  (OCR, tableaux et embeddings en plus). Le 2B seul tiendrait.
- Validation (duoXRSG, juge officiel) : score brut +0,31 (meilleur à ce jour, contre +0,255), graphiques -0,08
  (contre -0,375), mais après abstention +0,50 contre +0,525 pour le run 12 : non soumis.
- Graphiques gagnés : val1-010 (2,6 %), val1-041 (erreur évitée). Perdus : val1-060 (réponse juste, 28 %, mais
  jugée fausse par le juge officiel), et des questions sans graphique perturbées par les descriptions ajoutées
  au contexte (val1-005, 032, 076), comme au run 4.
- Utiliser les descriptions seulement en repli (quand le run 12 s'abstient) : +0,505. Non retenu.
