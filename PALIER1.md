# Palier 1 : RAG from scratch

Code : `my_rag/` (sans LangChain, LlamaIndex, FAISS, `bm25s` ni base vectorielle). Lancement sur GPU T4 :
`scripts/run_palier1.sh`. Journaux et fichiers : `results/palier1/`.

| Brique | Fichier | Ce qu'elle fait |
|---|---|---|
| Lecture | `extract.py` | texte de chaque page avec PyMuPDF (pas d'OCR) |
| Découpage | `chunk.py` | fenêtres de 200 mots, recouvrement de 50, sans jamais mélanger deux pages |
| BM25 | `bm25.py` | tokenisation, index inversé, score Okapi (k1 = 1,5, b = 0,75) |
| Dense | `dense.py` | embeddings `intfloat/multilingual-e5-small`, cosinus et top-k en NumPy |
| Métriques | `metrics.py` | Recall@k, MRR, nDCG@10 au niveau page |
| RAG | `rag.py` | 8 meilleurs chunks BM25 dans le prompt, 1 appel au générateur officiel, citations `[doc_id p.N]` |

## Vérifications

- **BM25 contre `bm25s`** (`check_bm25.py`) : même top 10 sur les 77 questions avec réponse ; écart de score
  maximal 4e-6 (après division par le facteur constant k1 + 1, que `bm25s` en variante Lucene n'applique pas).
- **Métriques contre l'évaluateur officiel** : écart 0 sur toutes les questions.

## Résultats sur validation_v1

Recherche, 77 questions avec réponse (8 443 chunks) :

| Recherche | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 maison | 0,558 | 0,662 | 0,409 | 0,411 |
| Dense maison (e5-small) | 0,565 | 0,662 | 0,412 | 0,422 |

Les deux méthodes font jeu égal. Par type (Recall@5, BM25 / dense) : texte 0,69 / 0,69, tableau 0,65 / 0,71,
graphique 0,67 / 0,67, synthèse 0,50 / 0,44, **multi-doc 0,20 / 0,20, scan 0,11 / 0,11**.

RAG de base (BM25 maison + générateur officiel), 100 questions, juge officiel :

| Score de confiance | Justes et citées | Fausses | Hallucinations (pièges) | Abstentions justes | Abstentions |
|---|---|---|---|---|---|
| **+0,02** | 21 | 36 | 3 | 18 | 20 |

Par type : piège +0,74, tableau 0, texte 0, multi-doc −0,20, scan −0,11, graphique −0,58, synthèse −0,75.
Le système fait presque autant d'erreurs que de bonnes réponses : à −1 l'erreur, il ne vaut guère mieux que
répondre INTROUVABLE à tout (+0,23).

## Cinq échecs commentés

**1. val1-015 (scan) : la page de la réponse est une image.**
« En 2021, combien de m² de terrain artificialisé par habitant dans les campagnes à habitat très éparpillé, et
dans les grandes villes ? » La réponse (2 830 m² et 204 m²) est sur la page 2 de `insee-ip2103`, un document
numérisé : PyMuPDF n'en extrait aucun texte, la page n'est jamais indexée et ne peut pas être retrouvée
(Recall@5 des questions sur scan : 0,11). Le modèle a en plus écrit « INTOUVABLE » (faute de frappe), que le
code n'a pas reconnu comme une abstention : la réponse est comptée fausse (−1) au lieu de 0.
*Palier 2 : OCR Tesseract des pages sans texte ; détection tolérante de INTROUVABLE.*

**2. val1-097 (tableau) : la question n'emploie aucun mot du document.**
« Les crédits des Sages de la rue de Montpensier ont-ils beaucoup augmenté entre 2020 et 2025 ? » Le document
parle du « Conseil constitutionnel » ; BM25 ne peut pas faire le lien, et la page de preuve
(`cc-presidence-republique-2025` p.19) n'est pas dans les 10 premiers résultats. Le modèle répond avec un
extrait sur les collectivités locales (+16,5 %) au lieu de +43,4 %.
*Palier 2 : recherche dense multilingue (bge-m3) en plus de BM25, puis reranker.*

**3. val1-012 (tableau) : la bonne page est trouvée, mais le tableau est illisible.**
La page de preuve (`insee-ip2082` p.1) est classée première. La rangée utile est extraite comme une suite
de nombres, une cellule par ligne : « Rural à habitat très dispersé / 1 992 / 1 655 411 / 12 / 10 / 2,9 / 101 ».
Le modèle prend la mauvaise colonne : « +10 %, de 1 655 à 1 812 » au lieu de +101 %.
*Palier 2 : cellules séparées par « | » et une rangée par ligne dans le texte extrait.*

**4. val1-006 (graphique) : la réponse est dessinée, pas écrite.**
« Entre 2018 et 2025, quelle année le prix spot a-t-il le moins souvent dépassé le barème ? » La page de
preuve est classée première, mais la réponse (2023, environ 800 heures) n'est lisible que sur la hauteur des
barres du graphique n° 7 : le texte extrait ne contient que le titre et les axes. Le modèle invente
(« 2024, 0 heure »).
*Palier 2 : l'abstention écarte ces réponses ; la description des graphiques par un modèle de vision a été
testée (run 4) et a fait baisser le score, faute de lectures fiables.*

**5. val1-031 (piège) : le modèle répond à une question sans réponse.**
« En 2021, quel pourcentage des salariés travaillaient à distance au moins un jour par semaine ? » Le corpus
ne contient pas ce chiffre pour 2021. Le modèle répond quand même « 18,9 % », en indiquant l'avoir « calculé à
partir des données de 2024 » : il préfère extrapoler plutôt que dire INTROUVABLE. Une seule erreur de ce type
coûte autant qu'une bonne réponse rapporte.
*Palier 2 : abstention calibrée (deux réponses indépendantes qui doivent concorder, chiffres absents des
extraits, réponses partielles).*

## Du palier 1 au palier 2

| | Palier 1 (ce fichier) | Palier 2 (`PALIER2.md`) |
|---|---|---|
| Recall@5 | 0,558 (BM25) / 0,565 (dense) | 0,948 (OCR, racinisation, hybride, reranker) |
| Score de confiance (validation) | +0,02 | +0,465 |
