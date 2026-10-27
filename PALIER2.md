# Palier 2 : le pipeline

Code : `rag2/` (le palier 1 from scratch reste dans `my_rag/`). Notebook : `notebooks/palier2_colab.ipynb`.

```
PDF ─► parse.py ─► chunking.py ─► search.py ───────────────────────► answer.py ─► calibrate.py ─► réponse
      texte +      4 stratégies    BM25 (racinisé) ┐                  générateur     INTROUVABLE si
      OCR scans    de découpage    bge-m3 (dense)  ┼─ RRF ─► reranker  officiel,      reranker < seuil
      et graphiques                                ┘   (bge-reranker)  1 appel        ou modèle abstenu
```

## 1. Parsing (lecture des PDF)

Mesure : part des valeurs clés des questions de validation présentes dans le texte extrait de la page de
preuve (une valeur absente ne peut pas être retrouvée). `python -m rag2.ablation_parsing`

| Approche | tout | scan | tableau | graphique | texte | temps (77 questions) |
|---|---|---|---|---|---|---|
| raw : texte PyMuPDF (palier 1) | 0,84 | 0,22 | 0,88 | 0,83 | 0,96 | 2 s |
| tables : raw + `find_tables()` | 0,84 | 0,22 | 0,88 | 0,83 | 0,96 | 73 s |
| md : pymupdf4llm (Markdown + OCR) | 0,94 | 1,00 | 0,88 | 0,83 | 0,96 | 189 s |
| **ocr : raw + OCR Tesseract (retenue)** | 0,93 | 0,94 | 0,88 | 0,83 | 0,96 | 283 s* |

\* OCR à 300 dpi. Sur le corpus entier : 97 pages sans texte + 479 images de graphiques, ~10 min sur 2 cœurs.

Choix : texte + OCR. Même gain que pymupdf4llm sur les scans, pour un coût bien plus faible sur les
2 600 pages (pymupdf4llm : ~1,4 s par page). L'extraction de tableaux n'apporte rien : les valeurs des
tableaux sont déjà dans le texte. Les valeurs encore manquantes sont surtout des **calculs** (un écart
entre deux chiffres, « 32 milliers » à écrire 32 000) ou des valeurs dessinées dans un graphique : c'est
le travail du LLM, ou d'une description des graphiques par un modèle de vision (piste suivante).

### Annotation de 30 pages difficiles

`annotation/annoter.html` (pages + les deux textes côte à côte), notes dans `annotation/annotations.csv`.
Annotation réalisée par Claude (assistant IA) en regardant chaque page rendue, à vérifier par Adam.
A = texte brut PyMuPDF, B = texte + OCR (retenue). Notes de 0 (mauvais) à 2 (bon), moyennes :

| Catégorie | n | A complet | A chiffres | A ordre | B complet | B chiffres | B ordre |
|---|---|---|---|---|---|---|---|
| scan | 8 | 0,2 | 0,2 | 0,2 | 2,0 | 1,1 | 2,0 |
| graphique vectoriel | 8 | 1,9 | 1,0 | 2,0 | 2,0 | 1,0 | 2,0 |
| tableau | 8 | 2,0 | 2,0 | 1,0 | 2,0 | 2,0 | 1,0 |
| graphique en image | 6 | 0,8 | 0,5 | 1,5 | 1,8 | 1,0 | 1,7 |
| **tout** | 30 | 1,3 | 1,0 | 1,2 | **2,0** | **1,3** | **1,7** |

Ce que l'annotation a appris :
- **L'OCR règle les scans** (texte complet, quelques erreurs de caractères) et récupère le texte des
  images insérées (une lettre scannée en annexe, des étiquettes de graphiques).
- **Elle a révélé un bug de B** : la première version lisait les blocs avec `sort=True`, qui trie par
  position verticale ; sur les pages Insee en 3 colonnes, les paragraphes des colonnes étaient mélangés.
  Corrigé (ordre natif du PDF), voir `rag2/parse.py`.
- **Les valeurs des graphiques restent le point faible** (chiffres : 1,0) : barres, courbes et camemberts
  ne portent souvent pas leurs valeurs en texte, et l'OCR n'en récupère qu'une partie (1 pourcentage
  sur 6 sur un camembert). Piste : décrire les graphiques à l'indexation avec un modèle de vision.
- **Tableaux** : toutes les valeurs sont là, mais la structure est fragile (A : une cellule par ligne ;
  B : une ligne par rangée, frontières de cellules perdues). Un tableau entièrement en image (page 133
  du rapport PAC) reste illisible.

## 2. Recherche (BM25 seul, sans GPU)

Recall@5 / MRR sur les 77 questions de validation qui ont une réponse.

| Configuration | Recall@5 | MRR |
|---|---|---|
| Palier 1 : BM25 maison, découpage naïf, sans OCR | 0,558 | 0,408 |
| + racinisation française et mots vides | 0,662 | 0,499 |
| + OCR (sans racinisation) | 0,643 | 0,485 |
| + OCR + racinisation | 0,786 | 0,601 |

Stratégies de découpage (BM25 racinisé, avec OCR) :

| Chunking | Chunks | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| naive (200 mots, recouvrement 50) | 8 772 | 0,779 | 0,831 | 0,583 |
| naive_title (+ titre du document) | 8 772 | 0,779 | 0,838 | 0,597 |
| page (une page = un chunk) | 2 625 | 0,792 | 0,851 | 0,583 |
| struct_title (paragraphes ~180 mots + titre) | 6 961 | 0,779 | 0,838 | 0,604 |

(Après correction de l'ordre de lecture. Avant correction, struct_title faisait 0,805 : l'écart
correspond à 2 questions sur 77, du bruit à cette taille d'échantillon. Les stratégies sont proches en
BM25 seul ; la comparaison qui compte est celle avec la recherche dense et le reranker.)

Point faible restant : les questions multi-doc (Recall@5 ≈ 0,40).

## 3. Recherche hybride et reranker (run 1, Kaggle T4)

Recall@5 sur les 77 questions de validation avec réponse (`results/run1/ablation_retrieval.json`) :

| Chunking | BM25 | Dense (bge-m3) | Hybride (RRF) | Hybride + reranker |
|---|---|---|---|---|
| naive | 0,779 | 0,818 | 0,851 | 0,942 |
| naive_title | 0,779 | 0,851 | 0,844 | **0,948** (MRR 0,813) |
| page | 0,792 | 0,792 | 0,838 | 0,890 |
| struct_title | 0,779 | 0,877 | 0,877 | 0,916 (MRR 0,833) |

- Le reranker est la brique qui rapporte le plus (+7 à +10 points de Recall@5 sur l'hybride).
- La fusion hybride aide surtout le découpage naïf ; avec struct_title, le dense seul fait aussi bien.
- Une page entière par chunk est la moins bonne option : trop de sujets mélangés dans un même vecteur.
- Retenu : naive_title + hybride + reranker. Recall@5 par type : graphique 1,0, multi-doc 0,8,
  scan 0,89, synthèse 1,0, tableau 0,88, texte 1,0. Recherche : 0,8 s par question.

## 4. Génération et abstention

### Run 1 (soumission 2026-10-27_1)

Validation, juge officiel (Qwen3-4B) : **+0,17** brut, **+0,28** après calibration (abstention 54 %).
41 réponses justes et citées, mais **35 réponses fausses** alors que la bonne page était presque toujours
dans le contexte : le problème est la génération, pas la recherche. Analyse des 35 erreurs :
- ~12 : le modèle applique la consigne « corrige la prémisse » à des questions normales (« Correction de
  l'affirmation initiale : ... ») puis invente ;
- ~10 : mauvais chiffre pris dans le contexte (ligne voisine d'un tableau, autre année) avec 8 extraits ;
- ~8 : graphiques dont les valeurs ne sont pas dans le texte : le modèle devine ;
- réponses partielles (« ... ; pour la 2e partie : INTROUVABLE »), et une faute « INTOUVABLE » non reconnue.

### Corrections pour le run 2

- consigne de correction de prémisse supprimée (INTROUVABLE sur une fausse prémisse rapporte déjà +1) ;
- 5 extraits au lieu de 8, réponse en une ou deux phrases sans Markdown, exemple de format ;
- toute réponse qui contient INTROUVABLE (même mal orthographié) ou qui dit que l'information manque
  devient INTROUVABLE. Rejoué sur les réponses du run 1 : score brut 0,17 → **0,29** (10 réponses
  fausses écartées pour 1 juste) ;
- 2e appel au générateur : auto-vérification (OUI/NON) de la réponse sur les extraits cités, utilisée
  comme critère d'abstention si la calibration le juge utile.

Abstention : on répond INTROUVABLE si le modèle s'abstient, si le score du reranker du meilleur passage
est sous un seuil, ou (option) si la réponse contient des chiffres absents des extraits. Le seuil et
l'option sont choisis pour maximiser le score de confiance sur validation (`rag2/calibrate.py`).
