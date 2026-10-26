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

Annotation manuelle de 30 pages difficiles : `annotation/annoter.html` (à remplir, export CSV).

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
| naive (200 mots, recouvrement 50) | 8 772 | 0,786 | 0,864 | 0,601 |
| naive_title (+ titre du document) | 8 772 | 0,792 | 0,851 | 0,588 |
| page (une page = un chunk) | 2 625 | 0,779 | 0,851 | 0,582 |
| struct_title (paragraphes ~180 mots + titre) | 6 974 | **0,805** | 0,838 | **0,609** |

Point faible restant : les questions multi-doc (Recall@5 = 0,40).

## 3. Recherche hybride, reranker, génération, abstention (GPU)

Résultats à compléter avec le notebook Colab : tableau `chunking × {bm25, dense, hybrid, hybrid_rerank}`,
score de confiance sur validation (juge officiel) avant et après calibration de l'abstention.

Abstention : on répond INTROUVABLE si le modèle s'abstient, si le score du reranker du meilleur passage
est sous un seuil, ou (option) si la réponse contient des chiffres absents des extraits. Le seuil et
l'option sont choisis pour maximiser le score de confiance sur validation (`rag2/calibrate.py`).
