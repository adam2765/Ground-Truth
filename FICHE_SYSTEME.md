# Fiche système : Ground Truth (Adam Fatah)

RAG sur 130 rapports publics français (Insee Première, Cour des comptes) : répond avec citation `[doc p.N]`, ou
`INTROUVABLE` quand la réponse n'est pas sûre. Générateur imposé : Ministral-3-3B (4 096 tokens d'entrée,
4 appels par question, GPU T4).

## Pipeline

```
INDEXATION (run.sh index, 18 min ; run.sh update, 4 min)
  PDF ─► parse.py : texte PyMuPDF + OCR Tesseract (pages scannées, texte des images)
      ─► tables.py : tableaux structurés (en-tête de colonne devant chaque valeur)
      ─► chunking.py : naive_title (fenêtres de 200 mots, recouvrement 50, titre du document en tête)
      ─► BM25 (bm25s, racinisation française) + bge-m3 (dense.npy)  ─► docmeta.py : graphe des éditions

RÉPONSE (run.sh answer, 22 s/question en moyenne)
  question ─► recherche par sous-question (split_question)
           ─► BM25 + dense, fusion RRF ─► 40 candidats ─► reranker bge-reranker-v2-m3 (cross-encoder)
           ─► éditions : ajout de l'édition récente, retrait de l'édition remplacée (sauf évolution)
           ─► contexte : 3 lignes clés + 5 extraits datés + rangées de tableau structurées
           ─► appel 1 : réponse courte, avec les tableaux structurés
           ─► appel 2 : citation de l'extrait puis réponse, SANS les tableaux structurés (lecture différente)
           ─► appel 3 (si besoin) : vérification sur les 4 rangées de tableau les plus proches,
                                    ou repêchage d'un INTROUVABLE quand un tableau est en jeu
           ─► calibrate.py : INTROUVABLE si les deux lectures ne concordent pas, ou si un chiffre de la
                             réponse est absent des extraits
```

## Composants

| Brique | Choix | Pourquoi (mesure) |
|---|---|---|
| Parsing | PyMuPDF + OCR Tesseract fra | Valeurs clés des scans présentes : 22 % → 94 %. pymupdf4llm aussi bon mais 5× plus lent |
| Découpage | naive_title, 200 mots | Meilleur Recall@5 avec reranker (0,948) ; une page par chunk : 0,890 |
| Recherche | BM25 + bge-m3, RRF | Recall@5 : BM25 0,78, dense 0,85, hybride 0,84-0,88 |
| Reranker | bge-reranker-v2-m3 | La brique qui rapporte le plus : Recall@5 0,844 → 0,948, MRR 0,81 |
| Concordance | 2 lectures sur 2 contextes différents | +0,355 → +0,465 (abstention sans seuil fin) |
| Tableaux structurés | 1re lecture seulement | Dans les deux lectures : +0,445 (sous la référence +0,465, elles se trompent ensemble) ; dans la 1re seulement : +0,475 |
| 3e appel ciblé, repêchage | rangées de tableau | +0,475 → +0,505 (run 9) → +0,525 (run 12, estimé) |
| Graphe des éditions | titres + dates, sans LLM | Palier 3 : la plus récente fait foi, sauf question d'évolution |

Modèles (tous open weights, téléchargés par `setup.sh`) : Ministral-3-3B (générateur officiel, inchangé),
BAAI/bge-m3 (embeddings), BAAI/bge-reranker-v2-m3 (reranker). Aucun fine-tuning. Pas de base vectorielle : la
recherche dense exacte (produit matriciel numpy sur ~11 000 vecteurs) prend ~1 ms.

## Résultats (validation, juge officiel)

| | v1 (100 q.) | v2 (40 q.) |
|---|---|---|
| Score de confiance (répétition générale du 9/11, + correctif des questions datées) | +0,42 → +0,455 | +0,2375 |
| Meilleur score v1 avant le palier 3 (run 12, estimé) | +0,525 | |
| Recall@5 de la recherche | 0,87 | 0,94 |

Points forts : questions pièges (+0,91 sur v1 : le système s'abstient), tableaux (+0,41 / +0,67), texte.
Points faibles : graphiques (valeurs dessinées, pas écrites), synthèses, questions temporelles. Les erreurs
restantes viennent surtout du générateur, qui choisit le mauvais chiffre dans une page pourtant bien retrouvée.

## Contraintes respectées

- 30 s par question en moyenne : 22,2 s mesurées, tout compris (max 47 s sur une question).
- 4 appels au générateur : 3 au maximum, 2,1 en moyenne (`runs/generator_log.jsonl`).
- `run.sh update` < 20 min : 4 min 21 s. `run.sh index` < 2 h : 18 min 22 s.
- Pas de réseau après `setup.sh` (`HF_HUB_OFFLINE=1`), pas de cache précalculé (`RAG_NO_CACHE=1`).
- Palier 1 from scratch dans `my_rag/` (BM25, dense, chunking sans librairie de recherche).

## Pistes testées et rejetées (détail : PALIER2.md, PALIER3.md)

Description des graphiques par modèle de vision (runs 4 et 14 : score brut meilleur, score final moins bon,
trop lent) ; vote à 3 ; vérificateur NLI ; 2e lecture sur pages entières ; 3e appel généralisé ; auto-vérification
par le générateur ; traduction des questions anglaises ; vérification des calculs par Python.
Transcription des pages numérisées par Qwen3-VL-2B (+1 question nette, 47 min d'indexation en plus, boucles de
répétition). Dates seulement quand deux éditions sont présentes (pas de gain).
Non tentées, avec la raison : fine-tuning du reranker (gain attendu dans le bruit de mesure, poids de 1 Go à
héberger hors GitHub) ;
RAG agentique piloté par le modèle (budget de 4 appels, modèle trop petit pour décider) ; quantification
(générateur imposé, et le temps est pris par la génération, pas par la recherche).
