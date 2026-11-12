# Palier 3 : corpus qui évolue, questions temporelles, anglais

Corpus v2 : 20 documents de plus (16 Insee Première de 2026, 4 rapports de la Cour des comptes d'octobre 2026),
dont plusieurs **nouvelles éditions** de documents du corpus v1 (comptes nationaux, salaires publics, comptes de
l'agriculture, émissions de gaz à effet de serre...). Validation v2 : 40 questions, dont 12 temporelles et 6 en
anglais.

## 1. Mise à jour incrémentale (`run.sh update`, `rag2/pipeline.py`, `cmd_update`)

Seuls les nouveaux PDF sont lus (OCR compris), découpés et encodés par bge-m3. Leurs vecteurs sont **ajoutés à la
fin** de `dense.npy` : les numéros des anciens chunks ne changent pas, rien n'est recalculé pour le corpus v1.
BM25 est reconstruit au chargement (quelques secondes), les tableaux structurés des nouveaux documents sont
ajoutés à `tables.json.gz`, et le graphe des éditions est recalculé.

| Étape (répétition générale du 9/11, Kaggle T4, tout recalculé) | Temps | Limite |
|---|---|---|
| `setup.sh` (dépendances + poids, seule étape avec réseau) | 2 min 25 s | |
| `run.sh index` (110 documents v1, OCR + tableaux + embeddings) | 18 min 22 s | 2 h |
| `run.sh update` (20 documents v2) | 4 min 21 s | 20 min |

## 2. Graphe des éditions (`rag2/docmeta.py`)

Version légère de GraphRAG : un nœud par document, une arête « édition plus récente » entre deux documents de la
même famille (Insee / Cour des comptes) dont les titres sont proches (cosinus bge-m3 ≥ 0,80) et les dates
différentes. Dates : `corpus/manifest.json` (à défaut, l'année la plus récente de la page 1). Sans modèle de
langage : construit en quelques secondes. Pour la Cour des comptes, il faut en plus le même nom de rapport
(`_stem`), sinon deux rapports annuels sur des sujets différents étaient reliés à tort.

Résultat : 17 documents ont une édition plus récente (ex. comptes nationaux 2024 → 2025, salaires de la
fonction publique d'État 2023 → 2024, compte de l'agriculture provisoire 2024 → prévisionnel 2025 → provisoire 2025).

Utilisation à la recherche (`add_newer_editions`, `rag2/search.py`) :
- si un des 5 meilleurs extraits vient d'un document remplacé, on ajoute juste après lui le meilleur extrait
  de l'édition récente (choisi par le reranker parmi ses 5 chunks les plus proches ; seuil 0,3) ;
- **filtre VersionRAG** : pour une question sur le contenu (pas sur une évolution), on retire les extraits des
  éditions remplacées, pour que le modèle 3B n'ait pas deux chiffres contradictoires sous les yeux. Une question
  d'évolution (« révisé », « successive estimates », « depuis », « entre … et … ») garde les deux éditions ;
- chaque extrait porte sa date de publication (`[insee-ip2105 p.1] (publié le 2026-05-28)`), et la consigne dit
  que la publication la plus récente fait foi.

**Correctif du 11/11 (questions datées)** : le filtre retirait aussi l'ancienne édition quand la question vise
une année que seule celle-ci couvre (« En 2023, combien gagne un enseignant de l'État ? » : l'édition 2026 donne
2024). 3 questions de la validation v1 perdues (val1-011, val1-039, val1-072). Option `--year-keep` : on garde
l'ancienne édition si la question donne une année sans « dernier / plus récent ». Testé par `scripts/run_years.sh`
(11/11, `results/years/`) : v1 +0,42 → +0,455 (val1-011, 039, 072 retrouvent leur preuve), mais v2 +0,2375 →
+0,2125 : val2-006 (« compte provisoire… en 2025 ») gardait le compte **prévisionnel** 2025 (8,4 Md€) à côté du
compte provisoire 2025 (8,0 Md€), et le modèle a pris le mauvais. **Règle affinée** (`ref_year`, année du titre
de chaque édition) : on garde l'ancienne édition seulement si elle porte sur l'année demandée **et** que la
nouvelle porte sur une année postérieure. Sur les 11 questions concernées, cette règle reproduit exactement l'une
des deux recherches du test, d'où le résultat sans nouveau run : **v1 +0,455, v2 +0,2375**. Activée dans `run.sh`.

## 3. Questions en anglais

- Détection : plus de mots-outils anglais que français (`is_english`).
- Recherche : la racinisation française de BM25 ne sert à rien en anglais ; les candidats du reranker sont les
  35 meilleurs du dense (bge-m3 est multilingue) + les 5 meilleurs de BM25, sans la fusion RRF.
- Réponse : consigne ajoutée (répondre en anglais, garder les nombres et unités du document) ; la détection des
  réponses hésitantes et la lecture des nombres (« 246,600 » = 246 600) gèrent l'anglais.
- **Traduction de la question en français avant de répondre : testée, rejetée.** Un appel court de plus (+2 s),
  aucun gain (v2 0,2375 dans les deux cas), et des termes perdus (« self-employed » traduit « professions
  libérales »).
- Sur les 6 questions anglaises de la validation, la recherche trouve la preuve pour les 6 ; les échecs viennent
  du raisonnement (calcul sur deux documents, estimations successives), pas de la langue.

## 4. Variante rapide (duoXRSf) et temps

duoXRSf : la 2e lecture répond en format court au lieu de recopier l'extrait. Plus rapide, mais moins bonne.

| Run | Variante | v1 | v2 | Temps de génération |
|---|---|---|---|---|
| v2opti1 (4/11) | duoXRS | +0,46 | +0,225 | 25,9 s |
| v2opti1 (4/11) | duoXRSf | | +0,19 | 19,6 s |
| v2opti2 (5/11) | duoXRS | +0,44 | +0,2375 | 26,2 s |
| v2opti2 (5/11) | duoXRSf | +0,39 | +0,20 | 19,9 s |

duoXRS est retenue : la limite du README est de 30 s par question **en moyenne**, tout compris.

## 5. Règle d'abstention de la finale (`rag2/abstention_final.json`)

Choisie sur la grille de `rag2/calibrate.py` (concordance × chiffres absents des extraits × seuil du reranker ×
dispense pour les réponses confirmées par les rangées de tableau), en maximisant la moyenne v1 / v2 :
concordance souple par inclusion, aucun chiffre absent des extraits, pas de seuil, dispense pour les rangées.

| Règle (mêmes réponses, run v2opti2) | v1 | v2 |
|---|---|---|
| ancienne (run 12 : concordance stricte, seuil 0,593, citation exigée) | +0,46 | +0,1875 |
| **finale** | +0,44 | **+0,2375** |

À nombre de questions égal, les deux se valent (0,382) ; la nouvelle est meilleure sur les questions du palier 3,
présentes dans le test.

## 6. Répétition générale (9/11, `scripts/run_final.sh`, `results/final2/`)

Les commandes exactes du jour J (`setup.sh`, `run.sh index`, `update`, `answer`), sans aucun cache, juge officiel.

| | v1 (100 q.) | v2 (40 q.) |
|---|---|---|
| Score de confiance | **+0,42** | **+0,2375** |
| Score brut (sans abstention) | +0,23 | +0,05 |
| Recall@5 / Recall@10 (questions avec preuve) | 0,87 / 0,88 | 0,94 / 1,00 |
| Temps tout compris (recherche + génération + abstention) | 19,7 s/q | 28,6 s/q |

- Moyenne sur les 140 questions : **22,2 s par question** (limite 30 s). 296 appels au générateur pour 140
  questions, **3 au maximum** par question (limite 4).
- Reproductibilité : v2 identique au run du 5/11 (+0,2375) ; v1 +0,42 contre +0,44, 5 réponses sur 100
  différentes (index recalculé sans cache, calcul en float16 pas parfaitement déterministe). Sous le seuil de
  0,05 de la vérification du mentor.
- Par type (v2) : tableau +0,67, multi-doc +0,29, texte +0,25, temporel +0,125, graphique 0, synthèse 0.

**Pourquoi la v1 est passée de +0,525 (run 12) à +0,42.** Le palier 3 a un coût sur les questions du palier 2 :

| Étape | v1 | v2 |
|---|---|---|
| Run 12 (fin du palier 2, estimation) | +0,525 | |
| + dates dans les extraits et consigne « la plus récente fait foi » (1re répétition, 4/11) | +0,475 | +0,20 |
| + filtre VersionRAG, anglais (v2opti1, ancienne règle d'abstention → nouvelle) | +0,46 | +0,225 |
| + règle d'abstention choisie sur v1 + v2 (v2opti2) | +0,44 | +0,2375 |
| Répétition générale sans cache (9/11) | +0,42 | +0,2375 |
| + correctif des questions datées (11/11, règle affinée) | **+0,455** | **+0,2375** |

Les dates et la consigne temporelle semblaient coûter ~0,05 sur la v1. **Testé le 12/11 et démenti**
(`results/temporal/`, variante duoXRSt : dates et consigne seulement si deux éditions sont dans le contexte ou si
la question porte sur une évolution ; 74 questions v1 et 17 v2 régénérées sans dates) : v1 +0,455 → +0,445, v2
+0,2375 → +0,2375. 8 questions basculent dans un sens ou dans l'autre, sans tendance : l'écart entre le run 12 et
la 1re répétition venait du bruit (et de l'estimation de +0,525), pas des dates. duoXRS est gardée ; le filtre VersionRAG retire la preuve de 3 questions v1 datées (corrigé par `--year-keep`, section 2).
Ces choix sont gardés parce que le test mélange les deux types de questions (dont 30 en anglais) : sans eux,
la v2 reste à +0,1875 / +0,20.

Soumissions v2 : 2026-11-04_1 (1re répétition, +0,20), 2026-11-04_2 (duoXRS + VersionRAG, +0,225),
2026-11-05_1 (règle choisie sur v1 + v2, **+0,2375**).

## 7. Ce qui reste difficile

- **Questions temporelles** (+0,125) : il faut comparer deux éditions et calculer l'écart entre deux estimations ;
  le modèle 3B prend souvent le chiffre d'une seule édition, ou les mauvais chiffres.
- **Vérification des calculs par Python** (Program of Thoughts / PAL, `rag2/calc.py`) : testée hors ligne sur les
  140 réponses, rejetée. Le modèle calcule juste, mais sur les mauvais opérandes (val2-004 : « 2 228 − 2 150 = 78 »
  est exact, il fallait 26 740 − 25 760 = 980). Remplacer la règle « chiffre absent des extraits » par « chiffre ni
  écrit ni recalculable » récupère 3 réponses justes mais en laisse passer 7 fausses (−0,02 sur v1, −0,075 sur v2).

## 8. Tableaux des pages numérisées lus par un modèle de vision (12/11, 3 essais, retenu)

`rag2/scan_vlm.py` (option `RAG_SCAN_VLM`, non activée dans `run.sh`) : les pages numérisées sont transcrites par
Qwen3-VL au lieu de Tesseract, tableaux en Markdown réécrits au format « rangée | en-tête : valeur ». Garde-fous :
repli sur Tesseract si le modèle manque ou plante, budget de temps, transcription trop courte ignorée.

- Essai 1 (Qwen3-VL-4B, les 97 pages) : 77 s/page. Les pages sans texte (garde, graphiques) font écrire au modèle
  des cases vides jusqu'à la limite ; budget de 60 min épuisé avant les Insee Première numérisés. Rien de mesuré.
- Essai 2 (Qwen3-VL-2B, les 36 pages riches en chiffres, `results/scans/`) : 47 min ; 23 transcriptions gardées,
  13 rejetées ; sur 24 des 36 pages, le modèle finit par répéter les mêmes lignes en boucle. v1 +0,455 → +0,465
  (val1-015 et val1-088 gagnées grâce à des tableaux mieux lus, val1-090 perdue à cause de chiffres faux dans une
  transcription), v2 inchangée.

Essai 2 non retenu : +1 question nette pour 47 min d'indexation, du texte parasite (boucles) dans l'index, et une
réponse fausse causée par des chiffres mal lus qui remplaçaient le texte Tesseract.

- **Essai 3 (retenu, activé dans `run.sh`)** : le modèle ne transcrit **que les tableaux** ; le texte des pages
  reste celui de Tesseract (recherche inchangée). Les tableaux transcrits rejoignent les tableaux structurés, que
  seule la 1re lecture voit : un chiffre mal lu fait diverger les deux lectures, d'où une abstention plutôt
  qu'une erreur (même principe que le run 7). Boucles coupées (ligne répétée plus de 2 fois). 21 min pour
  36 pages, 19 pages avec tableau. v1 +0,455 → **+0,475** : val1-020 gagnée (5 699 prénoms mixtes, lu dans le
  tableau ; Tesseract l'avait abîmé), val1-086 et val1-088 passent de faux à abstention, val1-090 (synthèse)
  perdue : le modèle s'est concentré sur les montants du tableau au lieu des comportements d'achat. v2 inchangée.
  Repli : modèle absent ou en erreur -> Tesseract seul ; budget de 60 min.
