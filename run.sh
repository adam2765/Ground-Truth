#!/usr/bin/env bash
# Contrat du run final. L'orga exécute EXACTEMENT ces commandes sur un GPU T4 16 Go, sans réseau
# (sauf setup.sh) :
#
#   bash setup.sh                                           # réseau autorisé : dépendances + poids
#   bash run.sh index  <dossier_pdf_v1> <dossier_index>     # corpus v1 (110 docs) : 2 h max
#   bash run.sh update <dossier_index> <dossier_pdf_v2>     # ajout des 20 docs v2 : 20 min max
#   bash run.sh answer <dossier_index> <questions.json> <sortie.json>   # 30 s par question en moyenne
#
# Notre pipeline (rag2/) : OCR + tableaux structurés + découpage naive_title + bge-m3 ; recherche hybride
# (BM25 + dense, RRF) + reranker bge-reranker-v2-m3, par sous-question ; graphe des éditions ; générateur
# officiel, variante duoXRS (2 lectures concordantes, vérification ciblée sur les rangées de tableau,
# repêchage) ; abstention calibrée sur la validation (rag2/abstention_final.json).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu RAG_NO_CACHE=1   # tout est recalculé
STRATEGY=naive_title; VARIANT=duoXRS; RULE=rag2/abstention_final.json
cmd="$1"; shift
case "$cmd" in
  index)
    mkdir -p "$2"
    echo "{\"strategy\": \"$STRATEGY\", \"mode\": \"hybrid_rerank\"}" > "$2/best.json"
    python -m rag2.pipeline index --pdf-dir "$1" --index-dir "$2" --strategies "$STRATEGY" ;;
  update)
    python -m rag2.pipeline update --index-dir "$1" --pdf-dir "$2" ;;
  answer)
    tmp=$(mktemp -d)
    python -m rag2.pipeline retrieve --index-dir "$1" --questions "$2" --out "$tmp/ret.json" --split
    python -m rag2.pipeline generate --index-dir "$1" --retrieval "$tmp/ret.json" --questions "$2" \
      --out "$tmp/raw.json" --variant "$VARIANT"
    mkdir -p "$(dirname "$3")"
    python -m rag2.calibrate apply --raw "$tmp/raw.json" --rule "$RULE" --out "$3"
    cp "$tmp/raw.json" "${3%.json}_raw.json" ;;
  *) echo "usage: run.sh index|update|answer ..." >&2; exit 1 ;;
esac
