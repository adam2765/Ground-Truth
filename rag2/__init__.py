"""Palier 2 : le pipeline (parsing + OCR, chunking, recherche hybride, reranker, abstention calibrée).

Le palier 1 (codé from scratch) reste dans my_rag/. Ici, les librairies sont autorisées.

Commandes principales (depuis la racine du dépôt) :
    python -m rag2.pipeline index    --pdf-dir corpus/pdf --index-dir index2
    python -m rag2.ablation          --index-dir index2 --questions data/validation_v1.json
    python -m rag2.pipeline answer   --index-dir index2 --questions data/validation_v1.json --out runs/val_raw.json
    python -m rag2.calibrate fit     --raw runs/val_raw.json --report runs/val_report.json --out index2/abstention.json
    python -m rag2.calibrate apply   --raw runs/cls_raw.json --rule index2/abstention.json --out runs/classement_v1.json
"""
import os as _os

# bm25s importe JAX s'il est installé (Colab) : JAX réserverait 75 % de la VRAM. On le force sur CPU.
_os.environ.setdefault("JAX_PLATFORMS", "cpu")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
