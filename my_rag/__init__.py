"""Palier 1 : RAG from scratch (sans LangChain, LlamaIndex, FAISS, bm25s ni base vectorielle).

Lancer les scripts depuis la racine du dépôt, sous forme de modules :
    python -m my_rag.extract
    python -m my_rag.chunk
    python -m my_rag.check_bm25
    python -m my_rag.eval_retrieval --method bm25
    python -m my_rag.eval_retrieval --method dense
    python -m my_rag.rag --questions data/validation_v1.json --out runs/rag_val1.json --backend echo
"""
