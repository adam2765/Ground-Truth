"""Étape 6 : RAG de base. Recherche -> contexte -> générateur officiel -> réponse + citations.

    python -m my_rag.rag --questions data/validation_v1.json --out runs/rag_val1.json --backend echo
    python -m my_rag.rag --questions data/validation_v1.json --out runs/rag_val1.json --backend hf     # sur GPU
    python -m groundtruth.evaluate --pred runs/rag_val1.json --gold data/validation_v1.json --judge none

Un seul appel au générateur par question. Le contexte = les meilleurs chunks, chacun précédé de sa
référence [doc_id p.N] ; on en met autant que le budget de 4 096 tokens le permet.
"""
import argparse
import re

from groundtruth.generator import OfficialGenerator
from groundtruth.io import load_questions, write_predictions
from my_rag.chunk import chunk_pages
from my_rag.eval_retrieval import build_retriever
from my_rag.extract import extract_pages

MAX_INPUT = 4096
MAX_NEW = 256

SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français, uniquement à partir des extraits fournis. "
    "Chaque extrait est précédé de sa référence [doc_id p.N]. "
    "Si la réponse n'est pas dans les extraits, réponds exactement : INTROUVABLE. "
    "Sinon, réponds en une ou deux phrases avec les chiffres exacts, puis indique tes sources sur une "
    "dernière ligne au format : SOURCES: [doc_id p.N], [doc_id p.N]"
)


def build_messages(question: str, contexts: list[str]) -> list[dict]:
    user = "Extraits :\n\n" + "\n\n".join(contexts) + f"\n\nQuestion : {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def fit_context(gen, question: str, hits: list[dict], max_chunks: int = 8) -> list[dict]:
    """Ajoute les chunks un par un (du meilleur au moins bon) tant que le prompt tient dans le budget."""
    kept = []
    for h in hits[:max_chunks]:
        trial = kept + [h]
        msgs = build_messages(question, [f"[{c['doc_id']} p.{c['page']}]\n{c['text']}" for c in trial])
        if gen.count_tokens(msgs) > MAX_INPUT - 32:            # petite marge de sécurité
            break
        kept = trial
    return kept


def parse_answer(raw: str) -> tuple[str, list[dict]]:
    """Sépare la réponse des sources, et extrait les citations [doc_id p.N] (5 au maximum, sans doublon)."""
    if "INTROUVABLE" in raw.upper():
        return "INTROUVABLE", []
    answer = re.split(r"\n?SOURCES\s*:", raw, flags=re.I)[0].strip()
    cits, seen = [], set()
    for d, p in re.findall(r"\[([\w\-]+)\s+p\.?\s*(\d+)\]", raw):
        if (d, int(p)) not in seen:
            seen.add((d, int(p)))
            cits.append({"doc_id": d, "page": int(p)})
    return answer, cits[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/validation_v1.json")
    ap.add_argument("--out", default="runs/rag_val1.json")
    ap.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--backend", choices=["hf", "openai", "echo"], default="hf")
    args = ap.parse_args()

    chunks = chunk_pages(extract_pages("corpus/pdf"))
    retriever = build_retriever(args.method, chunks)
    gen = OfficialGenerator(backend=args.backend)

    preds = []
    for q in load_questions(args.questions):
        gen.start_question(q["qid"])
        hits = [chunks[i] for i, _ in retriever.search(q["question"], k=50)]
        ctx = fit_context(gen, q["question"], hits)
        msgs = build_messages(q["question"], [f"[{c['doc_id']} p.{c['page']}]\n{c['text']}" for c in ctx])
        raw = gen.chat(msgs, max_new_tokens=MAX_NEW)
        answer, cits = parse_answer(raw)
        # si le modèle répond sans citer, on cite le meilleur chunk (citation absente = 0,5 au mieux)
        if answer != "INTROUVABLE" and not cits and ctx:
            cits = [{"doc_id": ctx[0]["doc_id"], "page": ctx[0]["page"]}]
        retrieved, seen = [], set()
        for h in hits:
            if (h["doc_id"], h["page"]) not in seen:
                seen.add((h["doc_id"], h["page"]))
                retrieved.append({"doc_id": h["doc_id"], "page": h["page"]})
        preds.append({"qid": q["qid"], "answer": answer, "citations": cits, "retrieved": retrieved[:20]})
        print(f"{q['qid']}: {len(ctx)} chunks | {answer[:80]}")
    write_predictions(args.out, preds, run_name=f"palier1-{args.method}")
    print("écrit :", args.out)


if __name__ == "__main__":
    main()
