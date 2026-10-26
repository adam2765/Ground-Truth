"""Génération : les meilleurs chunks + la question -> le générateur officiel -> réponse + citations.

Un seul appel au générateur par question (budget : 4 appels, 30 s en moyenne sur T4).
Pour chaque question, on garde aussi des indicateurs qui servent à calibrer l'abstention
(rag2/calibrate.py) : score du reranker, abstention du modèle, chiffres de la réponse absents des extraits.
"""
import re

from groundtruth.normalize import _close, extract_numbers

MAX_INPUT = 4096
MAX_NEW = 300
MAX_CHUNKS = 8

SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français (Insee, Cour des comptes), "
    "UNIQUEMENT à partir des extraits fournis. Chaque extrait commence par sa référence [doc_id p.N].\n"
    "Règles :\n"
    "1. Si les extraits ne permettent pas de répondre, écris exactement : INTROUVABLE\n"
    "2. Si la question affirme quelque chose que les extraits contredisent, corrige d'abord cette affirmation, "
    "puis donne l'information exacte.\n"
    "3. Sinon, réponds en une ou deux phrases, en français, même si la question est en anglais. "
    "Recopie les chiffres exactement comme dans les extraits, avec leur unité (%, euros, milliers, M€...). "
    "Si un calcul simple est nécessaire (écart, somme, évolution), donne le résultat et les chiffres utilisés.\n"
    "4. Termine par une dernière ligne : SOURCES: [doc_id p.N], en ne citant que les extraits qui contiennent "
    "la réponse."
)


def format_extract(c: dict) -> str:
    return f"[{c['doc_id']} p.{c['page']}]\n{c['body']}"


def build_messages(question: str, extracts: list[str]) -> list[dict]:
    user = "Extraits :\n\n" + "\n\n".join(extracts) + f"\n\nQuestion : {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def fit_context(gen, question: str, chunks: list[dict]) -> list[dict]:
    """Ajoute les chunks (du meilleur au moins bon) tant que le prompt tient dans 4 096 tokens."""
    kept = []
    for c in chunks[:MAX_CHUNKS]:
        trial = kept + [c]
        if gen.count_tokens(build_messages(question, [format_extract(x) for x in trial])) > MAX_INPUT - 16:
            continue                                   # trop long : on essaie le suivant (plus court ?)
        kept = trial
    if not kept and chunks:                            # même le premier ne tient pas : on le tronque
        c = dict(chunks[0])
        words = c["body"].split()
        while words and gen.count_tokens(build_messages(question, [format_extract(c)])) > MAX_INPUT - 16:
            words = words[: int(len(words) * 0.8)]
            c["body"] = " ".join(words)
        kept = [c]
    return kept


_REF = re.compile(r"\[([\w\-]+)\s*,?\s*p\.?\s*(\d+)\]")


def parse_answer(raw: str, context: list[dict]) -> tuple[str, list[dict], bool]:
    """-> (réponse, citations, le modèle s'est abstenu). Citations limitées aux pages du contexte."""
    text = raw.strip()
    if re.match(r"^\W*INTROUVABLE\W*$", text.split("\n")[0].strip(), re.I) or text.upper().startswith("INTROUVABLE"):
        return "INTROUVABLE", [], True
    answer = re.split(r"\n?\s*\**SOURCES?\**\s*:", text, flags=re.I)[0].strip()
    answer = re.sub(r"\s*" + _REF.pattern, "", answer).strip() or answer
    allowed = {(c["doc_id"], c["page"]) for c in context}
    cits, seen = [], set()
    for d, p in _REF.findall(text):
        key = (d, int(p))
        if key in allowed and key not in seen:
            seen.add(key)
            cits.append({"doc_id": d, "page": int(p)})
    if not cits and context:                           # pas de source exploitable : la meilleure page
        cits = [{"doc_id": context[0]["doc_id"], "page": context[0]["page"]}]
    return answer, cits[:5], False


def unsupported_numbers(answer: str, context: list[dict], question: str) -> int:
    """Nombre de valeurs de la réponse qui n'apparaissent ni dans les extraits ni dans la question.

    Indice d'hallucination. On ignore les années (1900-2100) et les petits entiers (<= 10), trop
    fréquents pour être informatifs. Un calcul légitime (écart de deux valeurs) compte aussi : c'est
    la calibration qui décide si cet indicateur doit déclencher une abstention.
    """
    ctx = " ".join(c["body"] for c in context) + " " + question
    ctx_nums = extract_numbers(ctx)
    n = 0
    for raw, scaled in extract_numbers(answer):
        if float(raw).is_integer() and (1900 <= raw <= 2100 or abs(raw) <= 10):
            continue
        if not any(_close(raw, r) or _close(scaled, s) or _close(raw, s) or _close(scaled, r) for r, s in ctx_nums):
            n += 1
    return n


def answer_one(gen, q: dict, chunks: list[dict], hits: list[dict]) -> dict:
    gen.start_question(q["qid"])
    cand = [chunks[h["idx"]] for h in hits]
    ctx = fit_context(gen, q["question"], cand)
    raw = gen.chat(build_messages(q["question"], [format_extract(c) for c in ctx]), max_new_tokens=MAX_NEW)
    answer, cits, llm_abstain = parse_answer(raw, ctx)
    scores = [h.get("score", 0.0) for h in hits]
    return {
        "qid": q["qid"], "answer": answer, "citations": cits,
        "_meta": {
            "raw": raw,
            "top_score": scores[0] if scores else None,
            "margin": (scores[0] - scores[1]) if len(scores) > 1 else None,
            "llm_abstain": llm_abstain,
            "unsupported": 0 if llm_abstain else unsupported_numbers(answer, ctx, q["question"]),
            "n_context": len(ctx),
        },
    }
