"""Génération : les meilleurs chunks + la question -> le générateur officiel -> réponse + citations.

Deux appels au générateur par question (budget : 4 appels, 30 s en moyenne sur T4) :
  1. la réponse, à partir des 5 meilleurs extraits ;
  2. une auto-vérification : le modèle relit les extraits cités et dit si la réponse y est bien écrite.
Pour chaque question, on garde des indicateurs qui servent à calibrer l'abstention (rag2/calibrate.py) :
score du reranker, abstention du modèle, chiffres absents des extraits, verdict de l'auto-vérification.

Leçons du run 1 (validation, 35 réponses fausses sur 77 questions avec réponse, alors que la bonne page
était dans le top 5 dans 95 % des cas) :
  - la consigne « corrige la prémisse si elle est fausse » faisait dérailler le modèle sur des questions
    normales (« Correction de l'affirmation initiale : ... » puis une invention) ; elle est supprimée :
    répondre INTROUVABLE à une question à fausse prémisse rapporte déjà +1 ;
  - avec 8 extraits, le modèle (3B) prenait souvent le chiffre d'une ligne voisine : on passe à 5 ;
  - réponses longues en Markdown : on demande une phrase courte, sans mise en forme.
"""
import re

from groundtruth.normalize import _close, extract_numbers

MAX_INPUT = 4096
MAX_NEW = 200
MAX_CHUNKS = 5

SYSTEM = (
    "Tu réponds à des questions sur des rapports publics français (Insee, Cour des comptes), "
    "UNIQUEMENT à partir des extraits fournis. Chaque extrait commence par sa référence [doc_id p.N].\n"
    "Règles :\n"
    "- Si la réponse exacte n'est pas écrite dans les extraits, écris seulement : INTROUVABLE\n"
    "- Sinon, réponds en une ou deux phrases courtes, en français, sans mise en forme (pas de gras, pas de liste). "
    "Recopie les chiffres exactement comme dans l'extrait, avec leur unité. Vérifie que le chiffre correspond "
    "bien à l'année, au lieu et à la catégorie demandés. Si la question demande un écart ou une somme, "
    "calcule-le et donne aussi les chiffres utilisés.\n"
    "- Termine par une ligne : SOURCES: [doc_id p.N] (seulement les extraits qui contiennent la réponse).\n"
    "Exemple de réponse :\n"
    "Le nombre de bénéficiaires est passé de 308 000 en 2022 à 276 600 en 2023.\n"
    "SOURCES: [cc-pac p.25]"
)

VERIFY_SYSTEM = (
    "Tu vérifies une réponse. On te donne des extraits de rapports, une question et une réponse proposée. "
    "Réponds OUI si chaque chiffre de la réponse est écrit dans les extraits (ou obtenu par un calcul simple "
    "à partir d'eux) et correspond bien à ce que demande la question (bonne année, bon lieu, bonne catégorie). "
    "Sinon réponds NON. Réponds par un seul mot : OUI ou NON."
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
_ABSTAIN = re.compile(r"\bIN?TR?O?U?VABLE\b", re.I)
_HEDGE = re.compile(r"\b(ne|n['’])\s*(\w+\s+)?(mentionne|précise|fourni|donne|indique|contien|permet|détaille)\w*\s+pas"
                    r"|pas d['’]information|aucun extrait|non (précisé|mentionné|disponible)", re.I)


def parse_answer(raw: str, context: list[dict]) -> tuple[str, list[dict], bool]:
    """-> (réponse, citations, le modèle s'est abstenu). Citations limitées aux pages du contexte."""
    text = raw.strip()
    # INTROUVABLE n'importe où (y compris mal orthographié : « INTOUVABLE ») ou réponse qui dit elle-même
    # que l'information manque : dans le run 1, ces réponses partielles étaient presque toujours fausses.
    if _ABSTAIN.search(text) or _HEDGE.search(text):
        return "INTROUVABLE", [], True
    answer = re.split(r"\n?\s*\**SOURCES?\**\s*:", text, flags=re.I)[0].strip()
    answer = answer.replace("**", "").replace("__", "")
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


def verify(gen, question: str, answer: str, context: list[dict], cits: list[dict]) -> bool | None:
    """2e appel : le modèle relit les extraits cités et juge si la réponse y est bien écrite."""
    cited = [c for c in context if {"doc_id": c["doc_id"], "page": c["page"]} in cits] or context[:2]
    user = ("Extraits :\n\n" + "\n\n".join(format_extract(c) for c in cited)
            + f"\n\nQuestion : {question}\n\nRéponse proposée : {answer}\n\nLa réponse est-elle exacte d'après les extraits ?")
    msgs = [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": user}]
    if gen.count_tokens(msgs) > MAX_INPUT - 8:
        return None
    out = gen.chat(msgs, max_new_tokens=4).strip().upper()
    if out.startswith("OUI") or out.startswith("YES"):
        return True
    if out.startswith("NON") or out.startswith("NO"):
        return False
    return None


def answer_one(gen, q: dict, chunks: list[dict], hits: list[dict], do_verify: bool = True) -> dict:
    gen.start_question(q["qid"])
    cand = [chunks[h["idx"]] for h in hits]
    ctx = fit_context(gen, q["question"], cand)
    raw = gen.chat(build_messages(q["question"], [format_extract(c) for c in ctx]), max_new_tokens=MAX_NEW)
    answer, cits, llm_abstain = parse_answer(raw, ctx)
    verified = None if (llm_abstain or not do_verify) else verify(gen, q["question"], answer, ctx, cits)
    scores = [h.get("score", 0.0) for h in hits]
    return {
        "qid": q["qid"], "answer": answer, "citations": cits,
        "_meta": {
            "raw": raw,
            "top_score": scores[0] if scores else None,
            "margin": (scores[0] - scores[1]) if len(scores) > 1 else None,
            "llm_abstain": llm_abstain,
            "unsupported": 0 if llm_abstain else unsupported_numbers(answer, ctx, q["question"]),
            "verified": verified,
            "n_context": len(ctx),
        },
    }
