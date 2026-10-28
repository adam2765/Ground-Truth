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
import time

from groundtruth.normalize import _close, extract_numbers
from rag2.tables import render

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

SYSTEM_QUOTE = (
    "Tu réponds à des questions sur des rapports publics français (Insee, Cour des comptes), "
    "UNIQUEMENT à partir des extraits fournis. Chaque extrait commence par sa référence [doc_id p.N].\n"
    "Si la réponse exacte n'est pas écrite dans les extraits, écris seulement : INTROUVABLE\n"
    "Sinon, réponds exactement sur trois lignes :\n"
    "EXTRAIT: \"<recopie mot pour mot la phrase ou la ligne de tableau qui contient la réponse>\"\n"
    "RÉPONSE: <une ou deux phrases courtes en français, sans mise en forme, avec les chiffres exacts et leur "
    "unité ; si la question demande un écart ou une somme, calcule-le à partir de l'extrait>\n"
    "SOURCES: [doc_id p.N]\n"
    "Vérifie que le chiffre correspond bien à l'année, au lieu et à la catégorie demandés.\n"
    "Exemple :\n"
    "EXTRAIT: \"une baisse du nombre des bénéficiaires de la PAC entre 2022 et 2023 de 308 000 à 276 600\"\n"
    "RÉPONSE: Le nombre de bénéficiaires est passé de 308 000 en 2022 à 276 600 en 2023.\n"
    "SOURCES: [cc-pac p.25]"
)

# Variantes comparées sur la validation (scripts/run_palier2.sh garde la meilleure pour le leaderboard)
VARIANTS = {
    "court5": {"system": "plain", "chunks": 5},       # réponse courte, 5 extraits
    "cite5": {"system": "quote", "chunks": 5},        # recopie d'abord l'extrait, puis répond
    "cite3": {"system": "quote", "chunks": 3},        # idem avec 3 extraits seulement
    # run 3 : réponse court5 + 2e réponse indépendante au format cite5 ; si les chiffres des deux réponses
    # ne concordent pas, le modèle devine (calibration : rag2/calibrate.py, critère « agree »)
    "duo": {"system": "plain", "chunks": 5, "second": "quote"},
    # duo + les 3 lignes (phrases, rangées de tableau) les mieux notées par le reranker, en tête du prompt
    "duoL": {"system": "plain", "chunks": 5, "second": "quote", "key_lines": 3},
    # run 5 : vote à 3. Si les deux premières réponses divergent, un 3e appel (réponse courte sur les 3
    # meilleurs extraits seulement) départage : on garde la réponse soutenue par au moins 2 appels sur 3.
    "trioL": {"system": "plain", "chunks": 5, "second": "quote", "key_lines": 3, "third": 3},
    # run 6 : duoL + les tableaux des pages du contexte, réécrits avec l'en-tête de colonne devant chaque
    # valeur (rag2/tables.py) : « Europe du Sud | 2014 : 16 » au lieu de « Europe du Sud | 11 | 11 | 16 ».
    "duoT": {"system": "plain", "chunks": 5, "second": "quote", "key_lines": 3, "tables": 2},
    # run 6 bis : les deux réponses lisent des contextes différents. 1re : avec les tableaux structurés ;
    # 2e : sans. Avec duoT, les deux appels lisaient la même version du tableau et se trompaient ensemble
    # (même piège que les descriptions de graphiques du run 4) : la concordance filtrait moins bien.
    "duoX": {"system": "plain", "chunks": 5, "second": "quote", "key_lines": 3, "tables": 2, "second_no_tables": True},
    # run 8 : duoX + consigne de lecture des tableaux (rangée puis colonne) pour la 1re réponse
    "duoXH": {"system": "plain", "chunks": 5, "second": "quote", "key_lines": 3, "tables": 2, "second_no_tables": True,
              "table_hint": True},
    # run 8 : 2e réponse avec les mêmes tableaux mais présentés colonne par colonne (« 2014 : Europe du Sud 16 ;
    # Afrique 36… ») : l'information est là pour les deux réponses, mais leur lecture diffère.
    "duoC": {"system": "plain", "chunks": 5, "second": "quote", "key_lines": 3, "tables": 2, "second_by_column": True},
}
DEFAULT_VARIANT = "duo"


def _key_numbers(answer: str) -> list[float]:
    return [r for r, _ in extract_numbers(answer)
            if not (float(r).is_integer() and (1900 <= r <= 2100 or abs(r) <= 10))]


def agree_inclus(a: str, b: str) -> bool | None:
    """Concordance « inclusion » (run 5) : tous les chiffres de la réponse la plus courte se retrouvent dans
    l'autre. « de 308 000 en 2022 à 276 600 » et « 276 600 » concordent ; la règle exacte les rejetait."""
    if a == "INTROUVABLE" or b in (None, "INTROUVABLE"):
        return False
    na, nb = _key_numbers(a), _key_numbers(b)
    if not na and not nb:
        return None
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    return bool(short) and all(any(_close(x, y) for y in long_) for x in short)


def agree(a: str, b: str) -> bool | None:
    """Les deux réponses donnent-elles les mêmes chiffres ? None si aucune ne contient de chiffre."""
    if a == "INTROUVABLE" or b == "INTROUVABLE":
        return False
    na, nb = _key_numbers(a), _key_numbers(b)
    if not na and not nb:
        return None
    return all(any(_close(x, y) for y in nb) for x in na) and all(any(_close(y, x) for x in na) for y in nb)

VERIFY_SYSTEM = (
    "Tu vérifies une réponse. On te donne des extraits de rapports, une question et une réponse proposée. "
    "Réponds OUI si chaque chiffre de la réponse est écrit dans les extraits (ou obtenu par un calcul simple "
    "à partir d'eux) et correspond bien à ce que demande la question (bonne année, bon lieu, bonne catégorie). "
    "Sinon réponds NON. Réponds par un seul mot : OUI ou NON."
)


def format_extract(c: dict) -> str:
    return f"[{c['doc_id']} p.{c['page']}]\n{c['body']}"


TABLE_HINT = (
    "\n- Pour lire un tableau : repère d'abord la rangée de la catégorie demandée (lieu, population, secteur), "
    "puis la colonne de l'indicateur et de l'année demandés ; la réponse est la valeur à leur croisement. "
    "Ne prends pas la valeur d'une rangée ou d'une colonne voisine."
)


def build_messages(question: str, extracts: list[str], system: str = "plain", lines: list[dict] | None = None,
                   table_hint: bool = False) -> list[dict]:
    head = ""
    if lines:
        head = ("Lignes les plus pertinentes (à vérifier dans les extraits) :\n"
                + "\n".join(f"[{x['doc_id']} p.{x['page']}] {x['text']}" for x in lines) + "\n\n")
    user = head + "Extraits :\n\n" + "\n\n".join(extracts) + f"\n\nQuestion : {question}"
    sys_prompt = SYSTEM_QUOTE if system == "quote" else SYSTEM
    if table_hint:                                     # inséré avant la ligne « SOURCES » des consignes
        cut = sys_prompt.index("\n- Termine") if "\n- Termine" in sys_prompt else len(sys_prompt)
        sys_prompt = sys_prompt[:cut] + TABLE_HINT + sys_prompt[cut:]
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]


def _family(doc_id: str) -> str:
    """Un rapport et sa synthèse disent souvent la même chose : même « famille »."""
    return doc_id[: -len("-synthese")] if doc_id.endswith("-synthese") else doc_id


def diversify(chunks: list[dict], scores: list[float], max_chunks: int, min_other: float = 0.3) -> list[dict]:
    """Choisit les extraits du contexte en évitant qu'un seul document prenne toutes les places.

    - au plus ~60 % des places pour une même famille (rapport + synthèse) ;
    - si tout vient d'une seule famille mais qu'un extrait d'un autre document a un score du reranker
      >= min_other, il remplace le dernier : les questions multi-doc demandent de croiser deux rapports.
    """
    cap = max(1, round(max_chunks * 0.6))
    kept, count = [], {}
    for i, c in enumerate(chunks):
        f = _family(c["doc_id"])
        # plafond par famille, seulement s'il existe plus loin un extrait pertinent d'un autre document
        other = any(_family(o["doc_id"]) != f and sc is not None and sc >= min_other
                    for o, sc in zip(chunks[i + 1:], scores[i + 1:]))
        if count.get(f, 0) >= cap and other:
            continue
        kept.append(c)
        count[f] = count.get(f, 0) + 1
        if len(kept) == max_chunks:
            break
    if len(kept) < max_chunks:                         # pas assez d'autres documents : on complète dans l'ordre
        for c in chunks:
            if c not in kept:
                kept.append(c)
                if len(kept) == max_chunks:
                    break
    fams = {_family(c["doc_id"]) for c in kept}
    if len(fams) == 1 and len(kept) > 1:
        for c, sc in zip(chunks, scores):
            if _family(c["doc_id"]) not in fams and sc is not None and sc >= min_other:
                kept[-1] = c
                break
    # on garde l'ordre du reranker (le meilleur d'abord)
    order = {id(c): i for i, c in enumerate(chunks)}
    return sorted(kept, key=lambda c: order[id(c)])


def add_tables(chunks: list[dict], tables: dict[str, list[dict]], max_tables: int, question: str = "") -> list[dict]:
    """Insère, juste après le premier extrait de chaque page qui contient un tableau, ce tableau réécrit
    avec ses en-têtes de colonnes (au plus max_tables pages, en partant des meilleurs extraits)."""
    out, seen = [], set()
    for c in chunks:
        out.append(c)
        key = f"{c['doc_id']}|{c['page']}"
        if key in tables and key not in seen and len(seen) < max_tables:
            seen.add(key)
            out.append({"doc_id": c["doc_id"], "page": c["page"], "table": True,
                        "body": "Tableau de cette page, chaque valeur précédée du titre de sa colonne :\n"
                                + "\n\n".join(render(t, question) for t in tables[key]),
                        "body_col": "Tableau de cette page, lu colonne par colonne :\n"
                                    + "\n\n".join(render(t, question, by_column=True) for t in tables[key])})
    return out


def fit_context(gen, question: str, chunks: list[dict], max_chunks: int = MAX_CHUNKS,
                lines: list[dict] | None = None) -> list[dict]:
    """Ajoute les chunks (du meilleur au moins bon) tant que le prompt tient dans 4 096 tokens."""
    kept = []
    for c in chunks[:max_chunks]:
        trial = kept + [c]
        if gen.count_tokens(build_messages(question, [format_extract(x) for x in trial], "quote", lines)) > MAX_INPUT - 16:
            continue                                   # trop long : on essaie le suivant (plus court ?)
        kept = trial
    if not kept and chunks:                            # même le premier ne tient pas : on le tronque
        c = dict(chunks[0])
        words = c["body"].split()
        while words and gen.count_tokens(build_messages(question, [format_extract(c)], "quote", lines)) > MAX_INPUT - 16:
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
    if _ABSTAIN.search(text):
        return "INTROUVABLE", [], True
    answer = re.split(r"\n?\s*\**SOURCES?\**\s*:", text, flags=re.I)[0].strip()
    m = re.search(r"R[ÉE]PONSE\s*:\s*(.+)", answer, flags=re.I | re.S)
    if m:                                              # format « cite » : on ne garde que la réponse
        answer = m.group(1).strip()
    answer = re.sub(r"^\s*EXTRAIT\s*:.*$", "", answer, flags=re.I | re.M).strip()
    answer = answer.replace("**", "").replace("__", "")
    if _HEDGE.search(answer):                          # (sur la réponse seule : un EXTRAIT peut contenir « ne ... pas »)
        return "INTROUVABLE", [], True
    answer = re.sub(r"\s*" + _REF.pattern, "", answer).strip() or answer
    allowed = {(c["doc_id"], c["page"]) for c in context}   # (lignes clés : toujours issues des extraits)
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


def _norm(t: str) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", t))


def quote_check(raw: str, context: list[dict]) -> bool | None:
    """La phrase recopiée (EXTRAIT: "...") existe-t-elle vraiment dans les extraits ?

    Vrai si au moins 80 % de ses mots (et tous ses nombres) se trouvent dans un même extrait.
    None si le modèle n'a pas donné d'extrait (format « court »).
    """
    m = re.search(r"EXTRAIT\s*:\s*[\"«“]?(.+?)[\"»”]?\s*$", raw, flags=re.I | re.M)
    if not m:
        return None
    q = _norm(m.group(1)).split()
    if len(q) < 3:
        return False
    nums = [w for w in q if any(ch.isdigit() for ch in w)]
    for c in context:
        body = set(_norm(c["body"]).split())
        if sum(w in body for w in q) / len(q) >= 0.8 and all(n in body for n in nums):
            return True
    return False


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


def answer_one(gen, q: dict, chunks: list[dict], hits: list[dict], do_verify: bool = True,
               variant: str = DEFAULT_VARIANT, key_lines: list[dict] | None = None,
               tables: dict[str, list[dict]] | None = None) -> dict:
    t_start = time.time()
    cfg = VARIANTS[variant]
    lines = (key_lines or [])[: cfg.get("key_lines", 0)] or None
    gen.start_question(q["qid"])
    cand = [chunks[h["idx"]] for h in hits]
    picked = diversify(cand, [h.get("score") for h in hits], cfg["chunks"])
    n_max = cfg["chunks"]
    if cfg.get("tables") and tables:
        picked = add_tables(picked, tables, cfg["tables"], q["question"])
        n_max = len(picked)
    ctx = fit_context(gen, q["question"], picked, n_max, lines)
    hint = bool(cfg.get("table_hint")) and any(c.get("table") for c in ctx)
    while hint and len(ctx) > 1 and gen.count_tokens(build_messages(
            q["question"], [format_extract(c) for c in ctx], cfg["system"], lines, True)) > MAX_INPUT - 8:
        ctx = ctx[:-1]
    raw = gen.chat(build_messages(q["question"], [format_extract(c) for c in ctx], cfg["system"], lines, hint),
                   max_new_tokens=MAX_NEW + (100 if cfg["system"] == "quote" else 0))
    answer, cits, llm_abstain = parse_answer(raw, ctx)
    second = agreement = None
    if cfg.get("second") and not llm_abstain:
        ctx2 = [c for c in ctx if not c.get("table")] if cfg.get("second_no_tables") else ctx
        if cfg.get("second_by_column"):                # 2e lecture : mêmes tableaux, présentés par colonne
            ctx2 = [dict(c, body=c["body_col"]) if c.get("table") else c for c in ctx]
            while len(ctx2) > 1 and gen.count_tokens(build_messages(
                    q["question"], [format_extract(c) for c in ctx2], cfg["second"], lines)) > MAX_INPUT - 8:
                ctx2 = ctx2[:-1]                        # (présentation par colonne un peu plus longue)
        raw2 = gen.chat(build_messages(q["question"], [format_extract(c) for c in ctx2], cfg["second"], lines),
                        max_new_tokens=MAX_NEW + 100)
        second, cits2, _ = parse_answer(raw2, ctx2)
        agreement = agree(answer, second)
        do_verify = False                              # la concordance remplace l'auto-vérification (run 2 : inutile)
        third = None
        if cfg.get("third") and agreement is not True:
            ctx3 = ctx[: cfg["third"]]
            raw3 = gen.chat(build_messages(q["question"], [format_extract(c) for c in ctx3], "plain", lines),
                            max_new_tokens=150)
            third, cits3, _ = parse_answer(raw3, ctx3)
            a_c, b_c = agree(answer, third), agree(second, third)
            if a_c is True:                            # 1re et 3e réponses d'accord : on garde la 1re
                agreement = True
            elif b_c is True:                          # 2e et 3e d'accord : on prend la 2e
                answer, cits, agreement = second, (cits2 or cits3 or cits), True
            elif agreement is None and a_c is None and b_c is None:
                agreement = None                       # aucune réponse chiffrée
            else:
                agreement = False
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
            "quote_ok": None if llm_abstain else quote_check(raw, ctx),
            "variant": variant,
            "second": second,
            "agree": agreement,
            "third": locals().get("third"),
            "n_context": len(ctx),
            "n_tables": sum(bool(c.get("table")) for c in ctx),
            "seconds": round(time.time() - t_start, 2),
        },
    }
