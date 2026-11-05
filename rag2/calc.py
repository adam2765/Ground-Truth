"""Vérification des calculs par Python (idée « Program of Thoughts » / PAL : Chen et al. 2022, arXiv 2211.12588 ;
Gao et al. 2022, arXiv 2211.10435). Le modèle 3B se trompe souvent en calculant (val2-004 : « 2 228 - 2 150 = 78 »
au lieu de 26 740 - 25 760 = 980). On ne lui fait pas confiance : chaque chiffre de la réponse doit soit figurer
dans les extraits, soit se recalculer exactement à partir de deux chiffres de la réponse qui y figurent
(écart, somme, rapport, évolution en %). Aucun appel au modèle : 0 s.

  derived(answer, ctx, question)  -> nombre de chiffres de la réponse ni écrits ni recalculables
  bad_equations(answer)            -> nombre d'égalités « a - b = c » fausses écrites par le modèle

Testé hors ligne le 5/11 sur les réponses du run v2opti2 (duoXRS, v1 + v2, 140 questions) : REJETÉ.
- aucune égalité fausse : le modèle calcule juste, il se trompe d'opérandes (val2-004 : 2 228 - 2 150 = 78 est
  exact, mais ce sont les mauvais chiffres ; il fallait 26 740 - 25 760) ;
- remplacer « chiffres non écrits » par « chiffres ni écrits ni recalculables » dans la règle d'abstention :
  3 réponses justes récupérées (val1-036, val1-043, val2-035), mais 7 fausses laissées passer : -2 sur v1, -3 sur v2.
  Le filtre « chiffre non écrit » attrape surtout des réponses fausses, calcul ou pas.
"""
from __future__ import annotations

import re

from groundtruth.normalize import _close, extract_numbers


def _skip(x: float) -> bool:                           # années et petits entiers : trop fréquents (cf. answer.py)
    return float(x).is_integer() and (1900 <= x <= 2100 or abs(x) <= 10)


def _in(x: float, nums: list[tuple[float, float]]) -> bool:
    return any(_close(x, r) or _close(x, s) for r, s in nums)


def _round_close(x: float, y: float) -> bool:
    """x (écrit par le modèle, arrondi) correspond-il au résultat exact y ? Tolérance : l'arrondi d'affichage
    (0,5 sur l'unité du dernier chiffre écrit), ou 0,5 % relatif."""
    if _close(x, y):
        return True
    s = f"{abs(x):.6f}".rstrip("0").rstrip(".")
    dec = len(s.split(".")[1]) if "." in s else 0
    return abs(abs(x) - abs(y)) <= 0.5 * 10 ** (-dec) + 1e-9


def _results(a: float, b: float) -> list[float]:
    out = [a + b, abs(a - b)]
    if b:
        out += [a / b, a / b * 100, (a - b) / b * 100, (b - a) / b * 100]
    if a:
        out += [b / a * 100, (b - a) / a * 100]
    return out


def derived(answer: str, ctx_text: str, question: str = "") -> int:
    from rag2.answer import numbers
    src = extract_numbers(ctx_text + " " + question)
    vals = [(r, s) for r, s in numbers(answer)]
    ok = [r for r, s in vals if _in(r, src) or _in(s, src)]           # opérandes : chiffres de la réponse écrits
    ok_s = ok + [s for r, s in vals if (_in(r, src) or _in(s, src)) and s != r]
    n = 0
    for r, s in vals:
        if _skip(r) or _in(r, src) or _in(s, src):
            continue
        if any(_round_close(r, y) or _round_close(s, y)
               for i, a in enumerate(ok_s) for b in ok_s[i + 1:] for y in _results(a, b)):
            continue
        n += 1
    return n


_EQ = re.compile(r"(\d[\d\s  ]*(?:[,.]\d+)?)\s*([-+−–×x*/])\s*(\d[\d\s  ]*(?:[,.]\d+)?)\s*=\s*(-?\d[\d\s  ]*(?:[,.]\d+)?)")


def _f(t: str) -> float:
    return float(re.sub(r"[\s  ]", "", t).replace(",", "."))


def bad_equations(answer: str) -> int:
    n = 0
    for a, op, b, c in _EQ.findall(answer):
        try:
            a, b, c = _f(a), _f(b), _f(c)
        except ValueError:
            continue
        y = {"+": a + b, "-": a - b, "−": a - b, "–": a - b, "*": a * b, "×": a * b, "x": a * b,
             "/": a / b if b else None}[op]
        if y is not None and not _round_close(c, y):
            n += 1
    return n
