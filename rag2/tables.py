"""Tableaux structurés : chaque valeur écrite avec l'en-tête de sa colonne.

Problème (run 4b, validation) : sur 41 questions avec réponse où l'on s'abstient, 36 ont la bonne page dans
les 5 extraits ; 14 sont des questions sur tableau. Le texte extrait donne une rangée comme
« Europe du Sud | 11 | 11 | 16 | 11 | 7 » : le modèle doit compter les colonnes pour trouver 2014, et se
trompe (11 % au lieu de 16 %).

Ici on écrit la rangée « Europe du Sud | 2006 : 11 | 2010 : 11 | 2014 : 16 | 2018 : 11 | 2023 : 7 ».

Méthode : PyMuPDF (find_tables) détecte le tableau et ses colonnes, mais le texte de ses cellules est souvent
abîmé (espaces perdus, « 12,5 » lu « 125\\n, »). On ne garde donc que la géométrie (cadre du tableau,
bornes des colonnes, hauteur de l'en-tête) et on relit les mots eux-mêmes (get_text("words"), propres),
qu'on range par ligne (même hauteur) puis par colonne (position horizontale).

    python -m rag2.tables --pdf-dir corpus/pdf --out cache/tables_v1.json.gz
"""
import argparse
import gzip
import json
import re
from pathlib import Path

import pymupdf

MAX_CHARS = 1500                     # au-delà, on ne garde que les rangées proches de la question
_NUM = re.compile(r"^[-−–+]?[\d][\d\s.,]*%?\*?$")


def _columns(t) -> list[tuple[float, float]]:
    """Bornes horizontales des colonnes : la rangée du tableau qui a le plus de cellules."""
    best = []
    for row in [t.header] + list(t.rows):
        cells = [c for c in row.cells if c is not None]
        if len(cells) > len(best):
            best = cells
    return sorted((c[0], c[2]) for c in best)


def _col_of(x: float, cols: list[tuple[float, float]]) -> int:
    for j, (x0, x1) in enumerate(cols):
        if x0 - 1 <= x <= x1 + 1:
            return j
    return min(range(len(cols)), key=lambda j: abs((cols[j][0] + cols[j][1]) / 2 - x))


def _lines(words: list) -> list[list]:
    """Regroupe les mots par ligne (centres verticaux à moins de 3 points)."""
    lines = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        yc = (w[1] + w[3]) / 2
        if lines and abs(lines[-1][0] - yc) < 3:
            lines[-1][1].append(w)
        else:
            lines.append([yc, [w]])
    return [sorted(ws, key=lambda w: w[0]) for _, ws in lines]


def table_text(page, t) -> str | None:
    cols = _columns(t)
    if len(cols) < 2:
        return None
    box = pymupdf.Rect(t.bbox)
    words = [w for w in page.get_text("words")
             if box.x0 - 1 <= (w[0] + w[2]) / 2 <= box.x1 + 1 and box.y0 - 1 <= (w[1] + w[3]) / 2 <= box.y1 + 1]
    if not words:
        return None
    head_y = t.header.bbox[3] if not t.header.external else box.y0
    yc = lambda w: (w[1] + w[3]) / 2

    def cells_of(line):
        cells = [[] for _ in cols]
        for w in line:
            cells[_col_of((w[0] + w[2]) / 2, cols)].append(w[4])
        return [" ".join(c) for c in cells]

    head = [[] for _ in cols]
    for line in _lines([w for w in words if yc(w) < head_y]):
        for j, c in enumerate(cells_of(line)):
            if c:
                head[j].append(c)
    head = [" ".join(h) for h in head]

    # Rangées : celles de find_tables. Une rangée détectée peut contenir plusieurs lignes visuelles :
    #  - chaque ligne a son libellé et ses valeurs (rangées fusionnées par erreur) -> une rangée par ligne ;
    #  - sinon (libellé sur deux lignes, centré verticalement à côté des valeurs) -> une seule rangée.
    rows = sorted({(r.bbox[1], r.bbox[3]) for r in t.rows if r.bbox[3] > head_y + 1})
    groups = [[] for _ in rows]
    for w in words:
        if yc(w) < head_y or not rows:
            continue
        k = next((i for i, (y0, y1) in enumerate(rows) if y0 - 1 <= yc(w) <= y1 + 1), None)
        if k is None:
            k = min(range(len(rows)), key=lambda i: abs((rows[i][0] + rows[i][1]) / 2 - yc(w)))
        groups[k].append(w)
    body = []
    for g in groups:
        lines = [cells_of(l) for l in _lines(g)]
        if not lines:
            continue
        vals = [c for c in lines if any(c[1:])]
        if len(vals) <= 1 or all(c[0] for c in vals):
            body.extend(lines)                         # (lignes de libellé seul : rattachées plus bas)
        else:
            body.append([" ".join(c[j] for c in lines if c[j]) for j in range(len(cols))])
    # lignes d'en-tête restées dans le corps (pas de nombre hors 1re colonne, avant la 1re rangée chiffrée)
    while body and not any(_NUM.match(c) for c in body[0][1:] if c):
        if any(body[0][1:]):
            head = [(h + " " + c).strip() for h, c in zip(head, body[0])]
            body.pop(0)
        else:
            break
    rows, label, pending = [], "", ""
    n_num = 0
    for cells in body:
        if not cells[0] and len(cells) > 2 and cells[1] and not _NUM.match(cells[1]):
            cells = [cells[1], ""] + cells[2:]         # libellé décalé dans la 2e colonne
        vals = [[j, c] for j, c in enumerate(cells) if j > 0 and c]
        if not vals:                                   # libellé seul (sur plusieurs lignes) : on le garde
            pending = (pending + " " + cells[0]).strip()
            continue
        if cells[0]:
            label = (pending + " " + cells[0]).strip()
        elif pending:
            label = pending
        pending = ""
        n_num += any(_NUM.match(v) for _, v in vals)
        rows.append([label or "(rangée)", vals])
    if n_num < 2:                                      # pas un tableau de chiffres (cadre de texte, encadré)
        return None
    return {"first": head[0], "heads": [h or f"colonne {j + 1}" for j, h in enumerate(head)], "rows": rows}


def render(t: dict, question: str = "", budget: int = MAX_CHARS, by_column: bool = False) -> str:
    """Texte d'un tableau pour le prompt. En-têtes courts : écrits devant chaque valeur
    (« Europe du Sud | 2014 : 16 »). En-têtes longs : numérotés une fois en légende, puis « (6) 101 ».
    Si le tableau dépasse le budget, on garde les rangées qui partagent le plus de mots avec la question
    (dans l'ordre du tableau)."""
    heads = t["heads"]
    used = sorted({j for _, vals in t["rows"] for j, _ in vals})
    inline = sum(len(heads[j]) for j in used) / max(len(used), 1) <= 30
    head_lines = [f"(première colonne : {t['first']})"] if t["first"] else []
    if inline:
        fmt = lambda vals: " | ".join(f"{heads[j]} : {v}" for j, v in vals)
    else:
        num = {j: k + 1 for k, j in enumerate(used)}
        head_lines.append("Colonnes : " + " ; ".join(f"({num[j]}) {heads[j]}" for j in used))
        fmt = lambda vals: " | ".join(f"({num[j]}) {v}" for j, v in vals)
    lines = [f"{label} | {fmt(vals)}" for label, vals in t["rows"]]
    if by_column:                                      # une ligne par colonne : « 2014 : Afrique 36 ; Europe du Sud 16 »
        keep_rows = _select(t, lines, head_lines, question, budget)
        cols = []
        for j in used:
            cells = [f"{t['rows'][i][0]} {v}" for i in keep_rows for jj, v in t["rows"][i][1] if jj == j]
            if cells:
                cols.append(f"{heads[j]} : " + " ; ".join(cells))
        return "\n".join(([f"(rangées : {t['first']})"] if t["first"] else []) + cols)
    lines = [lines[i] for i in _select(t, lines, head_lines, question, budget)]
    return "\n".join(head_lines + lines)


def _select(t: dict, lines: list[str], head_lines: list[str], question: str, budget: int) -> list[int]:
    """Indices des rangées gardées : toutes si elles tiennent dans le budget, sinon celles qui partagent le
    plus de mots avec la question (dans l'ordre du tableau)."""
    size = lambda ls: sum(len(x) + 1 for x in head_lines + ls)
    if size(lines) <= budget:
        return list(range(len(lines)))
    from rag2.search import tokenize
    q = set(tokenize(question))
    order = sorted(range(len(lines)), key=lambda i: -len(q & set(tokenize(t["rows"][i][0]))))
    keep = set()
    for i in order:
        if size([lines[k] for k in sorted(keep | {i})]) <= budget:
            keep.add(i)
    return sorted(keep)


def page_tables(page) -> list[dict]:
    try:
        tabs = page.find_tables().tables
    except Exception:
        return []
    return [x for x in (table_text(page, t) for t in tabs) if x]


def _doc_tables(pdf: str) -> dict[str, list[dict]]:
    out = {}
    with pymupdf.open(pdf) as doc:
        for pno, page in enumerate(doc):
            if len(page.get_text().strip()) < 50:          # page numérisée : pas de mots à placer
                continue
            tabs = page_tables(page)
            if tabs:
                out[f"{Path(pdf).stem}|{pno + 1}"] = tabs
    return out


def extract_dir(pdf_dir: str, workers: int | None = None) -> dict[str, list[dict]]:
    """{"doc_id|page": [tableaux de la page : {first, heads, rows}]} (texte du prompt : render()). find_tables est lent (~0,3 s par page) :
    un processus par document."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    pdfs = [str(p) for p in sorted(Path(pdf_dir).glob("*.pdf"))]
    out = {}
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count() or 2) as ex:
        for d in ex.map(_doc_tables, pdfs):
            out.update(d)
    return out


def load(path: str | Path) -> dict[str, list[dict]]:
    p = Path(path)
    if not p.exists():
        return {}
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def save(tabs: dict, path: str | Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(tabs, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tabs = extract_dir(a.pdf_dir)
    save(tabs, a.out)
    print(f"{len(tabs)} pages avec tableau -> {a.out}")


if __name__ == "__main__":
    main()
