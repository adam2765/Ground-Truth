"""Découpe des graphiques d'une page PDF (pour un modèle de vision spécialisé, qui attend un graphique seul).

Un graphique est soit une image intégrée assez grande, soit un amas de tracés vectoriels (barres, courbes, axes).
On regroupe les tracés proches (écart < GAP points) en amas, on garde les amas assez grands, puis on élargit le cadre
pour inclure les étiquettes, la légende et le titre (« Graphique n° 7 : ... ») écrits autour.
"""
import pymupdf

GAP = 12              # deux tracés à moins de 12 points l'un de l'autre appartiennent au même graphique
MIN_FRAC = 0.03       # amas d'au moins 3 % de la surface de la page
MARGIN = 28           # marge autour de l'amas pour les étiquettes d'axes et la légende
TITLE_UP = 45         # hauteur cherchée au-dessus pour le titre


def _merge(rects: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    rects = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        for r in rects:
            grown = pymupdf.Rect(r.x0 - GAP, r.y0 - GAP, r.x1 + GAP, r.y1 + GAP)
            for o in out:
                if grown.intersects(o):
                    o |= r
                    changed = True
                    break
            else:
                out.append(pymupdf.Rect(r))
        rects = out
    return rects


def chart_regions(page) -> list[pymupdf.Rect]:
    pr = page.rect
    area = pr.width * pr.height
    cand = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & pr
        if 0.05 * area <= r.width * r.height <= 0.9 * area:
            cand.append(r)
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"]) & pr
        if r.is_empty and not (r.width or r.height):
            continue
        if r.width > 0.95 * pr.width or r.height > 0.95 * pr.height:   # cadres de page, filets pleine largeur
            continue
        cand.append(r)
    out = []
    for r in _merge(cand):
        if r.width * r.height < MIN_FRAC * area or r.width < 80 or r.height < 60:
            continue
        box = pymupdf.Rect(r.x0 - MARGIN, r.y0 - TITLE_UP, r.x1 + MARGIN, r.y1 + MARGIN) & pr
        out.append(box)
    return _merge(out)


def crop_png(page, rect, dpi: int = 150) -> bytes:
    return page.get_pixmap(dpi=dpi, clip=rect).tobytes("png")
