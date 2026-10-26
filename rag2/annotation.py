"""Prépare l'annotation manuelle de 30 pages difficiles (exigence du palier 2).

    python -m rag2.annotation --index-dir index2 --out annotation/annoter.html

Sélection : des pages de preuve de validation_v1 de types scan, tableau et graphique, complétées par
des pages du corpus riches en tableaux ou en images. Pour chaque page, la page rendue en image, le texte
de l'approche A (texte brut PyMuPDF) et celui de l'approche B (texte + OCR, retenue). Tu notes chaque
approche de 0 à 2 sur trois critères, puis « Exporter CSV » enregistre annotation/annotations.csv.
"""
import argparse
import base64
import html
import json
import random
from pathlib import Path

import pymupdf

from groundtruth.io import load_questions

CRITERIA = [("complet", "tout le texte est là"), ("chiffres", "chiffres et unités corrects"),
            ("ordre", "ordre de lecture / tableaux lisibles")]


def select_pages(pages: list[dict], n: int = 30, seed: int = 0) -> list[tuple[str, int, str]]:
    by_key = {(p["doc_id"], p["page"]): p for p in pages}
    picked, seen = [], set()

    def add(d, p, cat):
        if (d, p) in by_key and (d, p) not in seen:
            seen.add((d, p))
            picked.append((d, p, cat))

    quotas = {"scan": 8, "tableau": 8, "graphique": 8}
    for q in load_questions("data/validation_v1.json"):
        if q["answerable"] and q["type"] in quotas:
            for g in q["evidence"]:
                for e in g:
                    if quotas[q["type"]] > 0 and (e["doc_id"], e["page"]) not in seen and (e["doc_id"], e["page"]) in by_key:
                        add(e["doc_id"], e["page"], q["type"])
                        quotas[q["type"]] -= 1
    # complément : pages avec un graphique OCRisé ou beaucoup de chiffres
    rng = random.Random(seed)
    rest = [p for p in pages if (p["doc_id"], p["page"]) not in seen and
            any(b.startswith("[Texte du graphique]") for b in p["blocks"])]
    rng.shuffle(rest)
    for p in rest:
        if len(picked) >= n:
            break
        add(p["doc_id"], p["page"], "image de graphique")
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="index2")
    ap.add_argument("--pdf-dir", default="corpus/pdf")
    ap.add_argument("--out", default="annotation/annoter.html")
    args = ap.parse_args()

    pages = json.loads((Path(args.index_dir) / "pages.json").read_text(encoding="utf-8"))
    by_key = {(p["doc_id"], p["page"]): p for p in pages}
    sel = select_pages(pages)

    cards = []
    for i, (d, pno, cat) in enumerate(sel, 1):
        with pymupdf.open(Path(args.pdf_dir) / f"{d}.pdf") as doc:
            page = doc[pno - 1]
            img = base64.b64encode(page.get_pixmap(dpi=90).tobytes("jpeg", jpg_quality=70)).decode()
            raw = page.get_text()
        final = by_key[(d, pno)]["text"]
        inputs = "".join(
            f'<tr><td>{html.escape(lab)}</td>'
            + "".join(f'<td><select data-k="{i}|{ap_}|{key}"><option></option><option>0</option><option>1</option>'
                      f'<option>2</option></select></td>' for ap_ in ("A", "B"))
            + "</tr>" for key, lab in CRITERIA)
        cards.append(f"""
<section data-id="{i}" data-doc="{d}" data-page="{pno}" data-cat="{cat}">
<h2>{i}. {d} — page {pno} <small>({cat})</small></h2>
<div class="row">
 <img src="data:image/jpeg;base64,{img}">
 <div class="col"><h3>A : texte brut</h3><pre>{html.escape(raw[:6000]) or '<em>(vide)</em>'}</pre></div>
 <div class="col"><h3>B : texte + OCR</h3><pre>{html.escape(final[:6000]) or '<em>(vide)</em>'}</pre></div>
</div>
<table><tr><th>Critère (0 = mauvais, 2 = bon)</th><th>A</th><th>B</th></tr>{inputs}</table>
<textarea data-k="{i}|note" placeholder="Commentaire (optionnel)"></textarea>
</section>""")

    page_html = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Annotation parsing</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--card:#fff;--line:#ddd}}
@media (prefers-color-scheme: dark){{:root{{--bg:#1b1b1a;--fg:#ecebe6;--card:#262624;--line:#444}}}}
body{{font-family:system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:16px;max-width:1500px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0}}
.row{{display:flex;gap:12px;flex-wrap:wrap}} img{{max-width:420px;width:100%;border:1px solid var(--line)}}
.col{{flex:1;min-width:280px}} pre{{white-space:pre-wrap;font-size:12px;max-height:520px;overflow:auto;
background:var(--bg);padding:8px;border-radius:6px}} table{{margin-top:8px;border-collapse:collapse}}
td,th{{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left}} textarea{{width:100%;margin-top:8px}}
#bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}}
button{{font-size:15px;padding:6px 14px}}
</style></head><body>
<div id="bar"><b>Annotation de {len(sel)} pages</b> — note A et B de 0 à 2 sur chaque critère, puis
<button onclick="exportCsv()">Exporter CSV</button> <span id="status"></span></div>
{''.join(cards)}
<script>
function exportCsv(){{
  const rows=[["id","doc_id","page","categorie","A_complet","A_chiffres","A_ordre","B_complet","B_chiffres","B_ordre","note"]];
  let missing=0;
  document.querySelectorAll("section").forEach(s=>{{
    const i=s.dataset.id, g=k=>{{const e=document.querySelector(`[data-k="${{i}}|${{k}}"]`);return e?e.value:""}};
    const r=[i,s.dataset.doc,s.dataset.page,s.dataset.cat];
    for(const a of ["A","B"]) for(const c of ["complet","chiffres","ordre"]){{const v=g(a+"|"+c); if(v==="")missing++; r.push(v)}}
    r.push(g("note").replace(/[\\n;"]/g," ")); rows.push(r);
  }});
  const csv=rows.map(r=>r.map(x=>'"'+String(x)+'"').join(";")).join("\\n");
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob(["\\ufeff"+csv],{{type:"text/csv"}}));
  a.download="annotations.csv"; a.click();
  document.getElementById("status").textContent = missing ? missing+" cases vides" : "complet";
}}
</script></body></html>"""
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page_html, encoding="utf-8")
    print(f"{len(sel)} pages -> {args.out}")


if __name__ == "__main__":
    main()
