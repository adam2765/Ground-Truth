"""Transcription des pages numérisées par un petit modèle de vision, à l'indexation (parsing : autorisé, confirmé
par le mentor : « le parsing est libre en open weights »). Le modèle transcrit, il ne répond à aucune question.

Pourquoi : Tesseract lit bien la prose des scans, mais abîme les tableaux (insee-ip2119 p.2 : la rangée
« Femme - Femme | 29,5 | 30,4 | 59,9 | /// | 40,1 » devient « 295 30,4 s9,s dit 40,1 »). Les 9 questions « scan »
de la validation v1 rapportent +0,11 en moyenne. Les tableaux transcrits en Markdown sont réécrits au format des
tableaux structurés (rag2/tables.py) : « rangée | en-tête : valeur ».

Garde-fous (la finale ne doit jamais casser à cause de cette étape) :
  - modèle absent du cache, erreur, plus de mémoire GPU : on garde le texte de Tesseract ;
  - budget de temps (SCAN_VLM_BUDGET, 60 min par défaut) : au-delà, les pages restantes gardent Tesseract ;
  - transcription nettement plus courte que l'OCR (page tronquée, refus) : on garde Tesseract.

    RAG_SCAN_VLM=Qwen/Qwen3-VL-4B-Instruct bash run.sh index corpus/pdf /tmp/index     (actif dans run.sh)
    python -m rag2.scan_vlm --index-dir /tmp/index --pdf-dir corpus/pdf --log runs/scan_v1.jsonl
"""
import argparse
import io
import json
import os
import re
import time
from pathlib import Path

PROMPT = (
    "Transcris fidèlement tout le texte de cette page d'un rapport public français, dans l'ordre de lecture "
    "(colonne par colonne). Recopie les chiffres exactement, avec leurs virgules. Écris chaque tableau en Markdown "
    "(| ... |), avec sa ligne d'en-têtes. N'ajoute aucun commentaire."
)
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def md_tables_to_rows(text: str) -> str:
    """Tableau Markdown -> une ligne par rangée, chaque valeur précédée de son en-tête de colonne."""
    out, table = [], []

    def flush():
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in table]
        rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) or not c for c in r)]
        if rows:
            head, body = rows[0], rows[1:]
            if not body:                                 # une seule ligne : on la garde telle quelle
                out.append(" | ".join(c for c in head if c))
            for r in body:
                cells = [r[0]] + [f"{head[j]} : {c}" if j < len(head) and head[j] else c
                                  for j, c in enumerate(r) if j > 0 and c]
                out.append(" | ".join(c for c in cells if c) + "\n")   # une rangée = un paragraphe
        table.clear()

    for ln in text.splitlines():
        if ln.strip().startswith("|"):
            table.append(ln)
        else:
            if table:
                flush()
            out.append(ln)
    if table:
        flush()
    return "\n".join(out)


def _paragraphs(text: str) -> list[str]:
    from rag2.parse import _clean
    text = md_tables_to_rows(text)
    paras = [_clean(p) for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]


def transcribe(jobs: list[tuple[str, str, int]], model_id: str | None = None, budget_s: float | None = None,
               batch: int = 4, max_new: int = 1536, dpi: int = 150) -> dict[str, str]:
    """jobs : [(clé, chemin du PDF, n° de page 0-based)] -> {clé: transcription}. Ne lève jamais d'exception."""
    model_id = model_id or os.environ.get("RAG_SCAN_VLM") or DEFAULT_MODEL
    budget_s = budget_s if budget_s is not None else float(os.environ.get("SCAN_VLM_BUDGET", 3600))
    res: dict[str, str] = {}
    if not jobs:
        return res
    t0 = time.time()
    try:
        import pymupdf
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if not torch.cuda.is_available():
            print("transcription des scans : pas de GPU, on garde Tesseract", flush=True)
            return res
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16,
                                                            device_map={"": 0}).eval()
        proc = AutoProcessor.from_pretrained(model_id)
        proc.tokenizer.padding_side = "left"
        prompt = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text",
                                           "text": PROMPT}]}], add_generation_prompt=True, tokenize=False)
        print(f"transcription des scans : {model_id} chargé en {time.time() - t0:.0f} s, {len(jobs)} pages",
              flush=True)
        for b in range(0, len(jobs), batch):
            if time.time() - t0 > budget_s:
                print(f"  budget de {budget_s / 60:.0f} min atteint : {len(jobs) - b} pages gardent Tesseract",
                      flush=True)
                break
            part = jobs[b:b + batch]
            imgs = []
            for _, path, pno in part:
                with pymupdf.open(path) as doc:
                    png = doc[pno].get_pixmap(dpi=dpi).tobytes("png")
                imgs.append(Image.open(io.BytesIO(png)).convert("RGB"))
            inputs = proc(text=[prompt] * len(part), images=imgs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            texts = proc.batch_decode(o[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for (k, _, _), t in zip(part, texts):
                res[k] = t.strip()
            print(f"  {min(b + batch, len(jobs))}/{len(jobs)} pages ({time.time() - t0:.0f} s)", flush=True)
        del model
        torch.cuda.empty_cache()
    except Exception as e:                              # jamais bloquant : on garde Tesseract
        print(f"transcription des scans impossible ({type(e).__name__}: {str(e)[:150]}) : on garde Tesseract",
              flush=True)
    print(f"transcription des scans : {len(res)} pages en {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


def apply(pages: list[dict], pdf_dir: str, log: str | None = None, **kw) -> int:
    """Remplace le texte Tesseract des pages numérisées par la transcription (si elle passe le contrôle)."""
    idx = [i for i, p in enumerate(pages) if p.get("source") == "ocr"]
    jobs = [(str(i), str(Path(pdf_dir) / f"{pages[i]['doc_id']}.pdf"), pages[i]["page"] - 1) for i in idx]
    jobs = [j for j in jobs if Path(j[1]).exists()]
    res = transcribe(jobs, **kw)
    n = 0
    logf = open(log, "a", encoding="utf-8") if log else None
    for k, t in res.items():
        rec = pages[int(k)]
        paras = _paragraphs(t)
        new = "\n".join(paras)
        ok = len(new) >= 0.6 * len(rec.get("text", ""))     # page tronquée ou refus : on garde Tesseract
        if logf:
            logf.write(json.dumps({"doc_id": rec["doc_id"], "page": rec["page"], "ok": ok, "vlm": t,
                                   "tesseract": rec.get("text", "")}, ensure_ascii=False) + "\n")
        if ok:
            rec["blocks"], rec["text"], rec["source"] = paras, new, "vlm"
            n += 1
    if logf:
        logf.close()
    print(f"transcription des scans : {n} pages remplacées sur {len(idx)}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--log")
    a = ap.parse_args()
    p = Path(a.index_dir) / "pages.json"
    pages = json.loads(p.read_text(encoding="utf-8"))
    apply(pages, a.pdf_dir, a.log)
    p.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
