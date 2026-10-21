"""Étape 2 : chunker naïf, taille fixe en mots avec recouvrement.

On découpe à l'intérieur de chaque page : un chunk = une seule page à citer.
"""


def chunk_pages(pages: list[dict], size: int = 200, overlap: int = 50) -> list[dict]:
    """Découpe chaque page en chunks de `size` mots qui se recouvrent de `overlap` mots."""
    assert 0 <= overlap < size
    chunks = []
    step = size - overlap                      # de combien on avance à chaque chunk
    for p in pages:
        words = p["text"].split()              # coupe sur espaces et \n, enlève les vides
        if not words:                          # page vide (scan) : rien à indexer
            continue
        for start in range(0, len(words), step):
            piece = words[start:start + size]
            chunks.append({
                "chunk_id": f'{p["doc_id"]}_p{p["page"]}_{start}',
                "doc_id": p["doc_id"],
                "page": p["page"],
                "text": " ".join(piece),
            })
            if start + size >= len(words):     # ce chunk atteint la fin de la page : on arrête
                break
    return chunks


if __name__ == "__main__":
    from my_rag.extract import extract_pages

    chunks = chunk_pages(extract_pages("corpus/pdf"))
    print(len(chunks), "chunks")
    print(chunks[0]["chunk_id"], "|", chunks[0]["text"][-120:])
    print(chunks[1]["chunk_id"], "|", chunks[1]["text"][:120])
