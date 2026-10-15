from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
INDEX_FILE = VECTOR_DB_DIR / "indian_law.index"
SECTIONS_FILE = VECTOR_DB_DIR / "indian_law_sections.txt"

@lru_cache(maxsize=1)
def _load_rag():
    """Load embedding model and FAISS only when a law search is actually requested."""
    import faiss
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    except TypeError:
        # Older sentence-transformers versions may not accept local_files_only.
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(str(INDEX_FILE))
    sections_text = SECTIONS_FILE.read_text(encoding="utf-8")
    law_sections = sections_text.split("\n\n---SECTION---\n\n")
    return model, index, law_sections


def search_law(query, top_k=3):
    model, index, law_sections = _load_rag()
    query_embedding = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    scores, indexes = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indexes[0]):
        if idx < 0 or idx >= len(law_sections):
            continue
        results.append({"section": law_sections[idx], "score": float(score)})
    return results


def get_relevant_law(query, top_k=3):
    results = search_law(query, top_k=top_k)
    if not results:
        return "No relevant Indian law reference found."
    return "\n\n".join(
        f"Similarity: {r['score']:.3f}\n{r['section']}" for r in results
    )
