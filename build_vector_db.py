from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LAW_FILE = BASE_DIR / "data" / "laws" / "indian_contract_act.txt"

VECTOR_DB_DIR = BASE_DIR / "vector_db"

VECTOR_DB_DIR.mkdir(exist_ok=True)


# =========================================================
# READ LAW FILE
# =========================================================

print("Reading Indian Contract Act data...")

text = LAW_FILE.read_text(encoding="utf-8")


if not text.strip():
    raise ValueError("The law file is empty.")


# =========================================================
# SPLIT INTO CHUNKS
# =========================================================

sections = []

current_section = ""

for line in text.splitlines():

    line = line.strip()

    if not line:
        continue

    if line.startswith("SECTION "):

        if current_section:
            sections.append(current_section.strip())

        current_section = line

    else:

        current_section += " " + line


if current_section:
    sections.append(current_section.strip())


print(f"Found {len(sections)} law sections.")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("Loading Sentence Transformer model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print("Creating embeddings...")

embeddings = model.encode(
    sections,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


# =========================================================
# CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# =========================================================
# SAVE FAISS DATABASE
# =========================================================

index_path = VECTOR_DB_DIR / "indian_law.index"

faiss.write_index(
    index,
    str(index_path)
)


# =========================================================
# SAVE SECTION TEXT
# =========================================================

sections_path = VECTOR_DB_DIR / "indian_law_sections.txt"

sections_path.write_text(
    "\n\n---SECTION---\n\n".join(sections),
    encoding="utf-8"
)


# =========================================================
# COMPLETE
# =========================================================

print()
print("==============================================")
print("INDIAN LAW VECTOR DATABASE CREATED")
print("==============================================")
print(f"Sections: {len(sections)}")
print(f"FAISS index: {index_path}")
print(f"Law sections: {sections_path}")
print("==============================================")