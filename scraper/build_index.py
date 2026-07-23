"""Phase 3: embed chunk records and build a FAISS index.

The canonical chunk file lives at data/processed/chunks.json. A legacy
chunks.json in the scraper root is still accepted as a fallback so the script
remains usable while the project layout is being cleaned up.
"""

from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
LEGACY_CHUNKS_FILE = BASE_DIR / "chunks.json"
INDEX_FILE = INDEX_DIR / "faiss.index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    from sentence_transformers import SentenceTransformer
    import faiss

    source_file = CHUNKS_FILE if CHUNKS_FILE.exists() else LEGACY_CHUNKS_FILE
    chunks = json.loads(source_file.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks from {source_file}")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # so inner product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    print(f"Embedded {len(chunks)} chunks -> {dim}-dim vectors")
    print(f"Index saved -> {INDEX_FILE}")
    print("chunks.json unchanged -- index position i corresponds to chunks[i]")


if __name__ == "__main__":
    main()