
from pathlib import Path
import json
import sys
import os

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
LEGACY_CHUNKS_FILE = BASE_DIR / "chunks.json"
INDEX_FILE = INDEX_DIR / "faiss.index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5  # how many chunks to retrieve


def load_chunks():
    source_file = CHUNKS_FILE if CHUNKS_FILE.exists() else LEGACY_CHUNKS_FILE
    return json.loads(source_file.read_text(encoding="utf-8"))


def retrieve(question, model, index, chunks, k=TOP_K):
    """Embed the question, search FAISS, return the top-k matching chunk dicts."""
    q_vec = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    distances, indices = index.search(q_vec, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # FAISS pads with -1 if fewer than k results exist
        results.append({
            "rank": rank + 1,
            "score": float(distances[0][rank]),
            "text": chunks[idx]["text"],
            "chunk_index": int(idx),
        })
    return results


def build_prompt(question, retrieved_chunks):
    """Assemble the retrieved context + question into a single prompt for the LLM."""
    context_block = "\n\n".join(
        f"[Source {r['rank']}]\n{r['text']}" for r in retrieved_chunks
    )
    return f"""Answer the question using ONLY the information in the sources below.
If the sources don't contain the answer, say you don't know -- do not guess.

Sources:
{context_block}

Question: {question}

Answer:"""


GROQ_MODEL = "llama-3.3-70b-versatile"  # free tier, strong quality


def generate_answer(prompt):
    """Send the prompt to Groq's free API and return the answer text."""
    from groq import Groq

    client = Groq()  # reads GROQ_API_KEY from env automatically
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    question = sys.argv[1]

    from sentence_transformers import SentenceTransformer
    import faiss

    print("Loading model and index...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(str(INDEX_FILE))
    chunks = load_chunks()

    print(f"Retrieving top {TOP_K} chunks for: {question!r}\n")
    retrieved = retrieve(question, model, index, chunks)

    for r in retrieved:
        preview = r["text"][:100].replace("\n", " ")
        print(f"  [{r['rank']}] score={r['score']:.3f} -> {preview}...")

    prompt = build_prompt(question, retrieved)

    if not os.environ.get("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set -- skipping answer generation.")
        print("Get a free key: https://console.groq.com/keys")
        print("Then: export GROQ_API_KEY='your-key-here'")
        return

    print(f"\nAsking Groq ({GROQ_MODEL})...\n")
    try:
        answer = generate_answer(prompt)
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
