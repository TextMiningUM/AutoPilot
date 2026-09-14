"""Quick RAG retrieval smoke test — ASCII output only."""
import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

W = Path(__file__).resolve().parent
CACHE  = W / "Data" / "VHF" / "VHF_Agents_Training"
chunks = json.loads((CACHE / "vhf_rag_chunks.json").read_text(encoding="utf-8"))
embs   = np.load(CACHE / "vhf_rag_embeddings.npy")
ids    = json.loads((CACHE / "vhf_rag_chunk_ids.json").read_text(encoding="utf-8"))
cbi    = {c["chunk_id"]: c for c in chunks}
print(f"Loaded: {len(chunks)} chunks, embeddings {embs.shape}")

model  = SentenceTransformer("all-MiniLM-L6-v2")

QUERIES = [
    "What is VHF Channel 70 used for?",
    "How do I send a MAYDAY call?",
    "What is the phonetic word for the letter M?",
    "What is the difference between MAYDAY and PAN-PAN?",
    "Which VHF channel is the international distress channel?",
]

for q in QUERIES:
    print()
    print("=" * 100)
    print("Q:", q)
    print("=" * 100)
    qe = model.encode([q], normalize_embeddings=True)[0]
    scores = embs @ qe
    top = np.argsort(-scores)[:5]
    for r, i in enumerate(top, 1):
        c = cbi[ids[i]]
        s = float(scores[i])
        preview = c["text"].replace("\n", " ")[:140]
        print(f"  {r}. [{s:.3f}] {c['source_file']}  ->  {c['chapter_title'][:40]!r}")
        print(f"     types={c['types']}  topics={c['topics']}")
        print(f"     text: {preview}...")
