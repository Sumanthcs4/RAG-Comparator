import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Paths
CHUNK_PATH = "../data/chunks"
FAISS_INDEX_PATH = "../vectorstore/faiss_index.index"
METADATA_PATH = "../vectorstore/faiss_metadata.json"

# Create vectorstore directory
os.makedirs("../vectorstore", exist_ok=True)

# Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Dimension of embeddings
VECTOR_DIM = 384
index = faiss.IndexFlatIP(VECTOR_DIM)

# Store metadata for retrieval
metadata = []

# Loop through all chunk files
chunk_files = sorted(os.listdir(CHUNK_PATH))
total_chunks = 0

for idx, file in enumerate(chunk_files, 1):
    file_path = os.path.join(CHUNK_PATH, file)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        if not chunks:
            print(f"WARNING: [{idx}/{len(chunk_files)}] Skipping empty file: {file}")
            continue

        for chunk in chunks:
            text = chunk["text"]
            vector = model.encode(text)
            vector = np.array(vector, dtype=np.float32).reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(vector)
            
            index.add(vector)

            metadata.append({
                "paper_id": chunk["paper_id"],
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "text": text
            })
            total_chunks += 1

        print(f"SUCCESS: [{idx}/{len(chunk_files)}] Embedded -> {file} ({len(chunks)} chunks)")
    
    except Exception as e:
        print(f"ERROR: [{idx}/{len(chunk_files)}] Error processing {file}: {e}")
        continue

# Save FAISS index
faiss.write_index(index, FAISS_INDEX_PATH)

# Save metadata
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"\nDone! Embedded {total_chunks} chunks from {len(chunk_files)} papers")
print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"Metadata saved to: {METADATA_PATH}")