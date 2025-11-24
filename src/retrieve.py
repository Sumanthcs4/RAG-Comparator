import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent  # Goes up from src/ to project root
FAISS_INDEX_PATH = PROJECT_ROOT / "vectorstore" / "faiss_index.index"
METADATA_PATH = PROJECT_ROOT / "vectorstore" / "faiss_metadata.json"

# Load model + FAISS + metadata
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

metadata = {i: item for i, item in enumerate(metadata)}


def search(query, top_k=5):
    """
    Search for top-k similar chunks
    
    Args:
        query: str - search query
        top_k: int - number of results to return
    
    Returns:
        list of dicts with chunk info
    """
    # Convert query to embedding
    query_vec = model.encode(query)
    query_vec = np.array([query_vec]).astype("float32")
    faiss.normalize_L2(query_vec)
    
    # FAISS search
    distances, indices = faiss_index.search(query_vec, top_k)
    
    # Return chunks with metadata
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        results.append({
            "chunk_id": idx,
            "distance": float(dist),
            "text": metadata[idx]["text"],
            "filename": metadata[idx]["filename"],
            "paper_id": metadata[idx]["paper_id"]
        })
    return results


