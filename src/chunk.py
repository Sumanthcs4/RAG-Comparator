import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# path
RAW_PATH = "../data/cleaned"
OUT_PATH = "../data/chunks"
os.makedirs(OUT_PATH, exist_ok=True)

files = sorted(os.listdir(RAW_PATH))

total_chunks = 0

for idx, file in enumerate(files, 1):
    try:
        with open(os.path.join(RAW_PATH, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data["cleaned_text"]

        # Split into chunks
        chunks = text_splitter.split_text(text)

        # Prepare chunked data with metadata
        chunked_data = []
        for c_idx, chunk in enumerate(chunks, 1):
            chunked_data.append({
                "paper_id": data["paper_id"],
                "filename": data["filename"],
                "chunk_id": c_idx,
                "text": chunk
            })

        # Save each paper's chunks
        out_path = os.path.join(OUT_PATH, f"{data['paper_id']}_chunks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunked_data, f, indent=2)

        total_chunks += len(chunks)
        print(f"SUCCESS: [{idx}/{len(files)}] Chunked -> {data['filename']} ({len(chunks)} chunks)")
    
    except Exception as e:
        print(f"ERROR: [{idx}/{len(files)}] Error processing {file}: {e}")
        continue

print(f"\nAll PDFs chunked! Total chunks: {total_chunks}")