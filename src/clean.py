import os
import json
import re

RAW_PATH = "../data/processed"
OUT_PATH = "../data/cleaned"

os.makedirs(OUT_PATH, exist_ok=True)

def clean_text(text):
    # 1. Remove hyphen-newline breaks: "transfor-\nmers" -> "transformers"
    text = re.sub(r'-\s*\n\s*', '', text)

    # 2. Replace multiple newlines with double newline (preserve paragraphs)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 3. Single newlines become spaces (within paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 4. Remove page headers/footers
    text = re.sub(r'arXiv:\S+', '', text)
    text = re.sub(r'Page\s*\d+', '', text)
    text = re.sub(r'Published as.*?Conference.*?\d{4}', '', text, flags=re.IGNORECASE)
    
    # ADDED: Remove common footer patterns
    text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)  # Remove page numbers at line end

    # 5. Replace fancy unicode
    unicode_map = {
        'ﬁ': 'fi',
        'ﬂ': 'fl',
        'ﬀ': 'ff',
        '—': '-',
        '–': '-',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '·': '',
        '•': '',
        '…': '...',  # ADDED: Ellipsis
    }
    for k, v in unicode_map.items():
        text = text.replace(k, v)

    # 6. Remove extra spaces
    text = re.sub(r' +', ' ', text)
    
    # 7. Clean up paragraph spacing
    text = re.sub(r'\n ', '\n', text)
    text = re.sub(r' \n', '\n', text)

    return text.strip()

files = sorted(os.listdir(RAW_PATH))

if not files:  # ADDED: Check for empty directory
    print("ERROR: No files found in", RAW_PATH)
    exit(1)

processed_count = 0  # ADDED: Track successful processing

for idx, file in enumerate(files, 1):
    if not file.endswith('.json'):  # ADDED: Skip non-JSON files
        continue
        
    try:
        with open(os.path.join(RAW_PATH, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned = clean_text(data["text"])

        out_data = {
            "paper_id": data.get("paper_id", f"paper_{idx:03d}"),
            "filename": data.get("filename", file.replace(".json", ".pdf")),
            "cleaned_text": cleaned
        }

        out_path = os.path.join(OUT_PATH, file)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        print(f"[{idx}/{len(files)}] Cleaned -> {data.get('filename', file)} ({len(cleaned)} chars)")
        processed_count += 1  # ADDED
    
    except Exception as e:
        print(f"ERROR: [{idx}/{len(files)}] Failed to process {file}: {e}")
        continue

print(f"\nDone! Successfully cleaned {processed_count}/{len(files)} papers")  # CHANGED