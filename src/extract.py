import os
import json
import fitz  # PyMuPDF

PDF_DIR = "../data/pdfs"
OUTPUT_DIR = "../data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        num_pages = len(doc)  # GET THIS FIRST before closing

        for page in doc:
            text = page.get_text()
            full_text += text + "\n"
        
        doc.close()  # Now it's safe to close
        return full_text, num_pages
    
    except Exception as e:
        print(f"ERROR: Failed to extract {pdf_path}: {e}")
        return None, 0

def process_all_pdfs():
    pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    
    if not pdf_files:
        print("ERROR: No PDF files found in", PDF_DIR)
        return

    success_count = 0

    for idx, pdf_file in enumerate(pdf_files, start=1):
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        print(f"[{idx}/{len(pdf_files)}] Extracting: {pdf_file}")

        text, num_pages = extract_text_from_pdf(pdf_path)
        
        if text is None:
            continue

        json_data = {
            "paper_id": f"paper_{idx:03d}",
            "filename": pdf_file,
            "num_pages": num_pages,
            "text": text
        }

        output_file = os.path.join(OUTPUT_DIR, f"paper_{idx:03d}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"    SUCCESS: {num_pages} pages, {len(text)} chars")
        success_count += 1
    
    print(f"\nDone! Successfully processed {success_count}/{len(pdf_files)} PDFs")

if __name__ == "__main__":
    process_all_pdfs()