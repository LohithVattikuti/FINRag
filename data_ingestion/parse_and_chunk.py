import os
import json
import html2text
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 🔧 Configurable parameters
# ──────────────────────────────────────────────────────────────────────────────

# How many words per chunk (approximate)
CHUNK_SIZE = 300  
# How many words to overlap between successive chunks
OVERLAP_SIZE = 50  

# Input folder (where your .html files live)
INPUT_DIR = "sec_filings"
# Output folder (where JSON chunks will be saved)
OUTPUT_DIR = "sec_chunks"

# ──────────────────────────────────────────────────────────────────────────────
# 🔄 Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def parse_html_to_text(html_content: str) -> str:
    """
    Convert raw HTML into clean plain text using html2text.
    - ignore_links: removes URLs
    - ignore_images: removes image alt text
    - body_width=0: prevents automatic wrapping
    """
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html_content)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP_SIZE) -> list[str]:
    """
    Split text into overlapping chunks of words.
    1. Split the full text into a list of words.
    2. Slide a window of size `chunk_size` over the words.
    3. Move the window forward by (chunk_size - overlap) each time.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        # Join the subset of words into one chunk
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Advance by chunk_size - overlap to create overlap
        start += chunk_size - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# 🔧 Main processing function
# ──────────────────────────────────────────────────────────────────────────────

def process_all_files(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    """
    1. Reads each .html file from input_dir.
    2. Parses HTML to plain text.
    3. Cleans up extra blank lines.
    4. Splits text into overlapping chunks.
    5. Saves each chunk as a JSON file with metadata.
    """
    # Ensure the output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # List all HTML files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith(".html")]

    for filename in tqdm(files, desc="Processing filings"):
        filepath = os.path.join(input_dir, filename)

        # 1️⃣ Read the raw HTML
        with open(filepath, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 2️⃣ Convert HTML → plain text
        text = parse_html_to_text(html_content)

        # 3️⃣ Remove extra blank lines and trim whitespace
        cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(cleaned_lines)

        # 4️⃣ Split into word-based chunks
        chunks = chunk_text(cleaned_text)

        # 5️⃣ Extract metadata from filename:
        #    Format assumed: TICKER_FORMTYPE_YYYY-MM-DD.html
        parts = filename.split("_")
        ticker = parts[0]
        form_type = parts[1]
        filing_date = parts[2].replace(".html", "")

        # 6️⃣ Save each chunk as its own JSON file
        for idx, chunk in enumerate(chunks):
            chunk_data = {
                "ticker": ticker,
                "form_type": form_type,
                "filing_date": filing_date,
                "chunk_id": idx,
                "text": chunk
            }
            out_filename = f"{ticker}_{form_type}_{filing_date}_chunk{idx}.json"
            out_path = os.path.join(output_dir, out_filename)
            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(chunk_data, out_f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# 🏁 Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    process_all_files()
