import pdfplumber
import os
import re

try:
    import wordninja
except ImportError:
    wordninja = None

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

pdf_files = [
    "file1.pdf",
    "file2.pdf",
    "file3.pdf",
    "file4.pdf",
    "file5.pdf",
    "file6.pdf"
]

output_file = os.path.join(data_dir, "raw_corpus.txt")

all_text = []
skipped_files = []
split_count = 0


def split_merged_token(token):
    """
    Split long merged alphabetic tokens like:
    'astudentneedstoproduce...' -> ['a', 'student', 'needs', 'to', ...]
    """
    global split_count

    if not token.isalpha() or len(token) < 20 or wordninja is None:
        return [token]

    pieces = wordninja.split(token)
    if len(pieces) < 2:
        return [token]

    # Avoid noisy over-splitting.
    if any(len(p) == 1 for p in pieces):
        return [token]

    if "".join(pieces).lower() != token.lower():
        return [token]

    split_count += 1
    return pieces


def normalize_and_split_text(word_text):
    word_text = re.sub(r"\s+", " ", word_text.strip())
    if not word_text:
        return []

    chunks = word_text.split(" ")
    tokens = []
    for chunk in chunks:
        # Keep only letters/digits as word boundaries for corpus quality.
        for part in re.findall(r"[A-Za-z]+|\d+", chunk):
            tokens.extend(split_merged_token(part))
    return tokens

def extract_page_text(page):
    """
    Extract text from a page using word-level extraction.
    This avoids merged words caused by PDF spacing issues.
    """
    words = page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
        x_tolerance=2,
        y_tolerance=3,
    )

    if not words:
        return ""

    # Reconstruct line-wise text with stable row grouping.
    lines = []
    current_line_tokens = []
    current_top = None
    threshold = 2.5

    for w in words:
        token_parts = normalize_and_split_text(w.get("text", ""))
        if not token_parts:
            continue

        if current_top is None:
            current_top = w["top"]

        if abs(w["top"] - current_top) > threshold:
            if current_line_tokens:
                lines.append(" ".join(current_line_tokens))
            current_line_tokens = token_parts
            current_top = w["top"]
        else:
            current_line_tokens.extend(token_parts)

    if current_line_tokens:
        lines.append(" ".join(current_line_tokens))

    return "\n".join(lines)


for pdf in pdf_files:
    pdf_path = os.path.join(data_dir, pdf)

    try:
        with pdfplumber.open(pdf_path) as pdf_file:
            for page in pdf_file.pages:
                page_text = extract_page_text(page)

                if page_text.strip():
                    all_text.append(page_text)

    except Exception as exc:
        skipped_files.append((pdf, str(exc)))
        print(f"Skipping {pdf}: {exc}")

# Save corpus
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_text))

print(f"Corpus extracted successfully to: {output_file}")
print(f"Total pages processed: {len(all_text)}")
if wordninja is None:
    print("Note: wordninja not installed, merged-token splitting disabled.")
else:
    print(f"Merged tokens split: {split_count}")

if skipped_files:
    print("\nSkipped files:")
    for filename, reason in skipped_files:
        print(f"- {filename}: {reason}")
else:
    print("\nNo files were skipped.")