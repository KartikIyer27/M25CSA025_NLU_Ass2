from nltk.tokenize import word_tokenize
import nltk
import os
import re

try:
    import wordninja
except ImportError:
    wordninja = None

nltk.download('punkt')
nltk.download('punkt_tab')

base_dir = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(base_dir, "cleaned data", "clean_corpus.txt"),
    os.path.join(base_dir, "clean_corpus.txt"),
]

input_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if input_path is None:
    raise FileNotFoundError(
        "clean_corpus.txt not found. Run clean_corpus.py first or place it in problem1 or problem1/cleaned data."
    )

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()


def split_run_on_token(token):
    """
    Split long merged tokens (e.g., astudentneedsto...) when possible.
    Falls back to original token if split confidence is low or package missing.
    """
    if not token.isalpha() or len(token) < 20 or wordninja is None:
        return [token]

    pieces = wordninja.split(token)
    if len(pieces) < 2:
        return [token]

    # Basic sanity checks to avoid over-splitting short fragments.
    if any(len(p) == 1 for p in pieces):
        return [token]

    if "".join(pieces).lower() != token.lower():
        return [token]

    return pieces


raw_tokens = word_tokenize(text)
tokens = []
split_count = 0

for tok in raw_tokens:
    tok = tok.strip().lower()
    if not tok:
        continue

    # Skip punctuation-only tokens.
    if re.fullmatch(r"\W+", tok):
        continue

    expanded = split_run_on_token(tok)
    if len(expanded) > 1:
        split_count += 1
    tokens.extend(expanded)

print("Total tokens:", len(tokens))
if wordninja is None:
    print("Note: wordninja not installed, merged-word splitting disabled.")
else:
    print("Merged tokens split:", split_count)

# save tokens
output_path = os.path.join(base_dir, "cleaned data", "tokens.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(" ".join(tokens))