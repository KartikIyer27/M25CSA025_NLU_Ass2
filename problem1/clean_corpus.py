import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

try:
    import wordninja
except ImportError:
    wordninja = None

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words('english'))

base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "data", "raw_corpus.txt")
output_dir = os.path.join(base_dir, "cleaned data")
word_dir = os.path.join(base_dir, "cleaned data")
word_path = os.path.join(output_dir, "c_corpus.txt")
output_path = os.path.join(output_dir, "clean_corpus.txt")
sentences_path = os.path.join(output_dir, "clean_sentences.txt")

with open(input_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Normalize line breaks and spacing.
text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
text = text.lower()

text = re.sub(r"\d+", " ", text)
text = re.sub(r"[^a-z\s]", " ", text)
text = re.sub(r"\s+", " ", text).strip()

words = text.split()
with open(word_path, "w", encoding="utf-8") as f:
    f.write(" ".join(words))


def split_run_on_word(word):
    if wordninja is None or len(word) < 20 or not word.isalpha():
        return [word]

    parts = wordninja.split(word)
    if len(parts) < 2:
        return [word]

    if any(len(p) == 1 for p in parts):
        return [word]

    if "".join(parts).lower() != word.lower():
        return [word]

    return parts


expanded_words = []
split_count = 0
for word in words:
    parts = split_run_on_word(word)
    if len(parts) > 1:
        split_count += 1
    expanded_words.extend(parts)


def merge_domain_terms(tokens):
  
    merged = []
    i = 0
    while i < len(tokens):
        curr = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None

        if curr == "b" and nxt == "tech":
            merged.append("btech")
            i += 2
            continue
        if curr == "m" and nxt == "tech":
            merged.append("mtech")
            i += 2
            continue
        if curr == "u" and nxt == "g":
            merged.append("ug")
            i += 2
            continue
        if curr == "p" and nxt == "g":
            merged.append("pg")
            i += 2
            continue

        merged.append(curr)
        i += 1

    return merged


def merge_domain_phrases(tokens):
   
    phrase_map = {
        ("full", "time"): "full_time",
        ("part", "time"): "part_time",
        ("dual", "degree"): "dual_degree",
        ("course", "work"): "course_work",
        ("grade", "point"): "grade_point",
        ("academic", "programmes"): "academic_programmes",
        ("academic", "programs"): "academic_programs",
        ("computer", "science"): "computer_science",
        ("artificial", "intelligence"): "artificial_intelligence",
        ("machine", "learning"): "machine_learning",
        ("word", "to", "vec"): "word2vec",
    }

    max_len = max(len(k) for k in phrase_map)
    merged = []
    i = 0
    while i < len(tokens):
        matched = False
        for n in range(max_len, 1, -1):
            if i + n > len(tokens):
                continue
            candidate = tuple(tokens[i:i + n])
            if candidate in phrase_map:
                merged.append(phrase_map[candidate])
                i += n
                matched = True
                break
        if not matched:
            merged.append(tokens[i])
            i += 1

    return merged


def normalize_tokens(text_fragment):
    frag = text_fragment.lower()
    frag = re.sub(r"\d+", " ", frag)
    frag = re.sub(r"[^a-z\s]", " ", frag)
    frag = re.sub(r"\s+", " ", frag).strip()
    if not frag:
        return []

    frag_tokens = frag.split()
    expanded = []
    for tok in frag_tokens:
        expanded.extend(split_run_on_word(tok))

    merged = merge_domain_terms(expanded)
    merged = merge_domain_phrases(merged)
    merged = [w for w in merged if w not in stop_words]
    return merged


normalized_words = merge_domain_terms(expanded_words)
normalized_words = merge_domain_phrases(normalized_words)

filtered_words = [w for w in normalized_words if w not in stop_words]

clean_text = " ".join(filtered_words)

# Build sentence-level corpus with exact deduplication to reduce repeated boilerplate.
raw_paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
seen_paragraphs = set()
unique_paragraphs = []
for para in raw_paragraphs:
    key = re.sub(r"\s+", " ", para.lower()).strip()
    if key in seen_paragraphs:
        continue
    seen_paragraphs.add(key)
    unique_paragraphs.append(para)

clean_sentences = []
for para in unique_paragraphs:
    for sent in sent_tokenize(para):
        sent_tokens = normalize_tokens(sent)
        if len(sent_tokens) >= 3:
            clean_sentences.append(" ".join(sent_tokens))

os.makedirs(output_dir, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(clean_text)

with open(sentences_path, "w", encoding="utf-8") as f:
    f.write("\n".join(clean_sentences))

print(f"Stopwords removed and corpus cleaned: {output_path}")
print(f"Sentence corpus written: {sentences_path}")
print(f"Original words: {len(words)}")
print(f"After splitting: {len(expanded_words)}")
print(f"After domain normalization: {len(normalized_words)}")
print(f"After stopword removal: {len(filtered_words)}")
print(f"Paragraphs (raw/unique): {len(raw_paragraphs)}/{len(unique_paragraphs)}")
print(f"Clean sentences: {len(clean_sentences)}")
if wordninja is None:
    print("Note: wordninja not installed, merged-word splitting disabled.")
else:
    print(f"Merged words split: {split_count}")