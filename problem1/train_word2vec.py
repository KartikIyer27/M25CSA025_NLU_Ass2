from word2vec_numpy import Word2Vec
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
tokens_path = os.path.join(base_dir, "cleaned data", "tokens.txt")
sentences_path = os.path.join(base_dir, "cleaned data", "clean_sentences.txt")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

if os.path.exists(sentences_path):
    with open(sentences_path, "r", encoding="utf-8") as f:
        sentences = [line.strip().split() for line in f if line.strip()]
else:
    with open(tokens_path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    sentences = [tokens[i:i + 20] for i in range(0, len(tokens), 20)]

EPOCHS = 25
WINDOW = 5
VECTOR_SIZE = 150
NEGATIVE = 8
MIN_COUNT = 2

print(f"Training sentences: {len(sentences)}")

# CBOW
cbow_model = Word2Vec(
    sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    sg=0,
    negative=NEGATIVE,
    min_count=MIN_COUNT,
    epochs=EPOCHS,
)
cbow_model.save(os.path.join(models_dir, "cbow.model"))

# Skip Gram
sg_model = Word2Vec(
    sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    sg=1,
    negative=NEGATIVE,
    min_count=MIN_COUNT,
    epochs=EPOCHS,
)
sg_model.save(os.path.join(models_dir, "skipgram.model"))

print("Models trained successfully!")