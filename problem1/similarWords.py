from word2vec_numpy import Word2Vec
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "skipgram.model")
model = Word2Vec.load(model_path)

words = ["research","student","phd","exam"]

for word in words:
    print("\nNearest words for:",word)
    print(model.wv.most_similar(word, topn=5))