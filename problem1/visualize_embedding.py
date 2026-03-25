
from word2vec_numpy import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model1 = Word2Vec.load(os.path.join(base_dir, "models", "skipgram.model"))
model2 = Word2Vec.load(os.path.join(base_dir, "models", "cbow.model"))

words = [
"student","faculty","campus","hostel","course",
"semester","degree","policy","rules","booking",
"office","department","sports","academic","program",
"room","charges","payment","approval","authority"
]

vectors = [model1.wv[word] for word in words]

# PCA reduction
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

plt.figure(figsize=(10,7))

for i, word in enumerate(words):
    x, y = result[i]
    plt.scatter(x, y)
    plt.text(x+0.01, y+0.01, word)

plt.title("Word Embedding Visualization (PCA)")
plot_dir = os.path.join(base_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)
output_path = os.path.join(plot_dir, "embedding_pca_skipgram.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
# plt.show()

print(f"Plot saved to: {output_path}")