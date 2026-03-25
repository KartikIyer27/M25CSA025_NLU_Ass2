from collections import Counter

with open("cleaned data/tokens.txt","r",encoding="utf-8") as f:
    tokens = f.read().split()

vocab = set(tokens)

print("Total Tokens:", len(tokens))
print("Vocabulary Size:", len(vocab))

counter = Counter(tokens)

print("\nTop 20 words:")
for word, freq in counter.most_common(20):
    print(word, freq)