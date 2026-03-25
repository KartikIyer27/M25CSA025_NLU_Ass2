from word2vec_numpy import Word2Vec

model = Word2Vec.load("models/skipgram.model")

def safe_analogy(pos, neg):
    # print(f"\n{label}")
    print(f"positive={pos}, negative={neg}")
    try:
        print(model.wv.most_similar(positive=pos, negative=neg, topn=5))
    except KeyError as e:
        print("Word missing:", e)

# Target analogy: ug : btech :: pg : mtech
# Vector form: btech - ug + pg -> expected mtech
safe_analogy(['ug', 'btech'], ['pg'])

print("\nUsing CBOW model:")
model1 = Word2Vec.load("models/cbow.model")

def safe_analogy_cbow(pos, neg):
    # print(f"\n{label}")
    print(f"positive={pos}, negative={neg}")
    try:
        print(model1.wv.most_similar(positive=pos, negative=neg, topn=5))
    except KeyError as e:
        print("Word missing:", e)

safe_analogy_cbow(['vehicle', 'parking'], ['student'])
safe_analogy_cbow(['security', 'campus'], ['student'])
safe_analogy_cbow(['equipment', 'sports'], ['vehicle'])