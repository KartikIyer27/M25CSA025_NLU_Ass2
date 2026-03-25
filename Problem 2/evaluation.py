
import os

def save(names, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for n in names:
            f.write(n + "\n")


def load_training_names():
    # Prefer TrainingNames1.txt to match the training data used by the models.
    for filename in ("TrainingNames1.txt", "TrainingNames.txt"):
        try:
            with open(filename, encoding="utf-8") as f:
                return set(x.strip().lower() for x in f if x.strip())
        except FileNotFoundError:
            continue
    raise FileNotFoundError("TrainingNames1.txt or TrainingNames.txt was not found.")


def evaluate(generated, train):
    total = len(generated)
    if total == 0:
        return 0.0, 0.0

    unique = len(set(generated))
    novel = [x for x in generated if x not in train]

    novelty = len(novel) / total
    diversity = unique / total
    return novelty, diversity


def load_generated_names(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, encoding="utf-8") as f:
        names = [x.strip().lower() for x in f if x.strip()]
    return names


def main():
    generated_rnn = load_generated_names("rnn.txt")
    generated_blstm = load_generated_names("blstm.txt")
    generated_attn = load_generated_names("attention.txt")

    if not generated_rnn:
        print("Missing or empty rnn.txt")
    if not generated_blstm:
        print("Missing or empty blstm.txt")
    if not generated_attn:
        print("Missing or empty attention.txt")

    train = load_training_names()

    print("RNN:", evaluate(generated_rnn, train))
    print("BLSTM:", evaluate(generated_blstm, train))
    print("ATTENTION:", evaluate(generated_attn, train))


if __name__ == "__main__":
    main()