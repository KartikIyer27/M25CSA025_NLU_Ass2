import torch
import torch.nn as nn
import random
import os
import json
import matplotlib.pyplot as plt

# =========================
# Load Data
# =========================

with open("TrainingNames1.txt") as f:
    names = [x.strip().lower() for x in f.readlines()]

names = [n for n in names if n.isalpha() and len(n) >= 3]
names = ["^" + n + "$" for n in names]

# =========================
# Vocabulary
# =========================

chars = sorted(list(set("".join(names))))

stoi = {ch:i+1 for i,ch in enumerate(chars)}
stoi["<PAD>"] = 0

itos = {i:ch for ch,i in stoi.items()}

PAD = 0
vocab_size = len(stoi)
CHECKPOINT_PATH = "rnn_attention_model.pth"
LOSS_CURVE_PATH = "rnn_attention_loss_curve.png"
GENERATED_NAMES_PATH = "attention.txt"
LOSS_JSON_PATH = "attention_loss.json"

# =========================
# Encode + Pad
# =========================

max_len = max(len(n) for n in names)

def encode(name):
    return [stoi[ch] for ch in name]

def pad(seq):
    return seq + [PAD]*(max_len - len(seq))

inputs = []
targets = []

for name in names:
    seq = encode(name)
    inputs.append(pad(seq[:-1]))
    targets.append(pad(seq[1:]))

X = torch.tensor(inputs)
Y = torch.tensor(targets)

# =========================
# RNN + Attention Model
# =========================

class RNNAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = nn.RNN(
            hidden_size,
            hidden_size,
            batch_first=True
        )

        # Attention layer
        self.attn = nn.Linear(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        outputs, hidden = self.rnn(x)   # outputs: [B, T, H]

        # Attention scores
        attn_scores = torch.bmm(outputs, outputs.transpose(1, 2))  # [B, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Context vector
        context = torch.bmm(attn_weights, outputs)  # [B, T, H]

        # Combine
        combined = torch.cat((outputs, context), dim=2)

        out = self.fc(combined)

        return out


model = RNNAttention(vocab_size, hidden_size=64)

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# Training
# =========================

if __name__ == "__main__":
    EPOCHS = int(os.environ.get("BOOTSTRAP_EPOCHS", "50"))
else:
    EPOCHS = 0
batch_size = 32
epoch_losses = []

for epoch in range(EPOCHS):

    perm = torch.randperm(X.size(0))
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]

    total_loss = 0

    for i in range(0, X.size(0), batch_size):

        xb = X_shuffled[i:i+batch_size]
        yb = Y_shuffled[i:i+batch_size]

        optimizer.zero_grad()

        out = model(xb)

        loss = criterion(
            out.view(-1, vocab_size),
            yb.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


def plot_loss_curve(losses, output_path):
    if not losses:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2)
    plt.title("RNN Attention Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss curve saved as '{output_path}'")


def save_generated_names(names, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
    print(f"Generated names saved as '{output_path}'")


def save_loss_json(losses, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(losses, f)
    print(f"Loss values saved as '{output_path}'")

if __name__ == "__main__":
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    plot_loss_curve(epoch_losses, LOSS_CURVE_PATH)
else:
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
        model.eval()

# =========================
# Generation
# =========================

def generate():

    name = "^"

    for _ in range(12):

        x = torch.tensor([[stoi.get(ch, PAD) for ch in name]])

        out = model(x)

        logits = out[0, -1]

        probs = torch.softmax(logits / 0.7, dim=0)

        probs[PAD] = 0
        probs = probs / probs.sum()

        idx = torch.multinomial(probs, 1).item()
        char = itos[idx]

        if char == "$" and len(name) > 3:
            break

        if char == "$":
            continue

        name += char

    return name[1:]


# =========================
# Generate Names
# =========================

if __name__ == "__main__":
    print("\nGenerated Names:\n")
    for _ in range(10):
        print(generate())


    generated_names = [generate() for _ in range(200)]
    save_generated_names(generated_names, GENERATED_NAMES_PATH)
    save_loss_json(epoch_losses, LOSS_JSON_PATH)

    with open("TrainingNames.txt") as f:
        train_names = set(x.strip().lower() for x in f.readlines())

    total = len(generated_names)
    unique = len(set(generated_names))
    novel = [n for n in generated_names if n not in train_names]

    novelty = len(novel) / total
    diversity = unique / total

    print("\n--- Evaluation ---")
    print("Total Generated:", total)
    print("Unique Names:", unique)
    print("Novel Names:", len(novel))
    print("Novelty Rate:", round(novelty, 3))
    print("Diversity:", round(diversity, 3))