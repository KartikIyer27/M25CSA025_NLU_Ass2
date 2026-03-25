import torch
import torch.nn as nn
import random
import os
import json
import matplotlib.pyplot as plt

with open("TrainingNames1.txt") as f:
    names = [x.strip().lower() for x in f.readlines()]

# Add start and end tokens
names = ["^" + n + "$" for n in names]


chars = sorted(list(set("".join(names))))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

vocab_size = len(stoi)

CHECKPOINT_PATH = "vanilla_rnn_model.pth"
LOSS_CURVE_PATH = "vanilla_rnn_loss_curve.png"
LOSS_JSON_PATH = "rnn_loss.json"
GENERATED_NAMES_PATH = "rnn.txt"

if __name__ == "__main__":
    print("Vocab size:", vocab_size)

data = []

for name in names:

    input_seq = [stoi[ch] for ch in name[:-1]]
    target_seq = [stoi[ch] for ch in name[1:]]

    data.append((input_seq, target_seq))

class VanillaRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.rnn = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=True
    )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):

        x = self.embedding(x)

        out, hidden = self.rnn(x, hidden)

        out = self.fc(out)

        return out, hidden


model = VanillaRNN(vocab_size)


def init_hidden(batch_size=1):
    num_layers = model.rnn.num_layers
    return torch.zeros(num_layers, batch_size, model.hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

random.shuffle(data) 


if __name__ == "__main__":
    EPOCHS = int(os.environ.get("BOOTSTRAP_EPOCHS", "30"))
else:
    EPOCHS = 0

def train_model(epochs):
    epoch_losses = []
    for epoch in range(epochs):

        total_loss = 0

        for x_seq, y_seq in data:

            x = torch.tensor(x_seq).unsqueeze(0)
            y = torch.tensor(y_seq)

            hidden = init_hidden(batch_size=1)

            optimizer.zero_grad()

            out, hidden = model(x, hidden)

            out = out.squeeze(0)

            loss = criterion(out, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()

        epoch_losses.append(total_loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    return epoch_losses


def plot_loss_curve(losses, output_path):
    if not losses:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2)
    plt.title("Vanilla RNN Training Loss")
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


def load_checkpoint_if_available():
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return True
    return False


if __name__ == "__main__":
    losses = train_model(EPOCHS)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    save_loss_json(losses, LOSS_JSON_PATH)
    plot_loss_curve(losses, LOSS_CURVE_PATH)
else:
    load_checkpoint_if_available()


def generate_name():

    name = "^"
    hidden = init_hidden(batch_size=1)

    while True:

        x = torch.tensor([[stoi[name[-1]]]])

        out, hidden = model(x, hidden)

        probs = torch.softmax(out[0,-1], dim=0)

        idx = torch.multinomial(probs, 1).item()

        char = itos[idx]

        if char == "$" or len(name) > 10:
            break

        name += char

    return name[1:]



if __name__ == "__main__":
    print("\nGenerated Names:\n")
    for _ in range(20):
        print(generate_name())

    
    generated_names = [generate_name() for _ in range(200)]
    save_generated_names(generated_names, GENERATED_NAMES_PATH)

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