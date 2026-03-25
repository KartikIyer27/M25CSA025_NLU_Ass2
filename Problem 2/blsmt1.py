import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import json

# from rnn_attention import generate

# =========================
# Config
# =========================

DATA_FILE = "TrainingNames1.txt"
HIDDEN_SIZE = 64
EMBED_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("BOOTSTRAP_EPOCHS", "100"))
LR = 0.001
MAX_GEN_LEN = 12
MIN_GEN_LEN = 3
TEMPERATURE = 0.8
SEED = 42
LOSS_CURVE_PATH = "blsmt1_loss_curve.png"
GENERATED_NAMES_PATH = "blstm.txt"
LOSS_JSON_PATH = "blstm_loss.json"

torch.manual_seed(SEED)

# =========================
# Load and clean data
# =========================

with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_names = [x.strip().lower() for x in f.readlines()]

names = []
for n in raw_names:
    if n.isalpha() and len(n) >= 3:
        names.append("^" + n + "$")

print("Total usable names:", len(names))

# =========================
# Vocabulary
# =========================

chars = sorted(list(set("".join(names))))
PAD_TOKEN = "<PAD>"

stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi[PAD_TOKEN] = 0
itos = {i: ch for ch, i in stoi.items()}

PAD = stoi[PAD_TOKEN]
vocab_size = len(stoi)

print("Vocabulary size:", vocab_size)
print("Characters:", chars)

# =========================
# Prepare forward/backward LM datasets
# =========================

max_len = max(len(n) for n in names)

def encode(seq):
    return [stoi[ch] for ch in seq]

def pad(seq, length):
    return seq + [PAD] * (length - len(seq))

forward_inputs = []
forward_targets = []
backward_inputs = []
backward_targets = []

for name in names:
    seq = encode(name)

    # Forward LM
    f_in = seq[:-1]
    f_tg = seq[1:]

    # Backward LM on reversed sequence
    rev = list(reversed(seq))
    b_in = rev[:-1]
    b_tg = rev[1:]

    forward_inputs.append(pad(f_in, max_len - 1))
    forward_targets.append(pad(f_tg, max_len - 1))
    backward_inputs.append(pad(b_in, max_len - 1))
    backward_targets.append(pad(b_tg, max_len - 1))

Xf = torch.tensor(forward_inputs, dtype=torch.long)
Yf = torch.tensor(forward_targets, dtype=torch.long)
Xb = torch.tensor(backward_inputs, dtype=torch.long)
Yb = torch.tensor(backward_targets, dtype=torch.long)

# =========================
# ELMo-style BLSTM
# =========================

class ELMoStyleBLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)

        self.forward_lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.backward_lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.forward_head = nn.Linear(hidden_size, vocab_size)
        self.backward_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, xf, xb):
        ef = self.embedding(xf)
        eb = self.embedding(xb)

        of, _ = self.forward_lstm(ef)
        ob, _ = self.backward_lstm(eb)

        logits_f = self.forward_head(of)
        logits_b = self.backward_head(ob)

        return logits_f, logits_b

    def forward_only(self, x, hidden=None):
        e = self.embedding(x)
        o, hidden = self.forward_lstm(e, hidden)
        logits = self.forward_head(o)
        return logits, hidden

# =========================
# Initialize
# =========================

model = ELMoStyleBLSTM(
    vocab_size=vocab_size,
    embed_size=EMBED_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

criterion = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# Parameter count
# =========================

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable parameters:", total_params)

# =========================
# Training
# =========================

num_samples = Xf.size(0)
epoch_losses = []

for epoch in range(EPOCHS):
    perm = torch.randperm(num_samples)

    Xf_sh = Xf[perm]
    Yf_sh = Yf[perm]
    Xb_sh = Xb[perm]
    Yb_sh = Yb[perm]

    total_loss = 0.0

    model.train()

    for i in range(0, num_samples, BATCH_SIZE):
        xf_batch = Xf_sh[i:i+BATCH_SIZE]
        yf_batch = Yf_sh[i:i+BATCH_SIZE]
        xb_batch = Xb_sh[i:i+BATCH_SIZE]
        yb_batch = Yb_sh[i:i+BATCH_SIZE]

        optimizer.zero_grad()

        logits_f, logits_b = model(xf_batch, xb_batch)

        loss_f = criterion(logits_f.reshape(-1, vocab_size), yf_batch.reshape(-1))
        loss_b = criterion(logits_b.reshape(-1, vocab_size), yb_batch.reshape(-1))
        loss = loss_f + loss_b

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    epoch_losses.append(total_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


def plot_loss_curve(losses, output_path):
    if not losses:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2)
    plt.title("BLSTM1 Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss curve saved as '{output_path}'")


plot_loss_curve(epoch_losses, LOSS_CURVE_PATH)


def save_generated_names(names, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
    print(f"Generated names saved as '{output_path}'")


def save_loss_json(losses, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(losses, f)
    print(f"Loss values saved as '{output_path}'")

# =========================
# Generation using forward LM only
# =========================

def sample_next_char(logits, name_so_far):
    probs = torch.softmax(logits / TEMPERATURE, dim=0)

    # Block PAD
    probs[PAD] = 0.0

    # Light repetition penalty
    if len(name_so_far) > 1:
        last_char = name_so_far[-1]
        if last_char in stoi:
            probs[stoi[last_char]] *= 0.2

    probs = probs / probs.sum()

    idx = torch.multinomial(probs, 1).item()
    return itos[idx]

def generate_name():
    model.eval()

    while True:
        name = "^"
        hidden = None

        for _ in range(MAX_GEN_LEN):
            x = torch.tensor([[stoi[ch] for ch in name]], dtype=torch.long)

            with torch.no_grad():
                logits, hidden = model.forward_only(x[:, -1:].contiguous(), hidden)

            next_logits = logits[0, -1]
            ch = sample_next_char(next_logits, name)

            if ch == "$":
                if len(name) - 1 >= MIN_GEN_LEN:
                    break
                else:
                    continue

            if ch == PAD_TOKEN:
                continue

            name += ch

        final_name = name[1:]
        if len(final_name) >= MIN_GEN_LEN and final_name.isalpha():
            return final_name

# =========================
# Generate samples
# =========================

print("\nGenerated Names:\n")
for _ in range(10):
    print(generate_name())


generated_names = [generate_name() for _ in range(200)]
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