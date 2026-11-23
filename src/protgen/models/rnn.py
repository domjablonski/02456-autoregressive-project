from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = ProteinDataset(path_to_data=ROOT / "data" / "train.csv")
val_data = ProteinDataset(path_to_data=ROOT / "data" / "val.csv")

PAD, BOS, EOS = 0, 1, 2
IGNORE_INDEX = -1
vocab_size = train_data.vocab_size
embed_dim = 32
hidden_dim = 256
num_layers = 2
batch_size = 64
epochs = 20
lr = 1e-3
num_samples = 256


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(self.embedding(x), hidden)    # (B, T, hidden_dim)
        return self.out(output), hidden                         # (B, T, vocab_size)


def sequence_loss(logits, targets):
    # cross_entropy wants the classes in dim 1: (B, vocab_size, T). Returns the summed loss and the
    # number of real targets, so padding never counts and the average is per predicted token.
    loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX, reduction="sum")
    return loss, (targets != IGNORE_INDEX).sum()


def evaluate(model, loader):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for x, x_shifted, _, _ in loader:
            x, x_shifted = x.to(device), x_shifted.to(device)
            logits, _ = model(x)
            loss, n = sequence_loss(logits, x_shifted)
            total += loss.item()
            count += n.item()
    return total / count


def train(model, train_loader, val_loader, checkpoint_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_total, train_count = 0, 0
        for x, x_shifted, _, _ in train_loader:
            x, x_shifted = x.to(device), x_shifted.to(device)
            logits, _ = model(x)
            loss, n = sequence_loss(logits, x_shifted)

            optimizer.zero_grad()
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total += loss.item()
            train_count += n.item()

        train_loss = train_total / train_count
        val_loss = evaluate(model, val_loader)
        print(f"epoch {epoch}: train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"best val loss: {best_val_loss:.3f}")


@torch.no_grad()
def generate(model, dataset, max_len=127):
    # One token at a time: the hidden state carries everything generated so far.
    model.eval()
    tokens, hidden = [BOS], None
    for _ in range(max_len):
        x = torch.tensor([[tokens[-1]]], device=device)       # (1, 1)
        logits, hidden = model(x, hidden)
        probs = F.softmax(logits[0, -1], dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == EOS:
            break
        tokens.append(next_token)
    return dataset.decode(tokens)


def main():
    torch.manual_seed(0)
    print(f"training on {device}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = RNN(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
    checkpoint_path = ROOT / "results" / "checkpoints" / "rnn.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train(model, train_loader, val_loader, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    sequences = [generate(model, train_data) for _ in range(num_samples)]
    out_path = ROOT / "results" / "sequences" / "rnn.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sequence": sequences}).to_csv(out_path, index=False)
    print(f"wrote {len(sequences)} sequences to {out_path}")


if __name__ == "__main__":
    main()
