"""The family predictor that classifier guidance steers with.

    uv run python src/protgen/models/classifier.py            # the guidance predictor
    uv run python src/protgen/models/classifier.py --judge    # the independent judge

Guidance needs p(family | prefix) at every step of generation, so this is trained on *partial*
sequences - the "noisy" predictor of the project plan. A GRU is causal, so one pass over a whole
sequence already gives the hidden state after every prefix of it: position j reads tokens 0..j and
nothing later. Training therefore scores every prefix of every protein at once, rather than
cutting each sequence at one random length and throwing the rest away.

--judge trains the same architecture on complete sequences only, with a different seed. That one
never guides anything; it is what labels the generated sets afterwards, so the number reported is
not the guidance predictor grading its own work.
"""
import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import FAMILIES, ProteinDataset

matplotlib.use("Agg")           # a cluster node has no display
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = ProteinDataset(path_to_data=ROOT / "data" / "train.csv")
val_data = ProteinDataset(path_to_data=ROOT / "data" / "val.csv")

PAD, BOS, EOS = 0, 1, 2
vocab_size = train_data.vocab_size
num_classes = len(FAMILIES)
embed_dim = 32
hidden_dim = 256
num_layers = 2
batch_size = 64
epochs = 15
lr = 1e-3


class Classifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, hidden=None):
        output, hidden = self.rnn(self.embedding(x), hidden)   # (B, T, hidden_dim)
        return self.out(output), hidden                        # (B, T, num_classes)

    def step(self, tokens, hidden):
        """One token on from `hidden`: log p(family | prefix + token) and the new state.

        `tokens` is (B, 1), so the 23 candidate next tokens of one generation step go through as
        one batch instead of 23 forward passes over the whole prefix.
        """
        logits, hidden = self(tokens, hidden)
        return F.log_softmax(logits[:, -1], dim=-1), hidden


def prefix_mask(lengths, max_length):
    """(B, T) true where position j is a prefix worth scoring: tokens 0..j are all real.

    Position j reads x[0..j]. x is <BOS>, the residues, <EOS>, then padding, so the last position
    whose prefix carries no padding is length + 1, the <EOS>.
    """
    positions = torch.arange(max_length, device=lengths.device)
    return positions.unsqueeze(0) <= (lengths + 1).unsqueeze(1)


def batch_loss(model, x, y, lengths):
    """Summed cross-entropy over every prefix in the batch, and how many there were."""
    logits, _ = model(x)                                    # (B, T, num_classes)
    mask = prefix_mask(lengths, x.shape[1])                 # (B, T)
    targets = y.unsqueeze(1).expand(-1, x.shape[1])         # the family is the same at every j
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return (loss * mask).sum(), mask.sum()


def full_sequence_loss(model, x, y, lengths):
    """Cross-entropy at the <EOS> position only: the judge sees complete sequences."""
    logits, _ = model(x)
    last = logits[torch.arange(len(x), device=x.device), lengths + 1]
    return F.cross_entropy(last, y, reduction="sum"), torch.tensor(len(x), device=x.device)


def evaluate(model, loader, loss_fn):
    model.eval()
    total, count = 0, 0
    with torch.no_grad():
        for x, _, y, lengths in loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            loss, n = loss_fn(model, x, y, lengths)
            total += loss.item()
            count += n.item()
    return total / count


def train(model, train_loader, val_loader, checkpoint_path, loss_fn):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_total, train_count = 0, 0
        for x, _, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            loss, n = loss_fn(model, x, y, lengths)

            optimizer.zero_grad()
            (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_total += loss.item()
            train_count += n.item()

        train_loss = train_total / train_count
        val_loss = evaluate(model, val_loader, loss_fn)
        print(f"epoch {epoch}: train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"best val loss: {best_val_loss:.3f}")


@torch.no_grad()
def prefix_accuracy(model, data, batch_size=batch_size):
    """Accuracy per prefix length: the plot that says where guidance can do anything.

    Prefix length j + 1 means the model has read x[0..j]: <BOS> alone at length 1, <BOS> and the
    first residue at length 2, and so on. A protein shorter than the prefix contributes nothing to
    that row, so `n` falls away as the lengths run out.
    """
    model.eval()
    loader = DataLoader(data, batch_size=batch_size)
    correct = torch.zeros(data.max_length, device=device)
    counted = torch.zeros(data.max_length, device=device)

    for x, _, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        logits, _ = model(x)
        hit = logits.argmax(dim=-1) == y.unsqueeze(1)       # (B, T)
        mask = prefix_mask(lengths, x.shape[1])
        correct += (hit & mask).sum(dim=0)
        counted += mask.sum(dim=0)

    keep = counted > 0
    return pd.DataFrame({
        "prefix_length": torch.arange(1, data.max_length + 1, device=device)[keep].cpu(),
        "accuracy": (correct[keep] / counted[keep]).cpu(),
        "n": counted[keep].cpu().int(),
    })


def plot_accuracy(table, out_path, title):
    chance = 1 / num_classes
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(table["prefix_length"], table["accuracy"], color="#2b6cb0")
    axes.axhline(chance, color="#a0aec0", linestyle="--", label=f"chance ({chance:.2f})")
    axes.set_xlabel("prefix length (tokens read)")
    axes.set_ylabel("accuracy")
    axes.set_ylim(0, 1)
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", action="store_true",
                        help="train the independent full-sequence judge instead")
    args = parser.parse_args()

    name = "classifier_judge" if args.judge else "classifier"
    loss_fn = full_sequence_loss if args.judge else batch_loss
    torch.manual_seed(1 if args.judge else 0)
    print(f"training {name} on {device}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = Classifier(vocab_size, embed_dim, hidden_dim, num_layers, num_classes).to(device)
    checkpoint_path = ROOT / "results" / "checkpoints" / f"{name}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train(model, train_loader, val_loader, checkpoint_path, loss_fn)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    table = prefix_accuracy(model, val_data)
    out_dir = ROOT / "results" / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{name}_prefix_accuracy.csv", index=False)
    plot_accuracy(table, out_dir / f"{name}_prefix_accuracy.png",
                  f"{name}: family accuracy by prefix length")

    print(f"\naccuracy at prefix length 1: {table['accuracy'].iloc[0]:.3f} (chance "
          f"{1 / num_classes:.3f})")
    for length in (5, 10, 20, 50):
        row = table[table["prefix_length"] == length]
        if not row.empty:
            print(f"accuracy at prefix length {length}: {row['accuracy'].iloc[0]:.3f}")
    print(f"accuracy on the full sequence: {table['accuracy'].iloc[-1]:.3f}")
    print(f"wrote {out_dir / f'{name}_prefix_accuracy.csv'} and the plot beside it")


if __name__ == "__main__":
    main()
