from protgen.data import ProteinDataset
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[3]
device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = ProteinDataset(path_to_data=ROOT / "data" / "train.csv")
val_data = ProteinDataset(path_to_data=ROOT / "data" / "val.csv")

k = 128
PAD, BOS, EOS = 0, 1, 2
vocab_size = train_data.vocab_size 
embed_dim = 16 
hidden_dims = 512
batch_size = 256
epochs = 10
num_samples = 256

def build_model(vocab_size, k):
    return nn.Sequential(
        nn.Embedding(vocab_size, embed_dim),
        nn.Flatten(start_dim=1),
        nn.Linear(k * embed_dim, hidden_dims),
        nn.ReLU(),
        nn.Linear(hidden_dims, vocab_size)
)

def make_windows(k, dataset):
    tokens = dataset.tokens                                # (N, 129)
    targets = dataset.tokens_shifted                       # (N, 129)

    padding = torch.zeros(len(tokens), k - 1, dtype=torch.long)
    padded = torch.cat([padding, tokens], dim=1)           # (N, k - 1 + 129)

    windows = padded.unfold(dimension=1, size=k, step=1)   # (N, 129, k)
    mask = targets != -1  
    return (windows[mask], targets[mask])

def evaluate(model, loader):
    model.eval()
    total = 0 
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total +=F.cross_entropy(model(xb), yb, reduction="sum").item()
    return total/len(loader.dataset)

def train(model, train_loader, val_loader, checkpoint_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss =F.cross_entropy(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(xb)

        train_loss = train_total / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader)
        print(f"epoch {epoch}: train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"best val loss: {best_val_loss:.3f}")


@torch.no_grad()
def generate(model, dataset, k, max_len=127):
    model.eval()
    tokens = [PAD] * (k - 1) + [BOS]
    for _ in range(max_len):
        window = torch.tensor([tokens[-k:]], device=device)
        probs = F.softmax(model(window), dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        if next_token == EOS:
            break
        tokens.append(next_token)
    return dataset.decode(tokens)


def main():
    torch.manual_seed(0)

    X_train, y_train = make_windows(k, train_data)
    X_val, y_val = make_windows(k, val_data)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model = build_model(train_data.vocab_size, k).to(device)
    print(f"training on {device}")

    checkpoint_path = ROOT / "results" / "checkpoints" / "mlp.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train(model, train_loader, val_loader, checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    sequences = [generate(model, train_data, k) for _ in range(num_samples)]
    out_path = ROOT / "results" / "sequences" / "mlp.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sequence": sequences}).to_csv(out_path, index=False)
    print(f"wrote {len(sequences)} sequences to {out_path}")



if __name__ == "__main__":
    main()


    
