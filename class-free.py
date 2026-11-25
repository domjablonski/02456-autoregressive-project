
from protgen.data import ProteinDataset
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = ProteinDataset(path_to_data=ROOT / "data" / "train.csv")
val_data = ProteinDataset(path_to_data=ROOT / "data" / "val.csv")

PAD, BOS, EOS = 0, 1, 2
IGNORE_INDEX = -1

vocab_size = train_data.vocab_size
num_families = len(train_data.fam_to_idx)
NULL_FAMILY = num_families

embed_dim = 32
hidden_dim = 256
num_layers = 2
batch_size = 64
epochs = 20
lr = 1e-3
num_samples = 256

# Probability of dropping the family condition during training
label_dropout = 0.1


class RNN(nn.Module):
    def __init__(
        self, vocab_size, num_families,
        embed_dim, hidden_dim, num_layers
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Includes one extra embedding for the null family.
        self.family_embedding = nn.Embedding(
            num_families + 1, embed_dim
        )

        self.rnn = nn.GRU(
            embed_dim, hidden_dim,
            num_layers, batch_first=True
        )

        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, family, hidden=None):
        # x: (B, T)
        # family: (B,)

        token_emb = self.embedding(x)              # (B, T, E)
        family_emb = self.family_embedding(family) # (B, E)

        # Add the family condition to every token embedding.
        conditioned_emb = token_emb + family_emb.unsqueeze(1)

        output, hidden = self.rnn(conditioned_emb, hidden)
        logits = self.out(output)

        return logits, hidden


def sequence_loss(logits, targets):
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        ignore_index=IGNORE_INDEX,
        reduction="sum"
    )
    return loss, (targets != IGNORE_INDEX).sum()


def evaluate(model, loader):
    model.eval()
    total, count = 0, 0

    with torch.no_grad():
        for x, x_shifted, labels, _ in loader:
            x = x.to(device)
            x_shifted = x_shifted.to(device)
            labels = labels.to(device)

            # Validation always uses the true family condition.
            logits, _ = model(x, labels)

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

        for x, x_shifted, labels, _ in train_loader:
            x = x.to(device)
            x_shifted = x_shifted.to(device)
            labels = labels.to(device)

            # Classifier-free guidance training:
            # randomly replace some labels with NULL_FAMILY.
            labels_cfg = labels.clone()

            drop_mask = (
                torch.rand(labels.shape[0], device=device)
                < label_dropout
            )

            labels_cfg[drop_mask] = NULL_FAMILY

            logits, _ = model(x, labels_cfg)

            loss, n = sequence_loss(logits, x_shifted)

            optimizer.zero_grad()
            (loss / n).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            train_total += loss.item()
            train_count += n.item()

        train_loss = train_total / train_count
        val_loss = evaluate(model, val_loader)

        print(
            f"epoch {epoch + 1}: "
            f"train loss {train_loss:.3f}, "
            f"val loss {val_loss:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"best val loss: {best_val_loss:.3f}")


@torch.no_grad()
def generate(
    model,
    dataset,
    family_name,
    guidance_w=1.0,
    max_len=127
):
    model.eval()

    # Get the ID of the requested family.
    family_id = dataset.fam_to_idx[family_name]

    # Conditional and unconditional family labels.
    cond_family = torch.tensor([family_id], device=device)
    uncond_family = torch.tensor([NULL_FAMILY], device=device)

    tokens = [BOS]

    # Keep separate hidden states for the two GRU passes.
    hidden_cond = None
    hidden_uncond = None

    for _ in range(max_len):
        x = torch.tensor(
            [[tokens[-1]]], dtype=torch.long, device=device
        )

        # Conditional prediction.
        logits_cond, hidden_cond = model(
            x, cond_family, hidden_cond
        )

        # Unconditional prediction.
        logits_uncond, hidden_uncond = model(
            x, uncond_family, hidden_uncond
        )

        # CFG combination.
        logits = (
            logits_uncond
            + guidance_w * (logits_cond - logits_uncond)
        )

        next_logits = logits[0, -1].clone()

        # Never generate padding or BOS as new tokens.
        next_logits[PAD] = float("-inf")
        next_logits[BOS] = float("-inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if next_token == EOS:
            break

        tokens.append(next_token)

    return dataset.decode(tokens)


def main():
    torch.manual_seed(0)

    print(f"training on {device}")
    print(f"family mapping: {train_data.fam_to_idx}")
    print(f"null family ID: {NULL_FAMILY}")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size
    )

    model = RNN(
        vocab_size,
        num_families,
        embed_dim,
        hidden_dim,
        num_layers
    ).to(device)

    checkpoint_path = (
        ROOT / "results" / "checkpoints" / "rnn_cfg.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train(model, train_loader, val_loader, checkpoint_path)

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    )

    # Generate sequences for each family at several guidance scales.
    guidance_scales = [0.0, 1.0, 2.0, 3.0]

    results = []

    for family_name in train_data.fam_to_idx:
        for w in guidance_scales:
            print(f"Generating {family_name}, w={w}")

            for _ in range(num_samples):
                sequence = generate(
                    model,
                    train_data,
                    family_name=family_name,
                    guidance_w=w
                )

                results.append({
                    "sequence": sequence,
                    "family": family_name,
                    "guidance_w": w
                })

    out_path = ROOT / "results" / "sequences" / "rnn_cfg.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"wrote {len(results)} sequences to {out_path}")


if __name__ == "__main__":
    main()
