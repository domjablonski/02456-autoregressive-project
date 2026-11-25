import inspect, math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from protgen.models import mlp, rnn

ROOT = Path(".")
PAD = 0
ck_m = torch.load("results/checkpoints/mlp.pt", map_location="cpu")
ck_r = torch.load("results/checkpoints/rnn.pt", map_location="cpu")

# MLP: infer vocab and window size from the weight shapes
vocab, embed = ck_m["0.weight"].shape
k = ck_m["2.weight"].shape[1] // embed
mlp_model = mlp.build_model(vocab, k)
mlp_model.load_state_dict(ck_m)
mlp_model.eval()
print(f"MLP: vocab={vocab} embed={embed} k={k}")

# RNN: infer sizes from the weight shapes
vocab_r, embed_r = ck_r["embedding.weight"].shape
hidden = ck_r["rnn.weight_hh_l0"].shape[1]
layers = len([n for n in ck_r if n.startswith("rnn.weight_ih_l")])
print("RNN signature:", inspect.signature(rnn.RNN))
rnn_model = rnn.RNN(vocab_r, embed_r, hidden, layers)
rnn_model.load_state_dict(ck_r)
rnn_model.eval()
print(f"RNN: vocab={vocab_r} embed={embed_r} hidden={hidden} layers={layers}")

for split in ["val", "test"]:
    ds = mlp.ProteinDataset(path_to_data=ROOT / "data" / f"{split}.csv")
    X, y = mlp.make_windows(k, ds)
    tot_all, tot_masked, n_masked = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            logits = mlp_model(X[i:i+8192])
            yb = y[i:i+8192]
            ce = F.cross_entropy(logits, yb, reduction="none")
            tot_all += ce.sum().item()
            keep = yb != PAD
            tot_masked += ce[keep].sum().item()
            n_masked += keep.sum().item()
    mlp_all = tot_all / len(X)
    mlp_masked = tot_masked / n_masked

    ds_r = rnn.ProteinDataset(path_to_data=ROOT / "data" / f"{split}.csv")
    rnn_loss = rnn.evaluate(rnn_model, DataLoader(ds_r, batch_size=64))

    print(f"\n[{split}]  targets: MLP all={len(X)}, non-PAD={n_masked}")
    print(f"  MLP, as logged (PAD counted): {mlp_all:.3f}  ppl {math.exp(mlp_all):.2f}")
    print(f"  MLP, PAD excluded:            {mlp_masked:.3f}  ppl {math.exp(mlp_masked):.2f}")
    print(f"  RNN, PAD excluded:            {rnn_loss:.3f}  ppl {math.exp(rnn_loss):.2f}")
