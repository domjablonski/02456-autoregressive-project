import importlib.util, torch
from torch.utils.data import DataLoader
spec = importlib.util.spec_from_file_location("cf", "class-free.py")
cf = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf)
m = cf.RNN(cf.vocab_size, cf.num_families, cf.embed_dim, cf.hidden_dim, cf.num_layers).to(cf.device)
m.load_state_dict(torch.load("results/checkpoints/rnn_cfg.pt", map_location=cf.device, weights_only=True))
m.eval()
for split in ["val", "test"]:
    ds = cf.ProteinDataset(path_to_data=cf.ROOT / "data" / f"{split}.csv")
    assert ds.fam_to_idx == cf.train_data.fam_to_idx, "family mapping differs"
    for name, null in [("true family", False), ("null family", True)]:
        tot = cnt = 0
        with torch.no_grad():
            for x, xs, lab, _ in DataLoader(ds, batch_size=64):
                x, xs, lab = x.to(cf.device), xs.to(cf.device), lab.to(cf.device)
                if null:
                    lab = torch.full_like(lab, cf.NULL_FAMILY)
                l, n = cf.sequence_loss(m(x, lab)[0], xs)
                tot += l.item(); cnt += n.item()
        print(split, name, round(tot / cnt, 3))
