import pandas as pd, os
from pathlib import Path
src = Path("results/eval/rnn_cfg")
d = pd.read_csv(src / "eval.csv")
for w, g in d.groupby("guidance_w"):
    out = Path(f"results/eval/rnn_cfg_w{int(w)}")
    (out / "pdb").mkdir(parents=True, exist_ok=True)
    g.to_csv(out / "eval.csv", index=False)
    for p in g["pdb"].dropna():
        name = os.path.basename(p)
        link = out / "pdb" / name
        if not link.exists():
            link.symlink_to((src / "pdb" / name).resolve())
    print(out, len(g))
