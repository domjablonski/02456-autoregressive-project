"""Classifier guidance: steer the RNN towards one protein family.

    uv run python src/protgen/models/guided.py --family HPr

Needs results/checkpoints/rnn.pt and classifier.pt, and classifier_judge.pt to score the result.

At each step the next token is sampled from

    log p(token | prefix, family) = log p(token | prefix) + lambda * log p(family | prefix + token)

the generator's own opinion plus the predictor's, weighted by lambda. lambda = 0 is the unguided
RNN, and a large lambda follows the predictor wherever it leads - including off the manifold of
plausible proteins, which is the trade-off the sweep is there to show.

Both models are GRUs, so each step costs one forward of one token: the prefix lives in the hidden
state, and the 23 candidate tokens go through the predictor as one batch rather than 23 passes
over the whole prefix.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from protgen.data import FAMILIES, ProteinDataset
import protgen.models.classifier as clf_module
import protgen.models.rnn as rnn_module

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = ROOT / "results" / "checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD, BOS, EOS = 0, 1, 2
DEFAULT_LAMBDAS = (0.0, 1.0, 5.0, 20.0)
num_samples = 256


def load_generator(data):
    model = rnn_module.RNN(data.vocab_size, rnn_module.embed_dim, rnn_module.hidden_dim,
                           rnn_module.num_layers).to(device)
    model.load_state_dict(torch.load(CHECKPOINTS / "rnn.pt", map_location=device,
                                     weights_only=True))
    return model.eval()


def load_classifier(name, data):
    model = clf_module.Classifier(data.vocab_size, clf_module.embed_dim, clf_module.hidden_dim,
                                  clf_module.num_layers, len(FAMILIES)).to(device)
    model.load_state_dict(torch.load(CHECKPOINTS / f"{name}.pt", map_location=device,
                                     weights_only=True))
    return model.eval()


@torch.no_grad()
def guided_generate(generator, classifier, data, family, lam, max_len=127):
    candidates = torch.arange(data.vocab_size, device=device).unsqueeze(1)   # (V, 1)
    tokens = [BOS]

    gen_hidden = None
    # The predictor reads the same token stream as the generator, <BOS> first.
    _, clf_hidden = classifier.step(torch.tensor([[BOS]], device=device), None)

    for _ in range(max_len):
        last = torch.tensor([[tokens[-1]]], device=device)
        gen_logits, gen_hidden = generator(last, gen_hidden)
        scores = F.log_softmax(gen_logits[0, -1], dim=-1)                    # (V,)

        if lam:
            # One step on from the prefix state, once per candidate token, in a single batch.
            hidden = clf_hidden.repeat(1, data.vocab_size, 1)
            family_logp, _ = classifier.step(candidates, hidden)             # (V, num_classes)
            scores = scores + lam * family_logp[:, family]

        next_token = torch.multinomial(F.softmax(scores, dim=-1), 1).item()
        if next_token == EOS:
            break
        tokens.append(next_token)
        _, clf_hidden = classifier.step(torch.tensor([[next_token]], device=device), clf_hidden)

    return data.decode(tokens)


@torch.no_grad()
def judge_families(judge, data, sequences):
    """The independent judge's family for each sequence, as a class index."""
    labels = []
    for sequence in sequences:
        x = data.encode(sequence).unsqueeze(0).to(device)
        logits, _ = judge(x)
        labels.append(logits[0, len(sequence) + 1].argmax().item())
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=FAMILIES[1], choices=FAMILIES,
                        help="the family to steer towards")
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--num-samples", type=int, default=num_samples)
    args = parser.parse_args()

    torch.manual_seed(0)
    data = ProteinDataset(path_to_data=ROOT / "data" / "train.csv")
    family = data.fam_to_idx[args.family]
    print(f"guiding towards {args.family} (class {family}) on {device}")

    generator = load_generator(data)
    classifier = load_classifier("classifier", data)
    judge = load_classifier("classifier_judge", data)

    rows, summary = [], []
    for lam in args.lambdas:
        setting = f"lambda{lam:g}"
        sequences = [guided_generate(generator, classifier, data, family, lam)
                     for _ in range(args.num_samples)]
        labels = judge_families(judge, data, sequences)

        on_target = sum(label == family for label in labels) / len(labels)
        mean_length = sum(len(sequence) for sequence in sequences) / len(sequences)
        print(f"{setting}: {on_target:.3f} judged {args.family}, mean length {mean_length:.1f}")

        rows += [{"setting": setting, "sequence": sequence} for sequence in sequences]
        summary.append({"setting": setting, "lambda": lam, "target": args.family,
                        "frac_on_target": on_target, "mean_length": mean_length})

    # One CSV with a `setting` column: eval.py groups by it, so the sweep is one row per lambda.
    out_path = ROOT / "results" / "sequences" / f"guided_{args.family}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    summary_path = ROOT / "results" / "guided" / f"family_control_{args.family}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print(f"\nwrote {len(rows)} sequences to {out_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
