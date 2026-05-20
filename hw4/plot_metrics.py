import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def load_metrics(path: str) -> Dict[str, List[float]]:
    steps = []
    train_loss = []
    val_loss = []
    bleu = []
    chrf = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            steps.append(record["step"])
            train_loss.append(record["train_loss"])
            val_loss.append(record["val_loss"])
            bleu.append(record["bleu"])
            chrf.append(record["chrf"])
    return {
        "step": steps,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "bleu": bleu,
        "chrf": chrf,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
                        required=True,
                        help="comma-separated metrics.jsonl paths")
    parser.add_argument("--labels",
                        required=True,
                        help="comma-separated labels")
    parser.add_argument("--out", default=os.path.join("plots", "metrics.png"))
    args = parser.parse_args()

    input_paths = [p.strip() for p in args.inputs.split(",") if p.strip()]
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    if len(input_paths) != len(labels):
        raise ValueError("inputs and labels must have the same length")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_train, ax_val = axes[0]
    ax_bleu, ax_chrf = axes[1]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', "#000000"]
    for path, label, color in zip(input_paths, labels, colors):
        data = load_metrics(path)
        ax_train.plot(data["step"], data["train_loss"], label=label, color=color)
        ax_val.plot(data["step"], data["val_loss"], label=label, color=color)
        ax_bleu.plot(data["step"], data["bleu"], label=label, color=color)
        ax_chrf.plot(data["step"], data["chrf"], label=label, color=color)

    ax_train.set_title("Train Loss")
    ax_val.set_title("Validation Loss")
    ax_bleu.set_title("BLEU")
    ax_chrf.set_title("chrF")

    for ax in axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("step")
        ax.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    # fig.savefig(args.out, dpi=150)
    plt.savefig(args.out, dpi=300)



if __name__ == "__main__":
    main()
