"""A miniature version of the Chinchilla experiment.

Training cost grows with parameters times tokens processed. With a fixed compute
budget, a bigger model can therefore only see less data. This trains a range of
sizes on several budgets, finds the best size for each, and fits the Chinchilla
formula for loss:

    loss = E + A / N^alpha + B / D^beta

N is parameters, D is training tokens, and E is the loss no amount of either removes.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from tinygpt.data import load_splits
from tinygpt.model import GPT, GPTConfig
from tinygpt.plots import INK_SECONDARY, SERIES, plt, style_axes
from tinygpt.training import TrainConfig, train

# (layers, heads, embedding size), from smallest to largest
SIZES = [(1, 2, 32), (2, 2, 48), (3, 4, 64), (3, 4, 96), (4, 4, 128)]
BUDGETS = [2.5e11, 5e11, 1e12]
BLOCK_SIZE = 64
BATCH_SIZE = 32


def iterations_for(budget, parameters, batch_size=BATCH_SIZE, block_size=BLOCK_SIZE):
    """Turn a budget of parameters times tokens into a number of training steps."""
    return max(1, round(budget / (parameters * batch_size * block_size)))


def fit_scaling_law(points):
    """Fit the formula to (N, D, loss) points. Returns its parameters and R squared.

    For fixed exponents the formula is linear in E, A and B, so least squares solves
    those directly, and a grid search covers the exponents. A, B and E must be
    non-negative for the formula to make physical sense.
    """
    n = torch.tensor([p[0] for p in points], dtype=torch.float64)
    d = torch.tensor([p[1] for p in points], dtype=torch.float64)
    y = torch.tensor([p[2] for p in points], dtype=torch.float64)
    total_variation = ((y - y.mean()) ** 2).sum().item()

    best = None
    grid = [i / 20 for i in range(1, 41)]  # exponents from 0.05 to 2.0
    for alpha in grid:
        for beta in grid:
            design = torch.stack([torch.ones_like(n), n**-alpha, d**-beta], dim=1)
            solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
            if (solution < 0).any():
                continue
            error = ((design @ solution - y) ** 2).sum().item()
            if best is None or error < best["error"]:
                best = {
                    "E": solution[0].item(),
                    "A": solution[1].item(),
                    "B": solution[2].item(),
                    "alpha": alpha,
                    "beta": beta,
                    "error": error,
                }
    if best is None:
        raise ValueError("no valid fit with non-negative coefficients")
    best["r_squared"] = 1 - best.pop("error") / total_variation if total_variation else 1.0
    return best


def predict_loss(fit, parameters, tokens):
    return fit["E"] + fit["A"] / parameters ** fit["alpha"] + fit["B"] / tokens ** fit["beta"]


def optimal_parameters(fit, budget):
    """Find the size with the lowest predicted loss when tokens = budget / parameters."""
    candidates = [10 ** (3 + i * 0.01) for i in range(501)]  # from 1,000 to 100 million
    return min(candidates, key=lambda n: predict_loss(fit, n, budget / n))


def plot(results, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    budgets = sorted({row["budget"] for row in results})
    for budget, color in zip(budgets, SERIES):
        rows = sorted((r for r in results if r["budget"] == budget), key=lambda r: r["parameters"])
        x = [r["parameters"] for r in rows]
        y = [r["val_loss"] for r in rows]
        label = f"Budget {budget:.2g}"
        ax.plot(x, y, color=color, lw=1.6, marker="o", ms=4, label=label)
        best = min(rows, key=lambda r: r["val_loss"])
        ax.scatter([best["parameters"]], [best["val_loss"]], s=90, facecolors="none", edgecolors=color, linewidths=1.5, zorder=3)
        ax.annotate(label, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK_SECONDARY, fontsize=9.5, annotation_clip=False)
    ax.set_xscale("log")
    style_axes(ax)
    ax.set(xlabel="Model parameters (log scale)", ylabel="Best validation loss (lower is better)")
    ax.set_title("With compute fixed, the best model size grows with the budget (circle = best)")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="../04-tiny-gpt/data")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--budget-scale", type=float, default=1.0, help="multiply every budget, for quick trials")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not (Path(args.data) / "train.txt").exists():
        parser.error(f"{args.data}/train.txt not found. Run prepare_data.py in Project 4 first.")

    tokenizer, train_ids, val_ids = load_splits(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for budget in (b * args.budget_scale for b in BUDGETS):
        for layers, heads, embd in SIZES:
            config = GPTConfig(tokenizer.vocab_size, BLOCK_SIZE, layers, heads, embd, dropout=0.0)
            torch.manual_seed(args.seed)
            model = GPT(config)
            parameters = model.num_parameters()
            iterations = iterations_for(budget, parameters)
            train_config = TrainConfig(
                max_iters=iterations,
                batch_size=BATCH_SIZE,
                warmup_iters=min(100, max(1, iterations // 10)),
                eval_every=max(1, iterations // 5),
                eval_iters=50,
                seed=args.seed,
            )
            history = train(model, train_ids, val_ids, train_config, log=None)
            results.append(
                {
                    "budget": budget,
                    "layers": layers,
                    "embd": embd,
                    "parameters": parameters,
                    "tokens": iterations * BATCH_SIZE * BLOCK_SIZE,
                    "iterations": iterations,
                    "val_loss": min(row["val_loss"] for row in history),
                    "seconds": history[-1]["seconds"],
                }
            )
            row = results[-1]
            print(
                f"budget {budget:.2g}  {parameters:>8,} params  {iterations:>6,} iters  val {row['val_loss']:.4f}  {row['seconds']:.0f}s",
                flush=True,
            )
            (out / "results.json").write_text(json.dumps(results, indent=1))

    fit = fit_scaling_law([(r["parameters"], r["tokens"], r["val_loss"]) for r in results])
    lines = [
        "# Scaling law results",
        "",
        "| Budget (parameters x tokens) | Parameters | Training iterations | Tokens seen | Best validation loss | Formula predicts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        predicted = predict_loss(fit, r["parameters"], r["tokens"])
        lines.append(
            f"| {r['budget']:.2g} | {r['parameters']:,} | {r['iterations']:,} | {r['tokens']:,} | {r['val_loss']:.4f} | {predicted:.4f} |"
        )
    lines += [
        "",
        "## Fitted formula",
        "",
        f"loss = {fit['E']:.3f} + {fit['A']:.3f} / N^{fit['alpha']:.2f} + {fit['B']:.3f} / D^{fit['beta']:.2f}",
        "",
        f"R squared: {fit['r_squared']:.3f}, from {len(results)} runs with one seed each.",
        "",
        "| Budget | Best size measured | Best size according to the formula |",
        "| --- | --- | --- |",
    ]
    for budget in sorted({r["budget"] for r in results}):
        best = min((r for r in results if r["budget"] == budget), key=lambda r: r["val_loss"])
        lines.append(f"| {budget:.2g} | {best['parameters']:,} | {optimal_parameters(fit, budget):,.0f} |")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "fit.json").write_text(json.dumps(fit, indent=1))
    plot(results, out / "scaling.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
