# Project 10: Mini scaling-law experiment (stretch)

A miniature version of the Chinchilla experiment. Training cost grows with parameters
times tokens processed, so with a fixed compute budget, a bigger model can only see
less data. This project asks which split works best.

It trains five Tiny GPT sizes on three compute budgets, then fits the Chinchilla formula
for loss:

```text
loss = E + A / N^alpha + B / D^beta
```

N is parameters, D is training tokens, and E is the loss that no amount of either removes.

**Limits:** these models are far smaller than the ones in the Chinchilla paper, and the
fit uses 15 runs with one seed each. It shows the idea, but don't apply its numbers to
real large models.

## Results

**Not available yet.** The experiment hasn't finished. When it has, results will be in
`outputs\results.md` and `outputs\scaling.png`.

## Run it

This uses the data from Project 4, so run its `prepare_data.py` first.

```powershell
uv sync
uv run python scaling.py
uv run pytest
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--budget-scale` | `1.0` | Multiplies every budget, for quick trials |
| `--seed` | `0` | Random seed |

## Files

| File | What it does |
| --- | --- |
| `scaling.py` | Runs the experiment, fits the formula and plots the results |
| `tests/` | Includes a check that the fit recovers a formula with known values |
