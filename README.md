# LightGCN

This repository is a cleaned LightGCN-only recommendation project built on top of a SelfRec-style training pipeline.

## Scope

- Model: `LightGCN`
- Dataset: `iFashion_UB`
- Task: top-N recommendation on the user-bundle graph

All BOD-specific code, configs, and search specs are removed from the active project scope.

## Requirements

```text
numba==0.53.1
numpy==1.20.3
scipy==1.6.2
torch>=1.7.0
```

## Project Layout

```text
conf/LightGCN.conf          Main LightGCN configuration
model/graph/LightGCN.py     Model definition and training loop
core_runtime.py             Shared runtime, data, metrics, and evaluation helpers
SELFRec.py                  Dynamic model loading entry
main.py                     Interactive launcher
scripts/auto_search.py      Hyperparameter search runner
scripts/search_specs/       LightGCN search specifications
dataset/iFashion_UB/        Train/test graph data
results/                    Experiment outputs
```

## Usage

1. Adjust `conf/LightGCN.conf` if needed.
2. Run `python main.py`.
3. Enter `LightGCN`, or press Enter to use the default.

## Hyperparameter Search

Example:

```bash
python scripts/auto_search.py --spec scripts/search_specs/lightgcn_stage1_learnrate.json
```

The search runner generates temporary configs, launches trials, captures logs, and writes summaries under `results/search_runs/`.

## Notes

- Dataset files use the graph format `user_id item_id weight`
- Best historical LightGCN runs are kept in `results/experiment_results.txt`
- The LightGCN trainer exports user and item embeddings after the best checkpoint is selected

## Acknowledgement

The implementation is based on the open-source recommendation library [SelfRec](https://github.com/Coder-Yu/SELFRec).
