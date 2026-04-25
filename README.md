# LightGCN

This repository is a cleaned LightGCN-only recommendation project built on top of a SelfRec-style training pipeline.

## Scope

- Model: `LightGCN`
- Datasets: `iFashion_UB`, `Youshu`, `NetEase`
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
conf/LightGCN_*.conf        One LightGCN config per dataset
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

1. Adjust the dataset-specific config if needed:
   `conf/LightGCN_iFashion.conf`, `conf/LightGCN_Youshu.conf`, or `conf/LightGCN_NetEase.conf`.
2. Run `python main.py`.
3. Enter the dataset number:
   `1` for `iFashion_UB`, `2` for `Youshu`, `3` for `NetEase`.

## Hyperparameter Search

Example:

```bash
python scripts/auto_search.py --spec scripts/search_specs/lightgcn_stage1_learnrate.json
```

The current bundled search specs target the `iFashion_UB` configuration by default. Add separate specs for `Youshu` and `NetEase` when you start tuning those datasets.

## Notes

- Dataset files use the graph format `user_id item_id weight`
- Best historical LightGCN runs are kept in `results/experiment_results.txt`
- The LightGCN trainer exports user and item embeddings after the best checkpoint is selected

## Acknowledgement

The implementation is based on the open-source recommendation library [SelfRec](https://github.com/Coder-Yu/SELFRec).
