"""Baseline -> learning rate -> regularization, selected exclusively on validation."""
from __future__ import annotations

import importlib
import math
import shutil
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from typing import Any

from scripts import auto_search as search


DEFAULT_MULTIPLIERS = [0.1, 0.25, 0.5, 2, 4, 10]
METADATA_KEYS = {"search.name", "trial.id", "trial.name", "trial.signature"}


def candidate_values(initial: str, multipliers: list[Any]) -> list[str]:
    center = Decimal(initial)
    factors = [Decimal(str(value)) for value in multipliers]
    if not center.is_finite() or center <= 0:
        raise ValueError("Initial learning rate and regularization must be positive and finite.")
    if (len(factors) != 6 or any(not f.is_finite() or f <= 0 for f in factors)
            or len(set(factors)) != 6 or sum(f < 1 for f in factors) != 3
            or sum(f > 1 for f in factors) != 3):
        raise ValueError("Provide six distinct positive multipliers: three below 1 and three above 1.")
    return [format(center * factor, "f") for factor in factors]


def metric_key(record: dict) -> tuple[float, ...]:
    metrics = record.get("metrics", {})
    if not metrics or "20" not in metrics:
        raise ValueError("Staged selection requires validation NDCG/Recall at Top-20.")
    values = tuple(float(metrics[top_n][name])
                   for top_n in sorted(metrics, key=int, reverse=True)
                   for name in ("NDCG", "Recall"))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("A trial produced non-finite validation metrics.")
    return values


def select_best(records: list[dict]) -> dict:
    if not records or any(record.get("status") != "success" for record in records):
        raise RuntimeError("All planned trials must succeed before selecting the next stage.")
    # max keeps the first record on exact ties, so the incumbent is retained.
    return max(records, key=metric_key)


def clean_config(record: dict) -> OrderedDict[str, str]:
    return OrderedDict((key, value) for key, value in record["config"].items() if key not in METADATA_KEYS)


def evaluate_test(config_path: Path, embedding_paths: dict[str, str]) -> dict:
    """Load the validation-selected embeddings; never train or select on test data."""
    import torch
    from core_runtime import FileIO, ModelConf, ranking_evaluation

    conf = ModelConf(str(config_path))
    conf.config["test.set"] = conf["final.test.set"]
    conf.config["evaluation.split"] = "test"
    training = FileIO.load_data_set(conf["training.set"], conf["model.type"])
    test = FileIO.load_data_set(conf["test.set"], conf["model.type"])
    if not test:
        raise ValueError("Final test data is empty.")
    module = importlib.import_module(f"model.{conf['model.type']}.{conf['model.name']}")
    model = getattr(module, conf["model.name"])(conf, training, test)
    try:
        for user, bundle, _ in test:
            if user not in model.data.user or bundle not in model.data.item:
                raise ValueError("Final test contains entities missing from the training graph.")
        user_vectors = torch.load(embedding_paths["user"], map_location="cpu", weights_only=True)
        bundle_vectors = torch.load(embedding_paths["bundle"], map_location="cpu", weights_only=True)
        model.user_emb = torch.stack([user_vectors[model.data.id2user[i]] for i in range(model.data.user_num)])
        model.item_emb = torch.stack([bundle_vectors[model.data.id2item[i]] for i in range(model.data.item_num)])
        if not torch.isfinite(model.user_emb).all() or not torch.isfinite(model.item_emb).all():
            raise ValueError("Selected embeddings contain non-finite values.")
        metrics = ranking_evaluation(model.data.test_set, model.test(), model.topN)
        return {"status": "success", "split": "test", "metrics": metrics,
                "users": len(model.data.test_set), "interactions": len(test)}
    finally:
        # Log owns a FileHandler; release it after this one-off evaluation, also on Windows.
        for handler in list(model.model_log.logger.handlers):
            handler.close()
            model.model_log.logger.removeHandler(handler)


def dataset_plan(config_path: Path, multipliers: list[Any], check_files: bool) -> dict:
    config = search.read_conf_file(config_path)
    if config.get("evaluation.split") != "validation" or "final.test.set" not in config:
        raise ValueError(f"{config_path}: require evaluation.split=validation and final.test.set.")
    if config.get("model.name") != "LightGCN" or config.get("model.type") != "graph":
        raise ValueError("Staged search currently supports graph LightGCN only.")
    _, options = search.parse_option_string(config["item.ranking"])
    if dict(options).get("-topN") != "10,20":
        raise ValueError("Staged search uses item.ranking=-topN 10,20 for consistent selection.")
    paths = {key: search.to_repo_path(config[key]) for key in ("training.set", "test.set", "final.test.set")}
    if len(set(paths.values())) != 3:
        raise ValueError("Training, validation and final test must be different files.")
    if check_files:
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Prepare separate train/valid/test first: missing {path}")
        check_partitions(paths)
    return {
        "dataset": config["dataset.name"], "base_config": str(config_path),
        "initial": {key: config[key] for key in ("learnRate", "reg.lambda")},
        "learning_rates": candidate_values(config["learnRate"], multipliers),
        "regularization_values": candidate_values(config["reg.lambda"], multipliers),
        "regularization_stage_learning_rate": "best of baseline and six learning-rate trials",
        "training_trials": 13, "final_test_evaluations": 1,
    }


def check_partitions(paths: dict[str, Path]) -> None:
    def pairs(path: Path) -> set[tuple[str, str]]:
        entries = set()
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 3 or not math.isfinite(float(parts[2])) or float(parts[2]) <= 0:
                    raise ValueError(f"{path}:{line_no}: require user bundle positive_weight.")
                entries.add((parts[0], parts[1]))
        if not entries:
            raise ValueError(f"{path}: empty split.")
        return entries

    splits = {key: pairs(path) for key, path in paths.items()}
    train = splits["training.set"]
    users, bundles = {u for u, _ in train}, {b for _, b in train}
    keys = list(splits)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if splits[left] & splits[right]:
                raise ValueError(f"Interaction overlap between {left} and {right}.")
    for key in ("test.set", "final.test.set"):
        if any(u not in users or b not in bundles for u, b in splits[key]):
            raise ValueError(f"{paths[key]} contains cold entities; prepare filtered evaluation splits first.")


def run_stage(name: str, base: OrderedDict[str, str], values: list[str] | None,
              parameter: str | None, output_dir: Path, settings: dict) -> list[dict]:
    stage_dir = output_dir / name
    base_path = stage_dir / "base.conf"
    search.write_conf_file(base_path, base)
    spec = {
        "search_name": f"{base['dataset.name']}_{name}", "base_config": str(base_path),
        "output_dir": str(stage_dir), "resume": settings.get("resume", True),
        "max_attempts": settings.get("max_attempts", 3), "timeout_sec": settings.get("timeout_sec"),
        "isolate_outputs": True,
    }
    if parameter:
        spec["grid"] = {parameter: values}
    else:
        spec["trials"] = [{"name": "initial", "overrides": {}}]
    spec_path = stage_dir / "search.json"
    search.dump_json(spec_path, spec)
    if search.run_search(spec_path, dry_run=False):
        raise RuntimeError(f"{base['dataset.name']}: {name} incomplete; fix failures and rerun to resume.")
    records = search.load_json(stage_dir / "run_results.json")["trials"]
    select_best(records)  # Reject missing/invalid validation metrics as well as failed runs.
    return records


def run_dataset(plan: dict, output_dir: Path, settings: dict) -> dict:
    base = search.read_conf_file(Path(plan["base_config"]))
    initial = run_stage("00_initial", base, None, None, output_dir, settings)[0]
    learning_trials = run_stage("01_learnRate", base, plan["learning_rates"], "learnRate", output_dir, settings)
    best_learning = select_best([initial, *learning_trials])
    second_base = clean_config(best_learning)
    regularization_trials = run_stage("02_reg_lambda", second_base, plan["regularization_values"],
                                      "reg.lambda", output_dir, settings)
    best = select_best([best_learning, *regularization_trials])
    winner_config = clean_config(best)
    config_path = output_dir / "best.conf"
    search.write_conf_file(config_path, winner_config)
    embedding_paths = {}
    for kind in ("user", "bundle"):
        source = Path(best["embedding_paths"][kind])
        target = output_dir / "best_embeddings" / f"{plan['dataset']}-{kind}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        embedding_paths[kind] = str(target)

    result = {
        "dataset": plan["dataset"], "selection_split": "validation",
        "selection_order": ["NDCG@20", "Recall@20", "NDCG@10", "Recall@10"],
        "parameters": {key: winner_config[key] for key in ("learnRate", "reg.lambda")},
        "best_epoch": best["best_epoch"], "validation_metrics": best["metrics"],
        "best_learning_rate_signature": best_learning["signature"], "winning_signature": best["signature"],
        "winning_trial": best["config_path"], "best_config": str(config_path),
        "embedding_paths": embedding_paths,
        "epoch_limit_reached": best["best_epoch"] == int(winner_config["num.max.epoch"]),
    }
    # The final test cannot influence any of the decisions above.
    test_fingerprint = search.signature_for_overrides({
        "winner": best["signature"],
        "test": search.file_digest(search.to_repo_path(winner_config["final.test.set"])),
        "embeddings": {kind: search.file_digest(Path(path)) for kind, path in embedding_paths.items()},
        "evaluator": search.file_digest(Path(__file__)),
    })
    test_result_path = output_dir / "final_test.json"
    previous = search.load_json(test_result_path) if settings.get("resume", True) and test_result_path.exists() else {}
    if previous.get("fingerprint") == test_fingerprint and previous.get("status") == "success":
        test_result = previous
        print(f"{plan['dataset']}: reusing final test evaluation for the same selected embeddings.")
    else:
        test_result = evaluate_test(config_path, embedding_paths)
        test_result["fingerprint"] = test_fingerprint
        search.dump_json(test_result_path, test_result)
    result["test_metrics"] = test_result["metrics"]
    search.dump_json(output_dir / "best_result.json", result)
    print(f"{plan['dataset']}: selected {result['parameters']} at epoch {result['best_epoch']}")
    if result["epoch_limit_reached"]:
        print("  Best epoch reached the configured limit; inspect the curve before claiming convergence.")
    print(f"  Best embeddings: {embedding_paths}")
    return result


def run_staged_search(spec_path: Path, dry_run: bool = False) -> int:
    settings = search.load_json(spec_path)
    output_dir = search.to_repo_path(settings["output_dir"])
    multipliers = settings.get("multipliers", DEFAULT_MULTIPLIERS)
    datasets = settings.get("datasets", [])
    if not datasets:
        raise ValueError("Staged search requires at least one dataset base_config.")
    if int(settings.get("max_attempts", 3)) < 1:
        raise ValueError("max_attempts must be at least 1.")
    plans = [dataset_plan(search.to_repo_path(entry["base_config"]), multipliers, not dry_run)
             for entry in datasets]
    names = [plan["dataset"] for plan in plans]
    if len(set(names)) != len(names):
        raise ValueError("Dataset names must be unique.")
    payload = {"selection_split": "validation", "multipliers": multipliers, "datasets": plans,
               "total_training_trials": 13 * len(plans), "dry_run": dry_run}
    search.dump_json(output_dir / "search_plan.json", payload)
    for plan in plans:
        print(f"{plan['dataset']}: initial + 6 learning rates + 6 regularization values")
        print(f"  learnRate: {plan['learning_rates']}")
        print(f"  reg.lambda: {plan['regularization_values']}")
    if dry_run:
        print(f"Dry run only: {output_dir / 'search_plan.json'}; stage 2 depends on stage 1 results.")
        return 0
    results = []
    for plan in plans:
        # Use a filesystem-safe directory while retaining original dataset IDs in configs.
        dataset_dir = output_dir / search.safe_name(plan["dataset"])
        results.append(run_dataset(plan, dataset_dir, settings))
    search.dump_json(output_dir / "best_results.json", {"datasets": results})
    return 0
