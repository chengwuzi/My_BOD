from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLE_SPEC = {
    "search_name": "bod_stage1",
    "base_config": "conf/BOD.conf",
    "output_dir": "results/search_runs/bod_stage1",
    "resume": True,
    "max_attempts": 3,
    "timeout_sec": None,
    "fixed_overrides": {
        "num.max.epoch": 20,
        "batch_size": 256
    },
    "grid": {
        "GM_AU.-weight_uniformity": [0.0, 0.05, 0.1],
        "GM_AU.-generator_lr": [0.0001, 0.0005]
    },
    "trials": [
        {
            "name": "manual_alignment_boost",
            "overrides": {
                "GM_AU.-weight_alignment": 2,
                "GM_AU.-weight_bpr": 1
            }
        }
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated hyperparameter searches without changing the original config files."
    )
    parser.add_argument("--spec", type=str, help="Path to a JSON search spec.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand trials and generate temporary configs without launching training.",
    )
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Print an example JSON search spec and exit.",
    )
    parser.add_argument(
        "--run-trial",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--result-json",
        type=str,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def to_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def safe_name(name: str) -> str:
    cleaned = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "trial"


def value_to_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def read_conf_file(path: Path) -> OrderedDict[str, str]:
    config: OrderedDict[str, str] = OrderedDict()
    with path.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            config[key] = value
    return config


def write_conf_file(path: Path, config: OrderedDict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")


def parse_option_string(option_string: str) -> tuple[list[str], list[tuple[str, str]]]:
    tokens = option_string.strip().split()
    prefix_tokens: list[str] = []
    option_pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens) and not is_option_token(tokens[index]):
        prefix_tokens.append(tokens[index])
        index += 1
    while index < len(tokens):
        token = tokens[index]
        if not is_option_token(token):
            index += 1
            continue
        key = token
        index += 1
        values: list[str] = []
        while index < len(tokens) and not is_option_token(tokens[index]):
            values.append(tokens[index])
            index += 1
        option_pairs.append((key, " ".join(values)))
    return prefix_tokens, option_pairs


def build_option_string(prefix_tokens: list[str], option_pairs: list[tuple[str, str]]) -> str:
    tokens = list(prefix_tokens)
    for key, value in option_pairs:
        tokens.append(key)
        if value:
            tokens.extend(value.split())
    return " ".join(tokens).strip()


def is_option_token(token: str) -> bool:
    return (token.startswith("-") or token.startswith("--")) and not token[1:].isdigit()


def set_option_value(option_string: str, option_key: str, option_value: str) -> str:
    prefix_tokens, option_pairs = parse_option_string(option_string)
    updated = False
    new_pairs: list[tuple[str, str]] = []
    for key, value in option_pairs:
        if key == option_key:
            new_pairs.append((key, option_value))
            updated = True
        else:
            new_pairs.append((key, value))
    if not updated:
        new_pairs.append((option_key, option_value))
    return build_option_string(prefix_tokens, new_pairs)


def apply_overrides(
    base_config: OrderedDict[str, str],
    overrides: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> OrderedDict[str, str]:
    config = OrderedDict(base_config)
    for key, value in overrides.items():
        value_str = value_to_string(value)
        if ".-" in key:
            main_key, option_suffix = key.split('.-', 1)
            option_key = '-' + option_suffix
            current = config.get(main_key, "")
            config[main_key] = set_option_value(current, option_key, value_str)
        else:
            config[key] = value_str
    if metadata:
        for key, value in metadata.items():
            config[key] = value_to_string(value)
    return config


def signature_for_overrides(overrides: dict[str, Any]) -> str:
    normalized = json.dumps(overrides, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def build_trials(spec: dict[str, Any]) -> list[dict[str, Any]]:
    fixed_overrides = dict(spec.get("fixed_overrides", {}))
    trials: list[dict[str, Any]] = []

    grid = spec.get("grid", {})
    if grid:
        grid_keys = list(grid.keys())
        grid_values = [grid[key] for key in grid_keys]
        for idx, combo in enumerate(itertools.product(*grid_values), start=1):
            combo_overrides = dict(fixed_overrides)
            combo_label_parts = []
            for key, value in zip(grid_keys, combo):
                combo_overrides[key] = value
                combo_label_parts.append(f"{key}={value}")
            trials.append(
                {
                    "source": "grid",
                    "index": idx,
                    "name": f"grid_{idx:04d}",
                    "display_name": " | ".join(combo_label_parts),
                    "overrides": combo_overrides,
                }
            )

    for idx, trial_spec in enumerate(spec.get("trials", []), start=1):
        trial_overrides = dict(fixed_overrides)
        trial_overrides.update(trial_spec.get("overrides", {}))
        trials.append(
            {
                "source": "manual",
                "index": idx,
                "name": trial_spec.get("name", f"manual_{idx:04d}"),
                "display_name": trial_spec.get("name", f"manual_{idx:04d}"),
                "overrides": trial_overrides,
            }
        )

    return trials


def load_completed_signatures(summary_jsonl: Path) -> set[str]:
    signatures: set[str] = set()
    if not summary_jsonl.exists():
        return signatures
    with summary_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            signature = record.get("signature")
            status = record.get("status")
            if signature and status in {"success", "failed", "timeout", "skipped"}:
                signatures.add(signature)
    return signatures


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def rebuild_summary_csv(summary_jsonl: Path, summary_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    if summary_jsonl.exists():
        with summary_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    fieldnames = [
        "search_name",
        "trial_id",
        "trial_name",
        "status",
        "model",
        "signature",
        "best_epoch",
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "duration_sec",
        "returncode",
        "config_path",
        "log_path",
        "error_type",
        "error",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "search_name": row.get("search_name"),
                    "trial_id": row.get("trial_id"),
                    "trial_name": row.get("trial_name"),
                    "status": row.get("status"),
                    "model": row.get("model"),
                    "signature": row.get("signature"),
                    "best_epoch": row.get("best_epoch"),
                    "recall@10": row.get("metrics", {}).get("10", {}).get("Recall"),
                    "ndcg@10": row.get("metrics", {}).get("10", {}).get("NDCG"),
                    "recall@20": row.get("metrics", {}).get("20", {}).get("Recall"),
                    "ndcg@20": row.get("metrics", {}).get("20", {}).get("NDCG"),
                    "duration_sec": row.get("duration_sec"),
                    "returncode": row.get("returncode"),
                    "config_path": row.get("config_path"),
                    "log_path": row.get("log_path"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
            )


def run_trial_subprocess(
    config_path: Path,
    result_json: Path,
    log_path: Path,
    timeout_sec: int | None,
    attempt_index: int = 1,
    max_attempts: int = 1,
) -> tuple[int | None, dict[str, Any]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-trial",
        "--config",
        str(config_path),
        "--result-json",
        str(result_json),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if result_json.exists():
        result_json.unlink()
    start_time = time.time()
    log_mode = "w" if attempt_index == 1 else "a"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        banner = (
            f"\n{'=' * 80}\n"
            f"Attempt {attempt_index}/{max_attempts}\n"
            f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n"
            f"Config: {config_path}\n"
            f"{'=' * 80}\n"
        )
        log_file.write(banner)
        log_file.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec,
            )
            returncode = completed.returncode
            duration_sec = time.time() - start_time
        except subprocess.TimeoutExpired as exc:
            returncode = None
            duration_sec = time.time() - start_time
            timeout_record = {
                "status": "timeout",
                "returncode": None,
                "duration_sec": round(duration_sec, 3),
                "error_type": "TimeoutExpired",
                "error": str(exc),
                "attempt": attempt_index,
            }
            dump_json(result_json, timeout_record)
            return returncode, timeout_record

    if result_json.exists():
        result_record = load_json(result_json)
    else:
        result_record = {
            "status": "failed" if returncode else "success",
            "returncode": returncode,
            "duration_sec": round(duration_sec, 3),
            "error_type": "MissingResultFile",
            "error": "Child process exited without writing result json.",
        }
        dump_json(result_json, result_record)

    result_record["returncode"] = returncode
    result_record["duration_sec"] = round(duration_sec, 3)
    result_record["attempt"] = attempt_index
    dump_json(result_json, result_record)
    return returncode, result_record


def execute_trial(config_path: Path, result_json: Path) -> int:
    from SELFRec import SELFRec
    from core_runtime import ModelConf

    start_time = time.time()
    try:
        conf = ModelConf(str(config_path))
        loader = SELFRec(conf)
        module = importlib.import_module(f"model.{conf['model.type']}.{conf['model.name']}")
        model_cls = getattr(module, conf["model.name"])
        model = model_cls(conf, loader.training_data, loader.test_data, **loader.kwargs)
        model.execute()
        best = getattr(model, "bestPerformance", None)
        metrics: dict[str, dict[str, float]] = {}
        best_epoch = None
        if best is not None:
            best_epoch = best.get("epoch")
            raw_metrics = best.get("metrics", {})
            for top_n, metric_dict in raw_metrics.items():
                metrics[str(top_n)] = metric_dict
        payload = {
            "status": "success",
            "model": conf["model.name"],
            "config_path": str(config_path),
            "best_epoch": best_epoch,
            "metrics": metrics,
            "duration_sec": round(time.time() - start_time, 3),
        }
        dump_json(result_json, payload)
        return 0
    except BaseException as exc:
        payload = {
            "status": "failed",
            "config_path": str(config_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "duration_sec": round(time.time() - start_time, 3),
        }
        dump_json(result_json, payload)
        return 1


def print_example_spec() -> None:
    print(json.dumps(EXAMPLE_SPEC, ensure_ascii=False, indent=2))


def run_search(spec_path: Path, dry_run: bool) -> int:
    spec = load_json(spec_path)
    search_name = spec.get("search_name", spec_path.stem)
    base_config_path = to_repo_path(spec["base_config"])
    output_dir = to_repo_path(spec.get("output_dir", f"results/search_runs/{search_name}"))
    summary_jsonl = output_dir / "summary.jsonl"
    summary_csv = output_dir / "summary.csv"
    configs_dir = output_dir / "configs"
    logs_dir = output_dir / "logs"
    trial_results_dir = output_dir / "trial_results"
    timeout_sec = spec.get("timeout_sec")
    resume = bool(spec.get("resume", True))
    max_attempts = int(spec.get("max_attempts", 3))
    if max_attempts < 1:
        raise ValueError("'max_attempts' must be at least 1.")

    base_config = read_conf_file(base_config_path)
    trials = build_trials(spec)
    if not trials:
        raise ValueError("No trials were produced. Please provide 'grid' and/or 'trials' in the search spec.")

    completed_signatures = load_completed_signatures(summary_jsonl) if resume else set()
    print(f"Search Name: {search_name}")
    print(f"Base Config: {base_config_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Trials Planned: {len(trials)}")
    print(f"Resume Mode: {resume}")
    print(f"Max Attempts Per Trial: {max_attempts}")
    print(f"Dry Run: {dry_run}")
    print("-" * 80)

    for idx, trial in enumerate(trials, start=1):
        overrides = trial["overrides"]
        signature = signature_for_overrides(overrides)
        trial_id = f"trial_{idx:04d}"
        trial_name = safe_name(trial["name"])
        config_path = configs_dir / f"{trial_id}_{trial_name}.conf"
        log_path = logs_dir / f"{trial_id}_{trial_name}.log"
        result_json = trial_results_dir / f"{trial_id}_{trial_name}.json"
        metadata = {
            "search.name": search_name,
            "trial.id": trial_id,
            "trial.name": trial_name,
            "trial.signature": signature,
        }
        config = apply_overrides(base_config, overrides, metadata=metadata)
        write_conf_file(config_path, config)

        print(f"[{idx}/{len(trials)}] {trial_id} | {trial['display_name']}")
        print(f"  signature: {signature}")
        print(f"  config: {config_path}")

        if resume and signature in completed_signatures:
            print("  skipped: already present in previous summary")
            continue

        if dry_run:
            print("  dry-run: config generated, training not launched")
            continue

        attempt_records: list[dict[str, Any]] = []
        returncode: int | None = None
        trial_result: dict[str, Any] = {}
        for attempt_index in range(1, max_attempts + 1):
            print(f"  attempt {attempt_index}/{max_attempts}")
            returncode, trial_result = run_trial_subprocess(
                config_path,
                result_json,
                log_path,
                timeout_sec,
                attempt_index=attempt_index,
                max_attempts=max_attempts,
            )
            attempt_records.append(
                {
                    "attempt": attempt_index,
                    "status": trial_result.get("status", "failed"),
                    "returncode": returncode,
                    "duration_sec": trial_result.get("duration_sec"),
                    "error_type": trial_result.get("error_type"),
                    "error": trial_result.get("error"),
                }
            )
            if trial_result.get("status") != "failed":
                break
            if attempt_index < max_attempts:
                print(
                    "  failed attempt "
                    f"{attempt_index}/{max_attempts}: "
                    f"{trial_result.get('error_type')} {trial_result.get('error')}"
                )
                print("  retrying...")

        record = {
            "search_name": search_name,
            "trial_id": trial_id,
            "trial_name": trial_name,
            "status": trial_result.get("status", "failed"),
            "signature": signature,
            "model": trial_result.get("model", config.get("model.name")),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "result_json": str(result_json),
            "returncode": returncode,
            "duration_sec": trial_result.get("duration_sec"),
            "best_epoch": trial_result.get("best_epoch"),
            "metrics": trial_result.get("metrics", {}),
            "error_type": trial_result.get("error_type"),
            "error": trial_result.get("error"),
            "overrides": overrides,
            "attempt_count": len(attempt_records),
            "attempts": attempt_records,
        }
        append_jsonl(summary_jsonl, record)
        completed_signatures.add(signature)
        rebuild_summary_csv(summary_jsonl, summary_csv)

        if record["status"] == "success":
            metrics = record["metrics"]
            recall20 = metrics.get("20", {}).get("Recall")
            ndcg20 = metrics.get("20", {}).get("NDCG")
            print(
                "  success: "
                f"attempts={record['attempt_count']} "
                f"best_epoch={record['best_epoch']} "
                f"recall@20={recall20} ndcg@20={ndcg20}"
            )
        else:
            print(
                f"  {record['status']}: "
                f"attempts={record['attempt_count']} "
                f"{record.get('error_type')} {record.get('error')}"
            )

    if not dry_run:
        rebuild_summary_csv(summary_jsonl, summary_csv)
        print("-" * 80)
        print(f"Search finished. Summary JSONL: {summary_jsonl}")
        print(f"Search finished. Summary CSV:   {summary_csv}")
    return 0


def main() -> int:
    args = parse_args()
    if args.print_example:
        print_example_spec()
        return 0
    if args.run_trial:
        if not args.config or not args.result_json:
            raise SystemExit("--run-trial requires --config and --result-json")
        return execute_trial(Path(args.config), Path(args.result_json))
    if not args.spec:
        raise SystemExit("Please provide --spec path/to/search.json or use --print-example")
    return run_search(to_repo_path(args.spec), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())


