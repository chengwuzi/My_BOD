from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert MultiCBR-style user_bundle train/tune/test splits into "
            "LightGCN graph dataset files."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Dataset directory names under dataset/, for example: Youshu NetEase",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "dataset",
        help="Root directory that contains dataset folders. Defaults to ./dataset",
    )
    parser.add_argument(
        "--weight",
        type=int,
        default=1,
        help="Edge weight written to output graph files. Defaults to 1.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"{path} line {line_no} should contain 2 tab-separated columns, got {len(parts)}: {line!r}"
                )
            user_id, bundle_id = parts
            pairs.append((user_id, bundle_id))
    return pairs


def deduplicate_preserve_order(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique_pairs.append(pair)
    return unique_pairs


def validate_test_pairs(train_pairs: list[tuple[str, str]], test_pairs: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    train_users = {user for user, _ in train_pairs}
    train_items = {item for _, item in train_pairs}
    test_users = {user for user, _ in test_pairs}
    test_items = {item for _, item in test_pairs}

    unseen_users = sorted(test_users - train_users)
    unseen_items = sorted(test_items - train_items)
    return unseen_users, unseen_items


def filter_test_pairs_against_train(
    train_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], int]:
    train_users = {user for user, _ in train_pairs}
    train_items = {item for _, item in train_pairs}

    filtered_pairs: list[tuple[str, str]] = []
    dropped_count = 0
    for pair in test_pairs:
        user_id, item_id = pair
        if user_id not in train_users or item_id not in train_items:
            dropped_count += 1
            continue
        filtered_pairs.append(pair)
    return filtered_pairs, dropped_count


def write_graph_file(path: Path, pairs: Iterable[tuple[str, str]], weight: int) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user_id, item_id in pairs:
            handle.write(f"{user_id} {item_id} {weight}\n")
            count += 1
    return count


def convert_dataset(dataset_dir: Path, weight: int) -> dict[str, int]:
    train_source = dataset_dir / "user_bundle_train.txt"
    tune_source = dataset_dir / "user_bundle_tune.txt"
    test_source = dataset_dir / "user_bundle_test.txt"

    for source in (train_source, tune_source, test_source):
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")

    train_pairs = deduplicate_preserve_order(read_pairs(train_source))
    tune_pairs = deduplicate_preserve_order(read_pairs(tune_source))
    test_pairs = deduplicate_preserve_order(read_pairs(test_source))

    merged_test_pairs = deduplicate_preserve_order([*tune_pairs, *test_pairs])
    unseen_users, unseen_items = validate_test_pairs(train_pairs, merged_test_pairs)
    filtered_test_pairs, dropped_test_pairs = filter_test_pairs_against_train(train_pairs, merged_test_pairs)
    all_pairs = deduplicate_preserve_order([*train_pairs, *filtered_test_pairs])

    result = {
        "train": write_graph_file(dataset_dir / "train.txt", train_pairs, weight),
        "test": write_graph_file(dataset_dir / "test.txt", filtered_test_pairs, weight),
        "ratings": write_graph_file(dataset_dir / "ratings.txt", all_pairs, weight),
        "raw_test": len(merged_test_pairs),
        "filtered_test": len(filtered_test_pairs),
        "dropped_test_pairs": dropped_test_pairs,
        "unseen_users": len(unseen_users),
        "unseen_items": len(unseen_items),
        "unseen_user_examples": unseen_users[:5],
        "unseen_item_examples": unseen_items[:5],
    }
    return result


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root

    for dataset_name in args.datasets:
        dataset_dir = dataset_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory is not found: {dataset_dir}")

        result = convert_dataset(dataset_dir, args.weight)
        print(f"Converted {dataset_name}.")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Train pairs:      {result['train']}")
        print(
            f"Test pairs:       {result['test']}  "
            f"(filtered from {result['raw_test']} merged tune+test pairs)"
        )
        print(f"Ratings pairs:    {result['ratings']}  (train + filtered test)")
        if result["dropped_test_pairs"]:
            print(
                f"Dropped test pairs: {result['dropped_test_pairs']}  "
                f"(unseen_users={result['unseen_users']}, unseen_items={result['unseen_items']})"
            )
            if result["unseen_user_examples"]:
                print(f"Unseen user examples: {result['unseen_user_examples']}")
            if result["unseen_item_examples"]:
                print(f"Unseen item examples: {result['unseen_item_examples']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
