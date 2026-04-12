from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(
        description="Convert MultiCBR iFashion_UB user-bundle splits into BOD graph dataset files."
    ).parse_args(
        namespace=argparse.Namespace(
            input_dir=repo_root / "iFashion_UB",
            output_dir=repo_root / "dataset" / "iFashion_UB",
            weight=1,
        )
    )


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


def validate_test_pairs(train_pairs: list[tuple[str, str]], test_pairs: list[tuple[str, str]]) -> None:
    train_users = {user for user, _ in train_pairs}
    train_items = {item for _, item in train_pairs}
    test_users = {user for user, _ in test_pairs}
    test_items = {item for _, item in test_pairs}

    unseen_users = sorted(test_users - train_users)
    unseen_items = sorted(test_items - train_items)

    if unseen_users:
        raise ValueError(
            f"Found {len(unseen_users)} test users missing from train. Example: {unseen_users[:5]}"
        )
    if unseen_items:
        raise ValueError(
            f"Found {len(unseen_items)} test bundles missing from train. Example: {unseen_items[:5]}"
        )


def write_bod_file(path: Path, pairs: Iterable[tuple[str, str]], weight: int) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for user_id, item_id in pairs:
            handle.write(f"{user_id} {item_id} {weight}\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    train_source = input_dir / "user_bundle_train.txt"
    tune_source = input_dir / "user_bundle_tune.txt"
    test_source = input_dir / "user_bundle_test.txt"

    for source in (train_source, tune_source, test_source):
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")

    train_pairs = deduplicate_preserve_order(read_pairs(train_source))
    tune_pairs = deduplicate_preserve_order(read_pairs(tune_source))
    test_pairs = deduplicate_preserve_order(read_pairs(test_source))

    merged_test_pairs = deduplicate_preserve_order([*tune_pairs, *test_pairs])
    all_pairs = deduplicate_preserve_order([*train_pairs, *merged_test_pairs])

    validate_test_pairs(train_pairs, merged_test_pairs)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_count = write_bod_file(output_dir / "train.txt", train_pairs, args.weight)
    test_count = write_bod_file(output_dir / "test.txt", merged_test_pairs, args.weight)
    ratings_count = write_bod_file(output_dir / "ratings.txt", all_pairs, args.weight)

    print("Converted iFashion_UB for BOD.")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Train pairs:      {train_count}")
    print(f"Test pairs:       {test_count}  (tune + test)")
    print(f"Ratings pairs:    {ratings_count}  (train + merged test)")


if __name__ == "__main__":
    main()
