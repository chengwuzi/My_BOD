from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import auto_search as search
from scripts import staged_search as staged
from scripts.prepare_multicbr_datasets import convert_separate_splits


class StagedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.test_calls = []
        self.fail_rate = None
        self.targets = {"MealRec+H": (0.004, 0.0000125), "MealRec+L": (0.001, 0.0004)}
        self.configs = []
        for name, reg in (("MealRec+H", "0.00005"), ("MealRec+L", "0.0001")):
            data = self.root / name
            data.mkdir()
            (data / "train.txt").write_text("u0 b0 1\nu0 b1 1\nu1 b1 1\nu1 b2 1\nu2 b0 1\nu2 b2 1\n")
            (data / "valid.txt").write_text("u0 b2 1\n")
            (data / "test.txt").write_text("u1 b0 1\n")
            config = search.read_conf_file(search.REPO_ROOT / "conf" / f"LightGCN_{name}.conf")
            config.update({"training.set": str(data / "train.txt"), "test.set": str(data / "valid.txt"),
                           "final.test.set": str(data / "test.txt"), "num.max.epoch": "2", "reg.lambda": reg})
            path = self.root / f"{name}.conf"
            search.write_conf_file(path, config)
            self.configs.append(path)
        self.spec_path = self.root / "staged.json"
        search.dump_json(self.spec_path, {"mode": "staged", "output_dir": str(self.root / "runs"),
                         "max_attempts": 2, "resume": True,
                         "datasets": [{"base_config": str(path)} for path in self.configs]})

    def fake_trial(self, config_path, result_json, log_path, timeout_sec, **kwargs):
        config = search.read_conf_file(config_path)
        self.calls.append(config)
        self.assertEqual(Path(config["test.set"]).name, "valid.txt")
        lr, reg = float(config["learnRate"]), float(config["reg.lambda"])
        if lr == self.fail_rate:
            return 1, {"status": "failed", "error_type": "SimulatedFailure", "error": "retry me"}
        target_lr, target_reg = self.targets[config["dataset.name"]]
        score = 0.8 - abs(math.log2(lr / target_lr)) * 0.05 - abs(math.log2(reg / target_reg)) * 0.01
        output_dir = Path(dict(search.parse_option_string(config["output.setup"])[1])["-dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for kind in ("user", "bundle"):
            path = output_dir / f"{kind}.pt"
            path.write_bytes(f"{config['dataset.name']} {kind} {lr} {reg}".encode())
            paths[kind] = str(path)
        result = {"status": "success", "best_epoch": 1, "embedding_paths": paths,
                  "metrics": {"10": {"NDCG": score / 2, "Recall": score / 2},
                              "20": {"NDCG": score, "Recall": score}}, "model": "LightGCN"}
        search.dump_json(result_json, result)
        return 0, result

    def fake_test(self, config_path, embedding_paths):
        config = search.read_conf_file(config_path)
        self.test_calls.append((config, len(self.calls)))
        self.assertEqual(Path(config["final.test.set"]).name, "test.txt")
        for kind, path in embedding_paths.items():
            self.assertIn(kind.encode(), Path(path).read_bytes())
        # Deliberately unrelated to validation; test metrics cannot select parameters.
        return {"status": "success", "split": "test", "metrics": {"20": {"NDCG": 0.0, "Recall": 0.0}}}

    def run_pipeline(self, dry_run=False):
        with patch.object(search, "run_trial_subprocess", side_effect=self.fake_trial), \
                patch.object(staged, "evaluate_test", side_effect=self.fake_test), \
                contextlib.redirect_stdout(io.StringIO()):
            return staged.run_staged_search(self.spec_path, dry_run=dry_run)

    def test_candidates_and_incumbent(self):
        self.assertEqual(staged.candidate_values("0.002", staged.DEFAULT_MULTIPLIERS),
                         ["0.0002", "0.00050", "0.0010", "0.004", "0.008", "0.020"])
        with self.assertRaises(ValueError):
            staged.candidate_values("0.002", [0.1, 0.25, 0.5, 1, 2, 4])
        incumbent = {"status": "success", "metrics": {"20": {"NDCG": 0.5, "Recall": 0.5}}}
        self.assertIs(staged.select_best([incumbent, dict(incumbent)]), incumbent)
        with self.assertRaises(ValueError):
            staged.select_best([{"status": "success", "metrics": {"20": {"NDCG": float('nan'), "Recall": 0.0}}}])

    def test_two_datasets_run_13_trials_each_and_resume(self):
        self.assertEqual(self.run_pipeline(), 0)
        self.assertEqual(len(self.calls), 26)
        self.assertEqual([count for _, count in self.test_calls], [13, 26])
        for name, expected_lr, expected_reg in (("MealRec+H", 0.004, 0.0000125), ("MealRec+L", 0.001, 0.0004)):
            calls = [config for config in self.calls if config["dataset.name"] == name]
            self.assertEqual(len(calls), 13)
            for config in calls[7:]:
                self.assertEqual(float(config["learnRate"]), expected_lr)
            result = search.load_json(self.root / "runs" / search.safe_name(name) / "best_result.json")
            self.assertEqual(float(result["parameters"]["learnRate"]), expected_lr)
            self.assertEqual(float(result["parameters"]["reg.lambda"]), expected_reg)
            self.assertEqual(result["test_metrics"]["20"]["NDCG"], 0.0)
        self.run_pipeline()
        self.assertEqual(len(self.calls), 26)
        self.assertEqual(len(self.test_calls), 2)

    def test_failures_are_retried_and_block_stage_two_until_resume(self):
        spec = search.load_json(self.spec_path)
        spec["datasets"] = spec["datasets"][:1]
        search.dump_json(self.spec_path, spec)
        self.fail_rate = 0.0002
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.run_pipeline()
        self.assertFalse(any(config["search.name"].endswith("02_reg_lambda") for config in self.calls))
        self.assertEqual(len(self.test_calls), 0)
        failed_calls = sum(float(config["learnRate"]) == self.fail_rate for config in self.calls)
        self.assertEqual(failed_calls, 2)
        old_count = len(self.calls)
        self.fail_rate = None
        self.run_pipeline()
        self.assertEqual(len(self.calls) - old_count, 7)  # retry one failed trial, then six reg trials
        self.assertEqual(len(self.test_calls), 1)

    def test_initial_configuration_survives_when_all_candidates_are_worse(self):
        self.targets = {"MealRec+H": (0.002, 0.00005), "MealRec+L": (0.002, 0.0001)}
        self.run_pipeline()
        for name, (lr, reg) in self.targets.items():
            result = search.load_json(self.root / "runs" / search.safe_name(name) / "best_result.json")
            self.assertEqual(float(result["parameters"]["learnRate"]), lr)
            self.assertEqual(float(result["parameters"]["reg.lambda"]), reg)
            self.assertIn("00_initial", result["winning_trial"])

    def test_missing_trial_embeddings_are_retrained_on_resume(self):
        self.run_pipeline()
        record = search.load_json(self.root / "runs" / "MealRec_H" / "00_initial" / "run_results.json")["trials"][0]
        Path(record["embedding_paths"]["user"]).unlink()
        self.run_pipeline()
        self.assertEqual(len(self.calls), 27)

    def test_changed_validation_data_invalidates_resume(self):
        self.run_pipeline()
        with (self.root / "MealRec+H" / "valid.txt").open("a") as handle:
            handle.write("u2 b1 1\n")
        self.run_pipeline()
        self.assertEqual(len(self.calls), 39)  # Only H's 13 trials are invalidated.
        self.assertEqual(len(self.test_calls), 3)

    def test_changed_base_configuration_invalidates_resume(self):
        spec = search.load_json(self.spec_path)
        spec["datasets"] = spec["datasets"][:1]
        search.dump_json(self.spec_path, spec)
        self.run_pipeline()
        base = search.read_conf_file(self.configs[0])
        base["num.max.epoch"] = "3"
        search.write_conf_file(self.configs[0], base)
        self.run_pipeline()
        self.assertEqual(len(self.calls), 26)
        self.assertEqual(len(self.test_calls), 2)

    def test_dry_run_never_trains_and_does_not_invent_stage_two_winner(self):
        self.run_pipeline(dry_run=True)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.test_calls, [])
        plan = search.load_json(self.root / "runs" / "search_plan.json")
        self.assertEqual(plan["total_training_trials"], 26)
        self.assertIn("best of baseline", plan["datasets"][0]["regularization_stage_learning_rate"])

    def test_partition_overlap_is_rejected_before_training(self):
        data = self.root / "MealRec+H"
        (data / "test.txt").write_text((data / "valid.txt").read_text())
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_pipeline()
        self.assertEqual(self.calls, [])

    def test_summary_recovers_partial_line_and_does_not_skip_failures(self):
        path = self.root / "summary.jsonl"
        path.write_text('{"status": "failed", "signature": "failed"}\n{"partial":', encoding="utf-8")
        search.append_jsonl(path, {"status": "success", "signature": "good"})
        self.assertEqual(search.load_completed_signatures(path), {"good"})
        search.rebuild_summary_csv(path, self.root / "summary.csv")


class PreparationTests(unittest.TestCase):
    def test_validation_and_test_are_kept_separate_and_cold_entities_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "raw", root / "prepared"
            source.mkdir()
            (source / "user_bundle_train.txt").write_text("u0\tb0\nu1\tb1\nu0\tb0\n")
            (source / "user_bundle_tune.txt").write_text("u0\tb1\nu9\tb0\n")
            (source / "user_bundle_test.txt").write_text("u1\tb0\nu1\tb9\n")
            report = convert_separate_splits(source, output)
            self.assertEqual((output / "valid.txt").read_text(), "u0 b1 1\n")
            self.assertEqual((output / "test.txt").read_text(), "u1 b0 1\n")
            self.assertEqual(report["splits"]["train"]["kept_pairs"], 2)
            self.assertEqual(report["splits"]["valid"]["dropped_pairs"], 1)
            self.assertEqual(report["splits"]["test"]["dropped_pairs"], 1)
            self.assertEqual((source / "user_bundle_tune.txt").read_text(), "u0\tb1\nu9\tb0\n")


if __name__ == "__main__":
    unittest.main()
