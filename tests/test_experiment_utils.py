import json
import tempfile
import unittest
from pathlib import Path

import experiment_utils as eu


class ObjectiveTests(unittest.TestCase):
    def test_defaults_preserve_bpb_minimization(self):
        self.assertEqual(eu.DEFAULT_PRIMARY_METRIC, "val_bpb")
        self.assertEqual(eu.DEFAULT_OBJECTIVE_DIRECTION, "min")
        self.assertTrue(eu.is_better(0.9, 1.0, "min"))
        self.assertFalse(eu.is_better(1.1, 1.0, "min"))
        result = eu.make_run_result(
            log_text='eval_metrics: {"val_bpb": 1.25, "val_loss": 2.5}\n',
            primary_metric=eu.DEFAULT_PRIMARY_METRIC,
            direction=eu.DEFAULT_OBJECTIVE_DIRECTION,
            train_exit=0,
        )
        self.assertEqual(result["objective_value"], 1.25)
        self.assertEqual(result["run_status"], "ok")

    def test_maximize_comparison(self):
        self.assertTrue(eu.is_better(0.8, 0.7, "max"))
        self.assertFalse(eu.is_better(0.6, 0.7, "max"))
        records = [
            {"status": "keep", "metrics": {"task_accuracy": 0.7}},
            {"status": "keep", "metrics": {"task_accuracy": 0.8}},
        ]
        self.assertEqual(eu.best_value(records, "task_accuracy", "max"), 0.8)

    def test_legacy_top_level_bpb_is_read_without_other_inference(self):
        old = {"status": "keep", "val_bpb": 1.0021}
        self.assertEqual(eu.metric_value(old, "val_bpb"), 1.0021)
        self.assertIsNone(eu.metric_value(old, "val_loss"))

    def test_val_loss_objective_is_selected_from_multi_metric_log(self):
        log = (
            'val_bpb:          1.002100\n'
            'val_loss:         2.340000\n'
            'eval_metrics: {"val_bpb": 1.0021, "val_loss": 2.34}\n'
            'peak_vram_mb:     1024.0\n'
        )
        result = eu.make_run_result(
            log_text=log,
            primary_metric="val_loss",
            direction="min",
            train_exit=0,
        )
        self.assertEqual(result["objective_value"], 2.34)
        self.assertEqual(result["metrics"], {"val_bpb": 1.0021, "val_loss": 2.34})
        self.assertEqual(result["run_status"], "ok")
        self.assertEqual(result["memory_gb"], 1.0)

    def test_missing_selected_objective_is_a_crash(self):
        result = eu.make_run_result(
            log_text='eval_metrics: {"val_bpb": 1.0}\n',
            primary_metric="val_loss",
            direction="min",
            train_exit=0,
        )
        self.assertEqual(result["run_status"], "crash")
        self.assertIsNone(result["objective_value"])

    def test_malformed_or_nonfinite_objective_is_a_crash(self):
        result = eu.make_run_result(
            log_text='eval_metrics: {bad json}\nval_loss: nan\n',
            primary_metric="val_loss",
            direction="min",
            train_exit=0,
        )
        self.assertEqual(result["run_status"], "crash")

    def test_new_multi_metric_record_round_trip(self):
        run = {
            "gpu": 0,
            "train_exit": 0,
            "memory_gb": 1.0,
            "metrics": {"val_bpb": 1.0, "val_loss": 2.0},
        }
        record = eu.make_result_record(
            iteration=1,
            status="keep",
            description="baseline",
            primary_metric="val_loss",
            direction="min",
            run=run,
            config={"hyperparameters": {"LR": 0.1}},
            prompt={"template": "prompts/candidate_default.txt"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            loaded = eu.load_jsonl(path)[0]
        self.assertEqual(loaded["primary_metric"], "val_loss")
        self.assertEqual(loaded["objective_value"], 2.0)
        self.assertEqual(loaded["metrics"]["val_bpb"], 1.0)
        self.assertEqual(loaded["val_bpb"], 1.0)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]
        self.default_template = self.repo / "prompts/candidate_default.txt"
        self.alternative_template = self.repo / "prompts/candidate_exploratory.txt"
        self.temp = tempfile.TemporaryDirectory()
        self.results = Path(self.temp.name) / "results.jsonl"
        records = [
            {
                "iteration": index,
                "status": "keep",
                "metrics": {"val_bpb": 2 - index / 10},
                "log": f"results/run/iter_{index}/train.log",
                "artifacts": {"history": f"results/run/iter_{index}/history_context.jsonl"},
            }
            for index in range(1, 6)
        ]
        self.results.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def render(self, template, mode, limit=None):
        return eu.render_prompt(
            template_path=template,
            results_path=self.results,
            history_mode=mode,
            history_limit=limit,
            primary_metric="val_bpb",
            objective_direction="min",
            iteration=6,
            base_commit="abcdef0",
            current_best=1.5,
            repo_root=self.repo,
        )

    def test_full_history_prompt(self):
        prompt, history, metadata = self.render(self.default_template, "all")
        self.assertEqual(len([line for line in history.splitlines() if line]), 5)
        self.assertIn('"iteration": 1', prompt)
        self.assertIn('"iteration": 5', prompt)
        self.assertEqual(metadata["history_records_supplied"], 5)
        self.assertEqual(metadata["history_mode"], "all")
        self.assertIsNone(metadata["history_limit"])
        self.assertNotIn("results.jsonl", prompt)
        self.assertNotIn("results/run", prompt)

    def test_recent_n_prompt(self):
        prompt, history, metadata = self.render(self.default_template, "recent", 2)
        self.assertNotIn('"iteration": 3', history)
        self.assertIn('"iteration": 4', history)
        self.assertIn('"iteration": 5', prompt)
        self.assertEqual(metadata["history_records_supplied"], 2)
        self.assertEqual(metadata["history_limit"], 2)

    def test_recent_n_with_fewer_records(self):
        _, history, metadata = self.render(self.default_template, "recent", 10)
        self.assertEqual(len(history.splitlines()), 5)
        self.assertEqual(metadata["history_records_supplied"], 5)

    def test_rendered_prompt_metadata_hashes(self):
        prompt, _, metadata = self.render(self.default_template, "all")
        self.assertEqual(metadata["rendered_prompt_sha256"], eu.sha256_text(prompt))
        self.assertEqual(
            metadata["template_sha256"],
            eu.sha256_text(self.default_template.read_text(encoding="utf-8")),
        )
        self.assertEqual(metadata["primary_metric"], "val_bpb")
        self.assertEqual(metadata["objective_direction"], "min")

    def test_template_selection_changes_identifier_and_content(self):
        default_prompt, _, default_metadata = self.render(self.default_template, "all")
        other_prompt, _, other_metadata = self.render(self.alternative_template, "all")
        self.assertNotEqual(default_prompt, other_prompt)
        self.assertEqual(other_metadata["template"], "prompts/candidate_exploratory.txt")
        self.assertNotEqual(default_metadata["template_sha256"], other_metadata["template_sha256"])


if __name__ == "__main__":
    unittest.main()
