import json
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import experiment_logging.experiment_logger as logger_module
from experiment_logging.experiment_logger import ExperimentLogger

FAKE_ARGS = {"batch_size": 4, "stage1_epochs": 10, "val_split": 0.1}


def make_logger(tmp_dir):
    with patch.object(logger_module, "EXPERIMENTS_DIR", tmp_dir):
        return ExperimentLogger(FAKE_ARGS), tmp_dir


# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(logger):
    with open(logger.path) as f:
        return json.load(f)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_init_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        assert os.path.exists(logger.path)


def test_init_json_structure():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        data = load_json(logger)
        assert data["args"] == FAKE_ARGS
        assert data["ended_at"] is None
        assert set(data["stages"].keys()) == {"1", "2", "3"}
        for s in ("1", "2", "3"):
            assert data["stages"][s]["epochs"] == []
            assert data["stages"][s]["end_reason"] is None
            assert data["stages"][s]["best_val_loss"] is None


def test_log_dataset_stats():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        logger.log_dataset_stats(
            n_train_descriptions=1800,
            n_val_descriptions=200,
            n_train_triples=5000,
            n_val_triples=600,
            n_relations=50,
        )
        data = load_json(logger)
        stats = data["dataset_stats"]
        assert stats["n_train_descriptions"] == 1800
        assert stats["n_val_descriptions"]   == 200
        assert stats["n_train_triples"]      == 5000
        assert stats["n_val_triples"]        == 600
        assert stats["n_relations"]          == 50


def test_log_epoch_appends_and_tracks_best():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)

        logger.log_epoch(1, epoch=1, train_loss=0.9, val_loss=0.8, is_new_best=True)
        logger.log_epoch(1, epoch=2, train_loss=0.7, val_loss=0.85, is_new_best=False)
        logger.log_epoch(1, epoch=3, train_loss=0.6, val_loss=0.75, is_new_best=True)

        data  = load_json(logger)
        stage = data["stages"]["1"]

        assert len(stage["epochs"]) == 3
        assert stage["epochs"][0]["epoch"]       == 1
        assert stage["epochs"][0]["is_new_best"] is True
        assert stage["epochs"][1]["is_new_best"] is False
        assert stage["best_val_loss"]            == round(0.75, 6)


def test_log_epoch_loss_values_rounded():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        logger.log_epoch(2, epoch=1, train_loss=0.123456789, val_loss=0.987654321, is_new_best=True)
        data = load_json(logger)
        entry = data["stages"]["2"]["epochs"][0]
        assert entry["train_loss"] == round(0.123456789, 6)
        assert entry["val_loss"]   == round(0.987654321, 6)


def test_log_resume():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        logger.log_resume(stage=2, from_epoch=15, best_val_loss=0.42)
        data = load_json(logger)
        stage = data["stages"]["2"]
        assert stage["resumed_from_epoch"] == 15
        assert stage["best_val_loss"]      == round(0.42, 6)


def test_log_stage_end_reasons():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        for stage, reason in [(1, "early_stop"), (2, "completed"), (3, "already_done")]:
            logger.log_stage_end(stage, reason)
        data = load_json(logger)
        assert data["stages"]["1"]["end_reason"] == "early_stop"
        assert data["stages"]["2"]["end_reason"] == "completed"
        assert data["stages"]["3"]["end_reason"] == "already_done"


def test_finish_sets_ended_at():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        assert load_json(logger)["ended_at"] is None
        logger.finish()
        assert load_json(logger)["ended_at"] is not None


def test_file_persists_across_multiple_calls():
    """Each call must keep all previous data — no overwrites that lose earlier entries."""
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = make_logger(tmp)
        logger.log_dataset_stats(100, 10, 300, 30, 20)
        logger.log_epoch(1, 1, 0.9, 0.8, True)
        logger.log_epoch(1, 2, 0.7, 0.75, True)
        logger.log_resume(2, 5, 0.6)
        logger.log_stage_end(1, "completed")
        logger.finish()

        data = load_json(logger)
        assert len(data["stages"]["1"]["epochs"])   == 2
        assert data["dataset_stats"]["n_relations"] == 20
        assert data["stages"]["2"]["resumed_from_epoch"] == 5
        assert data["ended_at"] is not None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
