import json
import os
import secrets
from datetime import datetime


EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "experiments")

#! Not so dynamic, but will do the work

class ExperimentLogger:
    def __init__(self, args: dict):
        os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = secrets.token_hex(3)
        self.path = os.path.join(EXPERIMENTS_DIR, f"{ts}_{uid}.json")
        self._data = {
            "experiment_id": f"{ts}_{uid}",
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "args": args,
            "dataset_stats": {},
            "stages": {str(s): {"epochs": [], "resumed_from_epoch": None,
                                "end_reason": None, "best_val_loss": None}
                       for s in (1, 2, 3)},
        }
        self._save()
        print(f"  [logger] experiment: {self.path}")

    def log_dataset_stats(
        self,
        n_train_descriptions: int,
        n_val_descriptions: int,
        n_train_triples: int,
        n_val_triples: int,
        n_relations: int,
    ) -> None:
        self._data["dataset_stats"] = {
            "n_train_descriptions": n_train_descriptions,
            "n_val_descriptions":   n_val_descriptions,
            "n_train_triples":      n_train_triples,
            "n_val_triples":        n_val_triples,
            "n_relations":          n_relations,
        }
        self._save()

    def log_epoch(
        self,
        stage: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        is_new_best: bool,
    ) -> None:
        entry = {
            "epoch":       epoch,
            "train_loss":  round(train_loss, 6),
            "val_loss":    round(val_loss,   6),
            "is_new_best": is_new_best,
            "timestamp":   datetime.now().isoformat(),
        }
        stage_data = self._data["stages"][str(stage)]
        stage_data["epochs"].append(entry)
        if is_new_best:
            stage_data["best_val_loss"] = round(val_loss, 6)
        self._save()

    def log_resume(self, stage: int, from_epoch: int, best_val_loss: float) -> None:
        stage_data = self._data["stages"][str(stage)]
        stage_data["resumed_from_epoch"] = from_epoch
        stage_data["best_val_loss"]      = round(best_val_loss, 6)
        self._save()

    def log_stage_end(self, stage: int, reason: str) -> None:
        """reason: 'completed' | 'early_stop' | 'nan_loss' | 'already_done'"""
        self._data["stages"][str(stage)]["end_reason"] = reason
        self._save()

    def finish(self) -> None:
        self._data["ended_at"] = datetime.now().isoformat()
        self._save()

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
