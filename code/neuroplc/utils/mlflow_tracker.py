#!/usr/bin/env python3
"""
NeuroPLC — MLflow Experiment Tracking
=======================================
Integrates MLflow for automatic experiment logging: parameters, metrics,
model checkpoints, and artifacts. Designed to work with or without a
remote tracking server (falls back to local file-based tracking).

Usage:
    from neuroplc.utils.mlflow_tracker import ExperimentTracker

    tracker = ExperimentTracker("teacher_training", config=cfg)
    with tracker:
        for epoch in range(epochs):
            loss = train_one_epoch(...)
            tracker.log_metric("train_loss", loss, step=epoch)
        tracker.log_model(model, "teacher_cnn")
        tracker.log_artifact("results/confusion_matrix.png")

    # Or without context manager:
    tracker = ExperimentTracker("student_kd")
    tracker.start()
    ... training ...
    tracker.finish()

MLflow UI:
    mlflow ui --backend-store-uri file:///D:/neuroplc-paper/results/mlruns
    → http://localhost:5000

    # "mlflow ui" may be slow to start on first call;
    # use the modelscope venv: D:/dev-tools/research/venv/Scripts/mlflow
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Optional, Any
from contextlib import ContextDecorator

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore
    HAS_MLFLOW = False
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'results' / 'mlflow.db'}"


def configure_mlflow(tracking_uri: str = None,
                     experiment_name: str = "neuroplc",
                     artifact_location: str = None):
    """
    One-time MLflow configuration. Safe to call multiple times.

    Args:
        tracking_uri:    MLflow tracking server URI (default: local file store)
        experiment_name: experiment name for this project
        artifact_location: custom artifact storage path
    """
    if not HAS_MLFLOW:
        return False

    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)

    # Find or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        mlflow.set_experiment(experiment_name)
    except Exception:
        # If mlflow server is not running, log a warning but don't crash
        print(f"  [MLflow] Warning: Could not configure experiment '{experiment_name}'. "
              f"Metrics will be logged to console only.")
        return False
    return True


# ============================================================================
# Experiment Tracker
# ============================================================================

class ExperimentTracker(ContextDecorator):
    """
    Thin wrapper around MLflow for tracking a single experiment run.

    Features:
      - Auto-starts/stops MLflow run
      - Logs config as params
      - Logs metrics per step
      - Logs PyTorch models as artifacts
      - Logs matplotlib figures / numpy arrays
      - Works offline (local file store) by default
    """

    def __init__(self, run_name: str,
                 config: Optional[dict] = None,
                 experiment_name: str = "neuroplc",
                 tracking_uri: Optional[str] = None,
                 tags: Optional[dict] = None,
                 enabled: bool = True):
        """
        Args:
            run_name:        human-readable name for this run
            config:          hyperparameter dict (logged as MLflow params)
            experiment_name: MLflow experiment name
            tracking_uri:    MLflow tracking URI
            tags:            extra MLflow tags
            enabled:         if False, tracker is a no-op (for quick debugging)
        """
        self.run_name = run_name
        self.config = config or {}
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or DEFAULT_TRACKING_URI
        self.tags = tags or {}
        self.enabled = enabled

        self._run = None
        self._start_time = None

    @property
    def _active(self):
        return self.enabled and HAS_MLFLOW

    # ── Context Manager ──

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.finish()
        return False

    # ── Lifecycle ──

    def start(self):
        """Start an MLflow run."""
        if not self._active:
            return

        configure_mlflow(
            tracking_uri=self.tracking_uri,
            experiment_name=self.experiment_name,
        )

        mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self._run = mlflow.active_run()
        self._start_time = time.time()

        # Log config parameters (flatten nested dicts)
        self._log_params_flat(self.config)

    def finish(self, status: str = "FINISHED"):
        """End the MLflow run."""
        if not self._active or self._run is None:
            return

        elapsed = time.time() - self._start_time if self._start_time else 0
        mlflow.log_metric("wall_time_seconds", elapsed)

        if status != "FINISHED":
            mlflow.set_tag("status", status)

        mlflow.end_run()
        self._run = None

    # ── Logging Methods ──

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a scalar metric."""
        if not self._active:
            return
        mlflow.log_metric(key, value, step=step)

    def log_metrics_batch(self, metrics: dict, step: Optional[int] = None):
        """Log multiple metrics at once."""
        if not self._active:
            return
        mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any):
        """Log a single parameter."""
        if not self._active:
            return
        mlflow.log_param(key, str(value))

    def log_params(self, params: dict):
        """Log multiple parameters."""
        if not self._active:
            return
        self._log_params_flat(params)

    def log_model(self, model, artifact_path: str = "model"):
        """Log a PyTorch model."""
        if not self._active:
            return
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            print(f"  [MLflow] Warning: Failed to log model: {e}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file or directory as an artifact."""
        if not self._active:
            return
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            print(f"  [MLflow] Warning: Failed to log artifact: {e}")

    def log_figure(self, fig, artifact_path: str):
        """Log a matplotlib figure."""
        if not self._active:
            return
        import matplotlib.pyplot as plt
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(tmp.name, artifact_path)
        os.unlink(tmp.name)

    def log_dict(self, d: dict, filename: str):
        """Log a dictionary as JSON artifact."""
        if not self._active:
            return
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(d, tmp, indent=2, ensure_ascii=False)
        mlflow.log_artifact(tmp.name, filename)
        os.unlink(tmp.name)

    def log_classification_report(self, y_true, y_pred,
                                   class_names: list = None,
                                   step: Optional[int] = None):
        """Log classification metrics: accuracy, per-class precision/recall/F1."""
        if not self._active:
            return
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            confusion_matrix,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        metrics = {
            "accuracy": float(acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
        }

        names = class_names or [f"class_{i}" for i in range(len(precision))]
        for i, name in enumerate(names):
            metrics[f"{name}_precision"] = float(precision[i])
            metrics[f"{name}_recall"] = float(recall[i])
            metrics[f"{name}_f1"] = float(f1[i])
            metrics[f"{name}_support"] = int(support[i])

        mlflow.log_metrics(metrics, step=step)

        # Log confusion matrix as artifact
        cm = confusion_matrix(y_true, y_pred)
        self.log_dict(
            {"confusion_matrix": cm.tolist(), "class_names": names},
            "confusion_matrix.json",
        )

        return metrics

    # ── Helpers ──

    def _log_params_flat(self, d: dict, prefix: str = ""):
        """Recursively flatten and log nested dict as params."""
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._log_params_flat(value, full_key)
            elif isinstance(value, (list, tuple)):
                mlflow.log_param(full_key, str(value))
            elif value is not None:
                # Truncate long values
                str_val = str(value)
                if len(str_val) > 500:
                    str_val = str_val[:497] + "..."
                mlflow.log_param(full_key, str_val)

    @property
    def run_id(self) -> Optional[str]:
        """Current MLflow run ID."""
        if self._run:
            return self._run.info.run_id
        return None

    @property
    def run_url(self) -> Optional[str]:
        """URL to this run in MLflow UI."""
        if self._run:
            exp_id = self._run.info.experiment_id
            run_id = self._run.info.run_id
            # Local tracking
            return f"mlflow://{exp_id}/{run_id}"
        return None


# ============================================================================
# Quick-start helpers
# ============================================================================

def get_tracker(run_name: str, config: dict = None, **kwargs) -> ExperimentTracker:
    """
    Factory for ExperimentTracker with sensible defaults for NeuroPLC.

    Usage:
        tracker = get_tracker("E1_teacher_vs_student", config=cfg)
        with tracker:
            ...
    """
    return ExperimentTracker(
        run_name=run_name,
        config=config,
        experiment_name="neuroplc",
        tags={"project": "neuroplc", "phase": "1"},
        **kwargs,
    )


# ============================================================================
# Sanity check
# ============================================================================

if __name__ == "__main__":
    print("NeuroPLC MLflow Tracker — Sanity Check")
    print(f"  Tracking URI: {DEFAULT_TRACKING_URI}")
    print(f"  MLflow version: {mlflow.__version__ if HAS_MLFLOW else 'not installed'}")

    # Quick test
    configure_mlflow(experiment_name="neuroplc_test")
    with ExperimentTracker("test_run", config={"lr": 0.001, "epochs": 10}) as t:
        for i in range(5):
            t.log_metric("loss", 1.0 / (i + 1), step=i)
        t.log_param("test_param", "hello")
        print(f"  Run ID: {t.run_id}")
    print("  OK — MLflow tracker works.")
