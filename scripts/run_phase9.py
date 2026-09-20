"""Phase 9 driver: E1 ``da_stage_b`` training with a guarded test stage.

Owner authorization (2026-09-19): E1 ONLY. This driver follows the Phase 6
driver's stage/state pattern but is fully isolated:

* artifacts in ``experiments/dual_attention_results/`` +
  ``model/dual_attention/`` (never ``experiments/baseline_results/`` or
  ``model/baselines/``);
* its own state file with ``test: pending`` — the Phase 6 STEP-G single-
  evaluation record is untouched;
* the ``test`` stage is structurally unauthorized: it refuses unless the
  owner both edits this file (ALLOW_TEST_EVALUATION) and passes --test;
  no test-evaluation implementation exists in this repository yet.

Usage (one epoch per invocation, detached):
    venv/Scripts/python.exe scripts/run_phase9.py            # next stage
    venv/Scripts/python.exe scripts/run_phase9.py --status   # inspect only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

STATE_PATH = os.path.join(_REPO, "experiments", "dual_attention_results",
                          "_state.json")
CFG_PATH = os.path.join(_REPO, "configs", "dual_attention_training.yaml")

# O4: test evaluation is NOT authorized. The stage below remains a refusing
# guard until the owner flips this constant in a separately authorized change.
ALLOW_TEST_EVALUATION = False


def _load_cfg() -> dict:
    import yaml
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _new_state() -> dict:
    return {"seed": 20260918, "da_stage_b": "pending", "test": "pending",
            "history": []}


def _load_state() -> dict:
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return _new_state()


def _save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def _progress(stage):
    def cb(epoch, row):
        print(f"[{stage}] epoch {row['epoch']}: "
              f"train_loss={row['train_loss']} "
              f"val_bag_roc_auc={row['val_bag_roc_auc']} "
              f"({row['epoch_seconds']}s)", flush=True)
    return cb


def _chunk_epochs() -> int:
    return int(os.environ.get("P9_CHUNK_EPOCHS", "1"))


def run_da_stage_b(cfg, st) -> None:
    if st.get("da_stage_b") in ("frozen", "done"):
        print("E1 already complete; refusing to retrain (single-run rule).")
        return
    from src.training import train_dual_attention as td
    from src.training.train_baseline import write_experiment_dir

    st["da_stage_b"] = "train"
    _save_state(st)
    print(f"=== E1 da_stage_b: {os.environ.get('P9_CHUNK_EPOCHS', '1')} "
          "chunk epoch(s) ===", flush=True)

    model, log, info = td.train_da_stage_b(
        cfg,
        chunk_epochs=_chunk_epochs(),
        chunk_state_path=os.path.join(_REPO, cfg["experiment"]["chunk_state"]),
        progress_cb=_progress("E1"),
    )

    if info.get("paused"):
        st["history"].append({"step": "da_stage_b_chunk", "info": info})
        _save_state(st)
        print(f"PAUSED after epoch {info['epochs_run']}; "
              f"next_epoch={info['next_epoch']}", flush=True)
        return

    # -------- finishing path: artifacts, best-restore verification --------
    exp_dir = os.path.join(_REPO, cfg["experiment"]["artifacts"])
    checkpoint = os.path.join(_REPO, cfg["experiment"]["checkpoint"])
    cfg_payload = {
        "experiment": dict(cfg["experiment"]),
        "training": dict(cfg["training"]),
        "model": dict(cfg["model"]),
        "seed": st["seed"],
        "initialization": cfg["experiment"].get("initialization", "fresh"),
    }
    metrics = {
        "experiment": "da_stage_b",
        "seed": st["seed"],
        "best_epoch": info["best_epoch"],
        "best_validation_bag_roc_auc": info["best_val_bag_roc_auc"],
        "epochs_run": info["epochs_run"],
        "stopped_early": info["stopped_early"],
        "threshold": 0.5,
        "threshold_tuned": False,
        "test_evaluated": False,
        "trainable_parameters": 184545,
        "state_dict_elements": 185141,
    }
    write_experiment_dir(exp_dir, cfg_payload, log, metrics, st["seed"])

    from src.mil.instance_generation import open_split
    from src.training.train_dual_attention import da_bag_predictions, \
        write_val_predictions
    val_ds = open_split("val")
    val_bags = [td.load_bag_batch(b, val_ds) for b in val_ds.bags]
    probs, labels, ids = da_bag_predictions(model, val_ds, bags=val_bags)
    write_val_predictions(
        os.path.join(exp_dir, "val_predictions.csv"),
        probs, labels, ids, val_bags,
        threshold=float(cfg["training"]["threshold"]))

    st["da_stage_b"] = "frozen"
    st["history"].append({"step": "da_stage_b_train", "info": info})
    _save_state(st)
    print(f"E1 COMPLETE: best epoch {info['best_epoch']} @ val bag ROC-AUC "
          f"{info['best_val_bag_roc_auc']:.6f} "
          f"({info['epochs_run']} epochs, early_stop={info['stopped_early']})",
          flush=True)


def run_test(cfg, st) -> None:
    """Structurally unauthorized guard (O4). No test-evaluation implementation
    exists in Phase 9; this stage cannot evaluate the test set."""
    if st.get("test") in ("done", "evaluated"):
        print("Phase 9 test already recorded; single-evaluation rule.")
        return
    if not ALLOW_TEST_EVALUATION:
        raise RuntimeError(
            "Phase 9 test evaluation is NOT authorized (O4). It requires a "
            "separate explicit owner authorization and is not implemented in "
            "this repository yet.")
    raise NotImplementedError(
        "The Phase 9 test-evaluation implementation does not exist; it will "
        "be added only under separate owner authorization.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["da_stage_b", "test"],
                    default="da_stage_b")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    st = _load_state()
    if args.status:
        print(json.dumps(st, indent=2))
        return 0

    cfg = _load_cfg()
    if args.stage == "da_stage_b":
        run_da_stage_b(cfg, st)
    else:
        run_test(cfg, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
