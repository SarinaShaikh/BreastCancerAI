"""Phase 6 execution driver: trains/evaluates B1, B3, B4 in strict order.

Resumable state machine (experiments/baseline_results/_state.json):
    b1:   "train" -> "frozen"    STEP B+C  (B1 training + freeze)
    b3:   "pending" -> "frozen"  STEP D    (B3 derived from frozen B1)
    b4:   "train" -> "frozen"    STEP E+F  (B4 training + freeze)
    test: "pending" -> "done"    STEP G    (single test evaluation, only after
                                          B1/B3/B4 are all frozen)

STRICT TEST RULE: the driver REFUSES to reach STEP G unless b1/b3/b4 are all
"frozen"; test evaluation runs exactly once per state file. The frozen
data/manifests are read-only here; no split is ever regenerated.

Usage:
    python scripts/run_phase6.py            # run whatever is unfinished
    python scripts/run_phase6.py --status   # print the state machine only
"""
from __future__ import annotations

import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from src.training import train_baseline as tb                  # noqa: E402
from src.training.metrics import classification_metrics        # noqa: E402

STATE_PATH = os.path.join(_REPO, "experiments", "baseline_results",
                          "_state.json")


def _load_cfg():
    return tb._load_baseline_config(_REPO)


def _new_state(cfg) -> dict:
    return {
        "seed": int(cfg["config"]["seed"]),
        "b1": "pending",
        "b3": "pending",
        "b4": "pending",
        "test": "pending",
        "history": [],
    }


def _load_state(cfg) -> dict:
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return _new_state(cfg)


def _save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def _progress(stage):
    def cb(epoch, row):
        print(f"[{stage}] epoch {epoch}: {row}", flush=True)
    return cb


def _chunk_epochs() -> int:
    """Epochs per execution chunk (env override; bit-exact resume makes the
    chunk size irrelevant to the result — only to wall-clock granularity)."""
    return int(os.environ.get("P6_CHUNK_EPOCHS", "1"))


def run_b1(cfg, st) -> None:
    print("=== STEP B: training B1 (image-level, bag-ROC-AUC selection) ===",
          flush=True)
    st["b1"] = "train"
    _save_state(st)
    t0 = time.time()
    cs = os.path.join(_REPO, "experiments", "baseline_results",
                      "_b1_chunk_state.pt")
    log: list = []
    info: dict = {}
    while True:
        model, log, info = tb.train_b1(
            cfg, seed=st["seed"], progress_cb=_progress("B1"),
            chunk_epochs=_chunk_epochs(), chunk_state_path=cs)
        if not info.get("paused"):
            break
        print(f"[B1] chunk done at epoch {info['epochs_run']}; "
              f"resuming at {info['next_epoch']} (bit-exact resume)",
              flush=True)
    b1cfg = cfg["b1_simple_cnn"]
    tb.write_experiment_dir(
        os.path.join(_REPO, b1cfg["artifacts"]), cfg, log,
        {"model": "B1_simple_cnn", "status": "trained_validation_only",
         "training_info": info, "test_evaluated": False, "seed": st["seed"],
         "provenance": {"source_commit": tb._git_commit(),
                        "manifest_hashes": tb._manifest_hashes(),
                        "frozen_test_sha256": tb.TEST_SPLIT_SHA256,
                        "preprocessing_version": tb.PREPROCESSING_VERSION,
                        "pipeline_version": tb.PIPELINE_VERSION,
                        "inherited_checkpoint": None}},
        st["seed"])
    st["b1"] = "frozen"
    st["history"].append({"step": "b1_train", "info": info,
                          "wall_seconds": round(time.time() - t0, 1)})
    _save_state(st)
    print(f"=== B1 frozen: {info} ===", flush=True)


def run_b3(cfg, st) -> None:
    print("=== STEP D: deriving B3 from the FROZEN B1 checkpoint ===",
          flush=True)
    st["b3"] = "derive"
    _save_state(st)
    t0 = time.time()
    from model.baselines.simple_cnn import SimpleCNN
    b1cfg = cfg["b1_simple_cnn"]
    b3cfg = cfg["b3_bag_aggregation"]
    model = SimpleCNN()
    payload = tb.load_checkpoint(
        os.path.join(_REPO, b1cfg["checkpoint"]), model)

    ds_val = tb._bd.UltrasoundBagDataset("val", root=tb._REPO)
    probs, labels, paths, bag_ids = tb.b1_image_predictions(model, ds_val)
    bag_p, bag_y, bag_pred, order = tb.aggregate_bags(
        probs, bag_ids, ds_val, threshold=float(b3cfg["threshold"]))
    val_bag = classification_metrics(bag_y, bag_p, y_pred=bag_pred,
                                     threshold=float(b3cfg["threshold"]))
    val_img = classification_metrics(labels, probs,
                                     threshold=float(b1cfg["threshold"]))

    b3_metrics = {
        "model": "B3_bag_aggregation_of_B1",
        "status": "frozen_from_b1",
        "inherited_checkpoint": os.path.join("model", "baselines",
                                             "b1_simple_cnn", "best.pt"),
        "inherited_checkpoint_epoch": payload["epoch"],
        "threshold": float(b3cfg["threshold"]),
        "threshold_tuning": "none (fixed 0.5; validation only inspected)",
        "seed": st["seed"],
        "validation_bag_level": val_bag,
        "validation_image_level_reference": val_img,
        "test_evaluated": False,
        "provenance": {"source_commit": tb._git_commit(),
                       "manifest_hashes": tb._manifest_hashes(),
                       "frozen_test_sha256": tb.TEST_SPLIT_SHA256,
                       "preprocessing_version": tb.PREPROCESSING_VERSION,
                       "pipeline_version": tb.PIPELINE_VERSION,
                       "inherited_checkpoint": os.path.join(
                           "model", "baselines", "b1_simple_cnn", "best.pt")},
    }
    # B3's own train_log.csv records the inheritance event (no training).
    tb.write_experiment_dir(
        os.path.join(_REPO, b3cfg["artifacts"]), cfg,
        [{"event": "b3_inherits_frozen_b1",
          "b1_checkpoint": b3_metrics["inherited_checkpoint"],
          "b1_epoch": payload["epoch"],
          "aggregation": b3cfg["aggregation"],
          "threshold": float(b3cfg["threshold"])}],
        b3_metrics, st["seed"],
        inherited_checkpoint=b3_metrics["inherited_checkpoint"])
    st["b3"] = "frozen"
    st["history"].append({"step": "b3_derive", "wall_seconds":
                          round(time.time() - t0, 1)})
    _save_state(st)
    print(f"=== B3 frozen (val bag ROC-AUC {val_bag['roc_auc']:.4f}) ===",
          flush=True)


def run_b4(cfg, st) -> None:
    print("=== STEP F: training B4 (bag-level mean-pooling MIL) ===",
          flush=True)
    st["b4"] = "train"
    _save_state(st)
    t0 = time.time()
    cs = os.path.join(_REPO, "experiments", "baseline_results",
                      "_b4_chunk_state.pt")
    log: list = []
    info: dict = {}
    while True:
        model, log, info = tb.train_b4(
            cfg, seed=st["seed"], progress_cb=_progress("B4"),
            chunk_epochs=_chunk_epochs(), chunk_state_path=cs)
        if not info.get("paused"):
            break
        print(f"[B4] chunk done at epoch {info['epochs_run']}; "
              f"resuming at {info['next_epoch']} (bit-exact resume)",
              flush=True)
    b4cfg = cfg["b4_meanpool_mil"]
    tb.write_experiment_dir(
        os.path.join(_REPO, b4cfg["artifacts"]), cfg, log,
        {"model": "B4_meanpool_mil", "status": "trained_validation_only",
         "training_info": info, "test_evaluated": False, "seed": st["seed"],
         "provenance": {"source_commit": tb._git_commit(),
                        "manifest_hashes": tb._manifest_hashes(),
                        "frozen_test_sha256": tb.TEST_SPLIT_SHA256,
                        "preprocessing_version": tb.PREPROCESSING_VERSION,
                        "pipeline_version": tb.PIPELINE_VERSION,
                        "inherited_checkpoint": None}},
        st["seed"])
    st["b4"] = "frozen"
    st["history"].append({"step": "b4_train", "info": info,
                          "wall_seconds": round(time.time() - t0, 1)})
    _save_state(st)
    print(f"=== B4 frozen: {info} ===", flush=True)


def run_test(cfg, st) -> None:
    if not (st["b1"] == "frozen" and st["b3"] == "frozen"
            and st["b4"] == "frozen"):
        raise RuntimeError("STEP G refused: B1/B3/B4 are not all frozen")
    if st.get("test") in ("done", "evaluated"):
        raise RuntimeError("STEP G already executed (single-evaluation rule); "
                           "delete experiments/baseline_results/_state.json "
                           "and re-run everything to repeat deliberately")
    print("=== STEP G: single frozen test evaluation (B1/B3/B4) ===",
          flush=True)
    st["test"] = "running"
    _save_state(st)
    t0 = time.time()

    from model.baselines.simple_cnn import SimpleCNN
    from model.baselines.meanpool_mil import MeanPoolMIL

    b1cfg, b3cfg, b4cfg = (cfg["b1_simple_cnn"], cfg["b3_bag_aggregation"],
                           cfg["b4_meanpool_mil"])
    thr = float(b1cfg["threshold"])
    ds_test = tb._bd.UltrasoundBagDataset("test", root=tb._REPO)

    # --- B1 + B3 (frozen checkpoint) -------------------------------------
    b1 = SimpleCNN()
    payload1 = tb.load_checkpoint(
        os.path.join(_REPO, b1cfg["checkpoint"]), b1)
    probs, labels, paths, bag_ids = tb.b1_image_predictions(b1, ds_test)
    img = classification_metrics(labels, probs, threshold=thr)
    bag_p, bag_y, bag_pred, order = tb.aggregate_bags(
        probs, bag_ids, ds_test, threshold=float(b3cfg["threshold"]))
    bag = classification_metrics(bag_y, bag_p, y_pred=bag_pred,
                                 threshold=float(b3cfg["threshold"]))

    b1_dir = os.path.join(_REPO, b1cfg["artifacts"])
    b1_metrics = json.load(open(os.path.join(b1_dir, "metrics.json"),
                                encoding="utf-8"))
    b1_metrics["test_image_level"] = img
    b1_metrics["test_evaluated"] = True
    b1_metrics["test_checkpoint_epoch"] = payload1["epoch"]
    with open(os.path.join(b1_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(b1_metrics, f, indent=2)
    b3_metrics = dict(bag)
    b3_metrics.update({
        "model": "B3_bag_aggregation_of_B1",
        "level": "bag",
        "test_evaluated": True,
        "seed": st["seed"],
        "inherited_checkpoint": os.path.join("model", "baselines",
                                             "b1_simple_cnn", "best.pt"),
        "provenance": {"source_commit": tb._git_commit(),
                       "manifest_hashes": tb._manifest_hashes(),
                       "frozen_test_sha256": tb.TEST_SPLIT_SHA256,
                       "preprocessing_version": tb.PREPROCESSING_VERSION,
                       "pipeline_version": tb.PIPELINE_VERSION,
                       "inherited_checkpoint": os.path.join(
                           "model", "baselines", "b1_simple_cnn", "best.pt")},
    })
    # preserve B3's validation record + schema fields in its metrics.json
    b3_path = os.path.join(_REPO, b3cfg["artifacts"], "metrics.json")
    b3_old = json.load(open(b3_path, encoding="utf-8"))
    b3_old["test_bag_level"] = bag
    b3_old["test_evaluated"] = True
    with open(b3_path, "w", encoding="utf-8") as f:
        json.dump(b3_old, f, indent=2)

    # --- B4 (frozen checkpoint) -------------------------------------------
    b4 = MeanPoolMIL()
    payload4 = tb.load_checkpoint(
        os.path.join(_REPO, b4cfg["checkpoint"]), b4)
    bp, by, bids = tb.b4_bag_predictions(b4, ds_test)
    b4_bag = classification_metrics(by, bp, threshold=float(b4cfg["threshold"]))
    b4_dir = os.path.join(_REPO, b4cfg["artifacts"])
    b4_metrics = json.load(open(os.path.join(b4_dir, "metrics.json"),
                                encoding="utf-8"))
    b4_metrics["test_bag_level"] = b4_bag
    b4_metrics["test_evaluated"] = True
    b4_metrics["test_checkpoint_epoch"] = payload4["epoch"]
    with open(os.path.join(b4_dir, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(b4_metrics, f, indent=2)

    st["test"] = "done"
    st["history"].append({"step": "test_eval", "wall_seconds":
                          round(time.time() - t0, 1)})
    _save_state(st)
    print("=== test evaluation complete (single evaluation rule honored) ===",
          flush=True)
    print(json.dumps({
        "B1_image": {k: img[k] for k in ("roc_auc", "pr_auc", "accuracy",
                                         "sensitivity", "specificity")},
        "B3_bag": {k: bag[k] for k in ("roc_auc", "pr_auc", "accuracy",
                                       "sensitivity", "specificity")},
        "B4_bag": {k: b4_bag[k] for k in ("roc_auc", "pr_auc", "accuracy",
                                          "sensitivity", "specificity")},
    }, indent=2), flush=True)


def main() -> int:
    cfg = _load_cfg()
    st = _load_state(cfg)
    if "--status" in sys.argv:
        print(json.dumps({k: v for k, v in st.items() if k != "history"},
                         indent=2))
        return 0
    stage = None
    if "--stage" in sys.argv:
        stage = sys.argv[sys.argv.index("--stage") + 1]
        assert stage in ("b1", "b3", "b4", "test"), stage
    def wanted(s):
        return stage is None or stage == s
    if st["b1"] != "frozen" and wanted("b1"):
        run_b1(cfg, st)
    if st["b1"] == "frozen" and st["b3"] != "frozen" and wanted("b3"):
        run_b3(cfg, st)
    if (st["b1"] == "frozen" and st["b3"] == "frozen"
            and st["b4"] != "frozen" and wanted("b4")):
        run_b4(cfg, st)
    if (st["b1"] == "frozen" and st["b3"] == "frozen"
            and st["b4"] == "frozen" and st["test"] not in ("done", "evaluated")
            and wanted("test")):
        run_test(cfg, st)
    print(json.dumps({k: v for k, v in st.items() if k != "history"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
