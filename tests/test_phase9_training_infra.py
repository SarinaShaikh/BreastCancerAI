"""Phase 9 infrastructure tests (E1 ``da_stage_b``).

Owner-authorized E1-only training infrastructure. Tests use SYNTHETIC bags
(random tensors, no dataset pixels, no split materialization) except the two
explicitly marked dataset-existence checks, which only verify file presence
and the frozen test SHA. The frozen test split is never constructed.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training import train_baseline as tb  # noqa: E402
from src.training import train_dual_attention as td  # noqa: E402
from src.models.dual_attention_mil import DualAttentionMIL  # noqa: E402
from src.mil.instance_generation import InstanceBatch  # NB: provenance built below

SEED = 20260918


# --------------------------------------------------------------- helpers
def _mk_batch(bag_id: str, label: int, n: int, seed: int) -> InstanceBatch:
    g = torch.Generator().manual_seed(seed)
    prov = tuple({
        "image_path": f"{bag_id}_{k}.png", "processed_relpath": "x", "md5": "d" * 32,
        "bag_id": bag_id, "source_group_id": bag_id, "label": "malignant" if label else "benign",
        "split": "train", "source_key": "k", "source_key_status": "filename_derived_unverified",
    } for k in range(n))
    return InstanceBatch(
        bag_id=bag_id, source_group_id=bag_id, label="malignant" if label else "benign",
        numeric_label=label, images=torch.rand(n, 1, 224, 224, generator=g),
        provenance=prov)


def _synthetic_bags(sizes, labels, seed0) -> list:
    return [_mk_batch(f"synthetic_{i}", l, n, seed0 + i)
            for i, (n, l) in enumerate(zip(sizes, labels))]


def _cfg(tmp: Path) -> dict:
    return {
        "experiment": {"id": "da_stage_b",
                       "checkpoint": str(tmp / "model" / "da_stage_b" / "best.pt"),
                       "chunk_state": str(tmp / "chunk" / "state.pt"),
                       "artifacts": str(tmp / "artifacts"),
                       "initialization": "fresh random"},
        "training": {"optimizer": "Adam", "learning_rate": 1e-3,
                     "loss": "BCEWithLogitsLoss (unweighted)",
                     "max_epochs": 20,
                     "early_stopping": {"monitor": "validation bag ROC-AUC",
                                        "patience": 5,
                                        "selection_rule": "strict improvement",
                                        "restore": "best checkpoint"},
                     "threshold": 0.5, "batching": "one optimizer step per bag",
                     "augmentation": "train-only horizontal flip p=0.5",
                     "validation": "deterministic, unaugmented"},
        "model": {"architecture": "Phase 8 DualAttentionMIL (locked)",
                  "trainable_parameters": 184545,
                  "state_dict_elements": 185141},
    }


def _expected_epoch_flip(n: int, rng_seed: int) -> list:
    rng = random.Random(rng_seed)
    return [rng.random() < 0.5 for _ in range(n)]


# ----------------------------------------------------------------- tests
def tda9_01_config_matches_authorization(tmp: Path) -> None:
    """TDA9-01: config protocol == the owner-authorized E1 protocol."""
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "dual_attention_training.yaml")
                         .read_text(encoding="utf-8"))
    e, t, m = cfg["experiment"], cfg["training"], cfg["model"]
    assert e["id"] == "da_stage_b" and e["seed"] == 20260918
    assert e["artifacts"] == "experiments/dual_attention_results/da_stage_b"
    assert e["checkpoint"] == "model/dual_attention/da_stage_b/best.pt"
    assert t["optimizer"] == "Adam" and float(t["learning_rate"]) == 1e-3
    assert "BCEWithLogitsLoss" in t["loss"] and "unweighted" in t["loss"]
    assert t["max_epochs"] == 20
    assert t["early_stopping"]["patience"] == 5
    assert "bag ROC-AUC" in t["early_stopping"]["monitor"]
    assert float(t["threshold"]) == 0.5
    assert m["trainable_parameters"] == 184545
    assert m["state_dict_elements"] == 185141
    assert cfg["test"]["authorized"] is False
    assert cfg["leakage"]["test_access_during_training"] == "PROHIBITED"


def tda9_02_module_forbids_test_split(tmp: Path) -> None:
    """TDA9-02: the trainer cannot construct the test split."""
    src = (REPO / "src" / "training" / "train_dual_attention.py").read_text()
    for banned in ['"test"', "'test'"]:
        assert banned not in src, f"test-split literal present: {banned}"
    assert tuple(td.PHASE9_SPLITS) == ("train", "val")
    try:
        from src.mil.instance_generation import open_split
        open_split("test")
    except ValueError:
        pass
    else:
        raise AssertionError("open_split('test') must be rejected")


def tda9_03_flip_stream_matches_phase6(tmp: Path) -> None:
    """TDA9-03: flip stream == Phase 6 discipline (seed+2, instance order)."""
    n = 40
    expected = tb._epoch_flip_mask(n, random.Random(SEED + 2))
    slices = td._bag_flip_slices(n, [14, 16, 10], random.Random(SEED + 2))
    assert [len(s) for s in slices] == [14, 16, 10]
    flat = [f for s in slices for f in s]
    assert list(flat) == list(expected)
    # one bag's slice equals the direct per-instance draws over its window
    rng = random.Random(SEED + 2)
    direct = [rng.random() < 0.5 for _ in range(n)]
    assert flat == direct


def tda9_04_flip_bitwise_phase5(tmp: Path) -> None:
    """TDA9-04: tensor flip is bitwise the Phase 5 PIL left-right flip."""
    from PIL import Image
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 1, 224, 224, generator=g)
    flags = [True, False, True]
    out = td._apply_flips(x, flags)
    for k, f in enumerate(flags):
        if f:
            ref = np.ascontiguousarray(
                np.array(Image.fromarray(x[k, 0].numpy()).transpose(
                    Image.FLIP_LEFT_RIGHT)))
            assert np.array_equal(out[k, 0].numpy(), ref), f"row {k}"


def tda9_05_single_b4_style_run_equivalence(tmp: Path) -> None:
    """TDA9-05: chunked run == single run (bit-exact chunk resume)."""
    sizes = [14, 16, 21, 16, 14]
    labels = [0, 1, 0, 1, 1]
    bags = _synthetic_bags(sizes, labels, 100)
    val = _synthetic_bags([16, 21, 14], [0, 1, 1], 200)

    m1, log1, i1 = td.train_da_stage_b(
        _cfg(tmp / "a"), bags=bags, val_bags=val, chunk_epochs=5,
        max_epochs_override=5)
    # resume path: 2-epoch chunks over the identical stream
    m2, log2, i2 = td.train_da_stage_b(
        _cfg(tmp / "b"), bags=bags, val_bags=val, chunk_epochs=2,
        max_epochs_override=5)                                      # ep 1-2
    m2, log2, i2 = td.train_da_stage_b(
        _cfg(tmp / "b"), bags=bags, val_bags=val, chunk_epochs=2,
        max_epochs_override=5)                                      # ep 3-4
    # third chunk: epoch 5 then finish (budget 5)
    m2, log2, i2 = td.train_da_stage_b(
        _cfg(tmp / "b"), bags=bags, val_bags=val, chunk_epochs=3,
        max_epochs_override=5)

    assert len(log1) == len(log2) == 5
    assert i1["best_epoch"] == i2["best_epoch"]
    assert abs(i1["best_val_bag_roc_auc"] - i2["best_val_bag_roc_auc"]) < 1e-12
    for r1, r2 in zip(log1, log2):
        assert r1["epoch"] == r2["epoch"]
        assert r1["train_loss"] == r2["train_loss"]
        assert r1["val_bag_roc_auc"] == r2["val_bag_roc_auc"]
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(),
                                  m2.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2), k1


def tda9_06_strict_improvement_and_patience(tmp: Path) -> None:
    """TDA9-06: strict-improvement selection; patience-5 stop."""
    sizes, labels = [16] * 8 + [14], [0, 1] * 4 + [1]
    bags = _synthetic_bags(sizes, labels, 300)
    val = _synthetic_bags([16, 14], [0, 1], 400)
    cfg = _cfg(tmp / "c")
    cfg["training"]["max_epochs"] = 8
    model, log, info = td.train_da_stage_b(cfg, bags=bags, val_bags=val,
                                           max_epochs_override=8)
    aucs = [r["val_bag_roc_auc"] for r in log]
    # first strict-max epoch wins (ties keep the earlier epoch) — this is
    # exactly the checkpoint rule, so these must agree mechanically.
    first_max_epoch = aucs.index(max(aucs)) + 1   # 1-indexed epoch number
    assert max(range(len(aucs)), key=lambda i: (aucs[i], -i)) \
        == first_max_epoch - 1
    assert info["best_epoch"] == first_max_epoch
    assert abs(info["best_val_bag_roc_auc"] - max(aucs)) < 1e-9
    # Phase 6 stopping rule, verbatim: stop when epoch - best_epoch >= patience.
    patience = 5
    expected_run = min(8, first_max_epoch + patience)
    assert info["epochs_run"] == expected_run
    assert info["stopped_early"] == (first_max_epoch + patience < 8)
    ck = torch.load(cfg["experiment"]["checkpoint"], map_location="cpu",
                    weights_only=False)
    assert ck["epoch"] == first_max_epoch
    assert abs(ck["validation_metric"] - max(aucs)) < 1e-9


def tda9_07_checkpoint_payload(tmp: Path) -> None:
    """TDA9-07: checkpoint payload matches the Phase 6 format + provenance."""
    bags = _synthetic_bags([16, 14], [0, 1], 500)
    val = _synthetic_bags([16, 14], [0, 1], 600)   # 2 classes -> valid AUC
    cfg = _cfg(tmp / "d")
    model, log, info = td.train_da_stage_b(cfg, bags=bags, val_bags=val,
                                           max_epochs_override=1)
    ck = torch.load(cfg["experiment"]["checkpoint"], map_location="cpu",
                    weights_only=False)
    assert set(ck) >= {"model_state_dict", "epoch", "seed", "validation_metric",
                       "config", "preprocessing_version", "pipeline_version",
                       "frozen_test_sha256"}
    assert ck["seed"] == SEED
    assert ck["frozen_test_sha256"] == tb.TEST_SPLIT_SHA256
    assert ck["config"]["experiment"]["id"] == "da_stage_b"


def tda9_08_val_predictions_artifact(tmp: Path) -> None:
    """TDA9-08: val-prediction writer: order, provenance, threshold rule."""
    from src.training.train_dual_attention import da_bag_predictions, \
        write_val_predictions
    bags = _synthetic_bags([16, 14], [0, 1], 700)
    val = _synthetic_bags([16, 21, 14], [0, 1, 1], 800)
    cfg = _cfg(tmp / "e")
    model, _, _ = td.train_da_stage_b(cfg, bags=bags, val_bags=val,
                                      max_epochs_override=1)
    probs, labels, ids = da_bag_predictions(model, None, bags=val)
    assert list(labels) == [b.numeric_label for b in val]
    assert ids == [b.bag_id for b in val]
    out = tmp / "e" / "val_predictions.csv"
    write_val_predictions(str(out), probs, labels, ids, val, threshold=0.5)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "bag_id,source_group_id,y_true,probability,prediction,split"
    rows = [dict(zip(lines[0].split(","), l.split(","))) for l in lines[1:]]
    assert [r["bag_id"] for r in rows] == ids
    for r, p in zip(rows, probs):
        assert float(r["probability"]) == p
        assert int(r["prediction"]) == int(float(r["probability"]) >= 0.5)
        assert r["split"] == "val"
    # provenance: source_group_id matches the batch
    by_id = {b.bag_id: b for b in val}
    for r in rows:
        assert r["source_group_id"] == by_id[r["bag_id"]].source_group_id


def tda9_09_test_guard_refuses(tmp: Path) -> None:
    """TDA9-09: the driver's test stage is structurally unauthorized."""
    src = (REPO / "scripts" / "run_phase9.py").read_text(encoding="utf-8")
    assert "ALLOW_TEST_EVALUATION = False" in src
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_phase9_guard",
                                                  REPO / "scripts" / "run_phase9.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.ALLOW_TEST_EVALUATION is False
    try:
        mod.run_test({}, {"test": "pending"})
    except RuntimeError as e:
        assert "NOT authorized" in str(e)
    except NotImplementedError:
        raise AssertionError("guard must refuse at the authorization layer")
    else:
        raise AssertionError("run_test must refuse")


def tda9_10_state_machine_starts_safe(tmp: Path) -> None:
    """TDA9-10: fresh driver state has test=pending; E1 artifacts isolated."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_phase9_state",
                                                  REPO / "scripts" / "run_phase9.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    st = mod._new_state()
    assert st["da_stage_b"] == "pending" and st["test"] == "pending"
    # isolation: paths under the dedicated Phase 9 roots
    src = (REPO / "configs" / "dual_attention_training.yaml").read_text()
    assert "experiments/dual_attention_results" in src
    assert "model/dual_attention/" in src
    assert "baseline_results" not in src.replace(
        "never ``experiments/baseline_results/``", "")  # doc reference only
    assert "model/baselines" not in src


def tda9_11_parameter_counts_locked(tmp: Path) -> None:
    """TDA9-11: E1 model = locked counts; frozen-trunk variant = 21,009."""
    tb.set_seed(SEED)
    m = DualAttentionMIL()
    pc = m.param_counts()
    assert pc["full_model"] == 184545
    assert m.state_dict_element_count(m) == 185141
    head_only = pc["head_excluding_trunk"] == 21009
    assert head_only
    m.extractor.freeze_trunk(True)
    pc2 = m.param_counts()
    assert pc2["full_model"] == 21009      # frozen trunk drops out
    m.extractor.freeze_trunk(False)
    assert m.param_counts()["full_model"] == 184545


def tda9_12_drivers_independent_of_phase6_state(tmp: Path) -> None:
    """TDA9-12: Phase 9 driver touches no Phase 6 state/artifacts."""
    src = (REPO / "scripts" / "run_phase9.py").read_text(encoding="utf-8")
    assert "baseline_results" in src  # doc comment only
    assert 'STATE_PATH = os.path.join(_REPO, "experiments", "dual_attention_results"' in src
    assert '"test": "pending"' in src
    # no Phase 6 driver import / invocation
    assert "run_phase6" not in src


def tda9_13_frozen_artifacts_untouched_by_trainer_import(tmp: Path) -> None:
    """TDA9-13: importing/constructing nothing writes frozen artifacts."""
    before = {p: p.stat().st_mtime_ns for p in
              (REPO / "experiments" / "baseline_results").rglob("*")
              if p.is_file()}
    from src.training import train_dual_attention  # noqa: F401
    after = {p: p.stat().st_mtime_ns for p in
             (REPO / "experiments" / "baseline_results").rglob("*")
             if p.is_file()}
    assert before == after


def tda9_14_frozen_test_sha_stable(tmp: Path) -> None:
    """TDA9-14: the frozen test manifest SHA is unchanged on disk."""
    import hashlib
    manifest = REPO / "data" / "manifests" / "test_split.csv"
    h = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert h == ("959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74")


def main() -> int:
    import tempfile
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("tda9_") and callable(v)]
    passed = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"PASS {name}")
                passed += 1
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e}")
    print(f"\n{passed}/{len(tests)} Phase 9 infrastructure tests passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
