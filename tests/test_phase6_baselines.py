"""Phase 6 validation suite for the baseline deliverables (B1/B3/B4).

Runs standalone (``python tests/test_phase6_baselines.py``) or under pytest.
Covers the owner instruction's minimum test list:

  1. B1 forward pass shape                    (V6B-01)
  2. B4 forward pass: one bag / multiple bags / different bag sizes (V6B-02)
  3. One binary logit per image/bag           (V6B-03)
  4. B4 truly uses mean pooling               (V6B-04)
  5. Label mapping benign=0, malignant=1      (V6B-05)
  6. Dataset interface reuse                  (V6B-06)
  7. No cross-split source_group_id overlap   (V6B-07)
  8. Validation/test deterministic behavior   (V6B-08)
  9. Train-only augmentation                  (V6B-09)
 10. Checkpoint save/load + identical outputs (V6B-10, V6B-11)
 12. Metric functions vs hand-computed cases (V6B-12)
 13. ROC-AUC / PR-AUC calculation             (V6B-13)
 14. Seeded short-run reproducibility         (V6B-14)
 15. Configuration validation                 (V6B-15)
 16. Artifact schema validation               (V6B-16)
 17. Phase boundary scan                      (V6B-17)

Scope guard: BASELINE models only — no Dual Attention MIL, no attention, no
MRI, no pretrained weights (B2 deferred), no committed checkpoints.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.baselines.simple_cnn import SimpleCNN, CHANNELS          # noqa: E402
from model.baselines.meanpool_mil import MeanPoolMIL                # noqa: E402
from src.training.metrics import classification_metrics             # noqa: E402
from src.training import train_baseline as tb                       # noqa: E402

TEST_SPLIT_SHA256 = "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
SEED = 20260918


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bd = _load("phase6b_bag_dataset", "src/mil/bag_dataset.py")
p5 = _load("phase6b_p5_pipeline", "src/preprocessing/pipeline.py")

RUNNERS = []


def check(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
                print(f"PASS  {name}")
                return True
            except Exception as e:  # noqa: BLE001
                print(f"FAIL  {name}: {e!r}")
                return False
        wrapper._label = name
        RUNNERS.append(wrapper)
        return wrapper
    return deco


# ------------------------------------------------------------------ 01
@check("V6B-01 B1 forward-pass shape (N,1) + params in 150k-250k")
def t01():
    m = SimpleCNN()
    n = m.param_count(m)
    assert 150_000 <= n <= 250_000, f"B1 params {n} outside 150k-250k"
    out = m(torch.randn(5, 1, 224, 224))
    assert out.shape == (5, 1), out.shape


# ------------------------------------------------------------------ 02
@check("V6B-02 B4 forward: one bag / multiple bags / different sizes")
def t02():
    m = MeanPoolMIL()
    assert m.forward_stacked(torch.randn(14, 1, 224, 224)).shape == (1, 1)
    assert m.forward_stacked(torch.randn(1, 1, 224, 224)).shape == (1, 1)
    assert m.forward_stacked(torch.randn(53, 1, 224, 224)).shape == (1, 1)
    xp = torch.randn(3, 53, 1, 224, 224)
    mask = torch.zeros(3, 53, dtype=torch.bool)
    mask[0, :14] = True
    mask[1, :21] = True
    mask[2, :53] = True
    assert m.forward_padded(xp, mask).shape == (3, 1)
    try:
        m.forward_padded(xp, torch.zeros_like(mask))
        raise AssertionError("fully-masked bag must raise")
    except ValueError:
        pass


# ------------------------------------------------------------------ 03
@check("V6B-03 one raw binary logit per image/bag; sigmoid in (0,1)")
def t03():
    m1 = SimpleCNN()
    m4 = MeanPoolMIL()
    lo = m1(torch.randn(7, 1, 224, 224))
    assert lo.shape == (7, 1) and lo.dtype == torch.float32
    p = torch.sigmoid(lo)
    assert bool((p > 0).all() and (p < 1).all())
    assert m4.forward_stacked(torch.randn(21, 1, 224, 224)).shape == (1, 1)


# ------------------------------------------------------------------ 04
@check("V6B-04 B4 truly uses MEAN pooling (invariance proof, no attention)")
def t04():
    torch.manual_seed(SEED)
    m = MeanPoolMIL().eval()
    emb = torch.randn(8, 1, 224, 224)
    base = m.forward_stacked(emb)
    perm = torch.randperm(8)
    assert torch.allclose(m.forward_stacked(emb[perm]), base, atol=1e-5), \
        "logit changed under instance permutation -> not a pure mean"
    dup = torch.cat([emb, emb], dim=0)
    assert torch.allclose(m.forward_stacked(dup), base, atol=1e-4), \
        "logit changed under instance duplication -> not a pure mean"
    xp = torch.zeros(1, 12, 1, 224, 224)
    xp[0, :8] = emb
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, :8] = True
    assert torch.allclose(m.forward_padded(xp, mask), base, atol=1e-5), \
        "padded zeros influenced the masked mean"
    mods = {type(x).__name__ for x in m.modules()}
    assert not any("ttention" in t for t in mods), mods


# ------------------------------------------------------------------ 05
@check("V6B-05 label mapping benign=0 malignant=1 (loader + manifests)")
def t05():
    assert bd.LABEL_MAP == {"benign": 0, "malignant": 1}
    ds = bd.UltrasoundBagDataset("val", root=str(REPO))
    for b in ds.bags:
        assert b.numeric_label == bd.LABEL_MAP[b.label]
        for i in b.instances:
            assert i.numeric_label == b.numeric_label


# ------------------------------------------------------------------ 06
@check("V6B-06 dataset interface reuse: loader is the only bag source")
def t06():
    src = (REPO / "src" / "training" / "train_baseline.py").read_text(
        encoding="utf-8")
    assert "UltrasoundBagDataset" in src
    ds = bd.UltrasoundBagDataset("test", root=str(REPO))
    assert len(ds) == 73 and len(ds.instances) == 1350


# ------------------------------------------------------------------ 07
@check("V6B-07 no cross-split source_group_id overlap")
def t07():
    sets = {}
    for s in ("train", "val", "test"):
        ds = bd.UltrasoundBagDataset(s, root=str(REPO))
        sets[s] = {b.source_group_id for b in ds.bags}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])
    assert sum(map(len, sets.values())) == 496


# ------------------------------------------------------------------ 08
@check("V6B-08 validation/test deterministic (repeat loads identical; no aug)")
def t08():
    for s in ("val", "test"):
        ds = bd.UltrasoundBagDataset(s, root=str(REPO))
        b0 = ds.bags[0]
        assert np.array_equal(ds.load_bag_images(b0), ds.load_bag_images(b0))
    ds = bd.UltrasoundBagDataset("test", root=str(REPO))
    arr = ds.load_instance_image(ds.bags[0].instances[0])
    try:
        p5.augment(Image.fromarray(arr), "test", random.Random(0))
        raise AssertionError("augment() accepted split='test'")
    except p5.PreprocessingError:
        pass


# ------------------------------------------------------------------ 09
@check("V6B-09 train-only augmentation: seeded flip ~p=0.5 (val/test raise)")
def t09():
    ds = bd.UltrasoundBagDataset("train", root=str(REPO))
    arr = ds.load_instance_image(ds.bags[0].instances[0])
    rng = random.Random(SEED)
    flips = sum(1 for _ in range(200)
                if not np.array_equal(
                    np.asarray(p5.augment(Image.fromarray(arr), "train", rng)),
                    arr))
    assert 70 <= flips <= 130, f"flip rate {flips}/200 not ~p=0.5"
    f = tb._phase5_augment(arr, rng)
    assert f.dtype == np.float32 and f.shape == arr.shape


# ------------------------------------------------------------------ 10+11
@check("V6B-10/11 checkpoint save/load round-trip -> identical outputs")
def t10():
    torch.manual_seed(SEED)
    m = SimpleCNN()
    cfg = {"b1_simple_cnn": {"checkpoint": "unused", "max_epochs": 1,
                             "patience": 5, "learning_rate": 1e-3,
                             "batch_size": 32, "threshold": 0.5}}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "best.pt")
        tb.save_checkpoint(p, m, epoch=3, seed=SEED, val_metric=0.87, cfg=cfg)
        m2 = SimpleCNN()
        payload = tb.load_checkpoint(p, m2)
        for k in ("model_state_dict", "epoch", "seed", "validation_metric",
                  "config", "preprocessing_version", "pipeline_version",
                  "frozen_test_sha256"):
            assert k in payload, f"checkpoint missing {k}"
        assert payload["seed"] == SEED
        assert payload["frozen_test_sha256"] == TEST_SPLIT_SHA256
        x = torch.randn(2, 1, 224, 224)
        m.eval()
        m2.eval()
        assert torch.allclose(m(x), m2(x), atol=1e-6)
    # B4 round trip
    torch.manual_seed(SEED)
    m4 = MeanPoolMIL()
    m4b = MeanPoolMIL()
    xb = torch.randn(16, 1, 224, 224)
    m4.eval()
    with torch.no_grad():
        ref = m4.forward_stacked(xb)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "b4.pt")
        tb.save_checkpoint(p, m4, epoch=1, seed=SEED, val_metric=0.5,
                           cfg={"b4_meanpool_mil": {"checkpoint": "unused"}})
        tb.load_checkpoint(p, m4b)
        m4b.eval()
        with torch.no_grad():
            assert torch.allclose(ref, m4b.forward_stacked(xb), atol=1e-6)


# ------------------------------------------------------------------ 12
@check("V6B-12 metrics vs hand-computed values (incl. undefined cases)")
def t12():
    y = [0, 0, 1, 1, 1, 0, 1, 0]
    p = [0.1, 0.4, 0.35, 0.8, 0.7, 0.2, 0.9, 0.05]
    pred = [0, 1, 0, 1, 1, 0, 1, 0]
    m = classification_metrics(y, p, y_pred=pred)
    assert (m["TP"], m["FN"], m["TN"], m["FP"]) == (3, 1, 3, 1)
    assert abs(m["sensitivity"] - 3 / 4) < 1e-12
    assert abs(m["specificity"] - 3 / 4) < 1e-12
    assert abs(m["precision"] - 3 / 4) < 1e-12
    assert abs(m["recall"] - 3 / 4) < 1e-12
    assert abs(m["f1"] - 0.75) < 1e-12
    assert abs(m["accuracy"] - 6 / 8) < 1e-12
    assert m["undefined"] == []
    m2 = classification_metrics([0, 0], [0.2, 0.8])
    assert m2["sensitivity"] is None and m2["roc_auc"] is None
    assert "roc_auc (single-class truth)" in m2["undefined"]
    m3 = classification_metrics([1, 1], [0.2, 0.8], y_pred=[0, 1])
    assert m3["specificity"] is None
    assert "specificity (TN + FP == 0)" in m3["undefined"]


# ------------------------------------------------------------------ 13
@check("V6B-13 ROC-AUC / PR-AUC (perfect separation + hand-checked case)")
def t13():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.1, 0.8, 0.9])
    assert abs(m["roc_auc"] - 1.0) < 1e-12
    assert abs(m["pr_auc"] - 1.0) < 1e-12
    # Hand-checked non-trivial case (verified by enumeration):
    # y=[0,1,1,0], p=[0.5,0.9,0.4,0.2] -> concordant pairs 3/4 -> ROC 0.75;
    # AP = 0.5*1.0 + 0.5*(2/3) = 0.833333...
    m2 = classification_metrics([0, 1, 1, 0], [0.5, 0.9, 0.4, 0.2])
    assert abs(m2["roc_auc"] - 0.75) < 1e-12
    assert abs(m2["pr_auc"] - (0.5 + 1.0 / 3.0)) < 1e-12


# ------------------------------------------------------------------ 14
@check("V6B-14 seeded short-run reproducibility (B1, 1 epoch, 6 bags)")
def t14():
    outs = []
    for _run in range(2):
        tb.set_seed(SEED)
        ds = bd.UltrasoundBagDataset("train", root=str(REPO))
        bags = ds.bags[:6]
        xs, ys = [], []
        flip_rng = random.Random(SEED + 1)
        for b in bags:
            for inst in b.instances:
                a = ds.load_instance_image(inst)
                a = tb._phase5_augment(a, flip_rng)
                xs.append(a)
                ys.append(b.numeric_label)
        x = torch.from_numpy(np.stack(xs)).unsqueeze(1).float()
        y = torch.tensor(ys, dtype=torch.float32)
        perm = list(range(x.shape[0]))
        random.Random(SEED).shuffle(perm)
        x, y = x[perm], y[perm]
        torch.manual_seed(SEED + 42)
        model = SimpleCNN()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lf = torch.nn.BCEWithLogitsLoss()
        tot = 0.0
        last_logits = None
        for s in range(0, x.shape[0], 64):
            opt.zero_grad()
            last_logits = model(x[s:s + 64]).squeeze(1)
            loss = lf(last_logits, y[s:s + 64])
            loss.backward()
            opt.step()
            tot += float(loss.item())
        outs.append((round(tot, 6), round(float(last_logits.sum().item()), 6)))
    assert outs[0] == outs[1], f"runs diverged: {outs}"


# ------------------------------------------------------------------ 15
@check("V6B-15 configuration validation (owner-approved values)")
def t15():
    import yaml
    cfg = yaml.safe_load(open(REPO / "configs" / "baseline_config.yaml",
                              encoding="utf-8"))
    assert cfg["config"]["seed"] == 20260918
    assert cfg["config"]["device"] == "cpu"
    b1, b4 = cfg["b1_simple_cnn"], cfg["b4_meanpool_mil"]
    assert b1["input_shape"] == [1, 224, 224] == b4["input_shape"]
    assert b1["optimizer"] == b4["optimizer"] == "Adam"
    assert b1["learning_rate"] == b4["learning_rate"] == 0.001
    assert b1["max_epochs"] == b4["max_epochs"] == 20
    assert b1["early_stopping"]["metric"] == "bag_roc_auc"
    assert b1["early_stopping"]["patience"] == b4["early_stopping"]["patience"] == 5
    assert b1["threshold"] == 0.5 == b4["threshold"]
    assert cfg["data"]["frozen_test_sha256"] == TEST_SPLIT_SHA256
    assert cfg["b3_bag_aggregation"]["inherits"] == "b1_simple_cnn/best.pt"
    assert cfg["b2_transfer_learning"]["status"].startswith("DEFERRED")
    assert list(b1["channels"]) == list(CHANNELS) == list(b4["channels"])


# ------------------------------------------------------------------ 16
@check("V6B-16 artifact schema validation (executed experiment dirs)")
def t16():
    base = REPO / "experiments" / "baseline_results"
    required = {"config.yaml", "train_log.csv", "metrics.json",
                "seed.txt", "env.txt", "provenance.json"}
    for d in ("b1", "b3", "b4"):
        dd = base / d
        assert dd.is_dir(), f"missing experiments/baseline_results/{d}"
        missing = required - {p.name for p in dd.iterdir()}
        assert not missing, f"{d}: missing artifacts {missing}"
        met = json.load(open(dd / "metrics.json", encoding="utf-8"))
        assert met["seed"] == SEED
        prov = met["provenance"]
        assert prov["frozen_test_sha256"] == TEST_SPLIT_SHA256
        assert len(prov["manifest_hashes"]) == 3
        if d == "b3":
            assert prov["inherited_checkpoint"] == \
                "model/baselines/b1_simple_cnn/best.pt"
        else:
            assert prov["inherited_checkpoint"] is None
        for level in ("image_level", "bag_level"):
            if level in met:
                for k in ("accuracy", "sensitivity", "specificity",
                          "precision", "recall", "f1", "roc_auc", "pr_auc",
                          "confusion_matrix"):
                    assert k in met[level], f"{d}/{level} missing {k}"


# ------------------------------------------------------------------ 17
@check("V6B-17 phase boundary scan (no attention/DA-MIL/B2/pretrained/MRI)")
def t17():
    import ast
    targets = ["model/baselines/simple_cnn.py",
               "model/baselines/meanpool_mil.py",
               "src/training/train_baseline.py",
               "src/training/metrics.py"]
    banned = {"attention", "Attention", "MultiheadAttention",
              "DualAttention", "resnet", "ResNet", "efficientnet",
              "densenet", "MRI", "mri", "wandb", "mlflow", "tensorboard"}
    for rel in targets:
        tree = ast.parse((REPO / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in banned, (rel, node.id)
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned, (rel, node.attr)
    assert not (REPO / "model" / "baselines" / "transfer_learning.py").exists()
    # Phase 8 (owner-approved roadmap) places the Dual Attention MIL model
    # under src/models/; exactly those authorized files are exempt. ANY
    # OTHER file under src/models/ still fails this guard (future-guard
    # retained, file-level exception pattern per V-11/V2-11 precedent).
    phase8_approved_models = {
        "__init__.py",
        "dual_attention_mil.py",
    }
    models_dir = REPO / "src" / "models"
    if models_dir.exists():
        for py in sorted(models_dir.glob("*.py")):
            assert py.name in phase8_approved_models, (
                "unauthorized file in src/models/: "
                f"{py.name} (Phase 6 boundary guard retained)")
    src_all = "\n".join((REPO / rel).read_text(encoding="utf-8")
                        for rel in targets)
    assert "torchvision.models" not in src_all
    assert "torch.hub" not in src_all
    assert "urlopen" not in src_all and "requests.get" not in src_all
    gl = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "*.pt" in gl and "*.pth" in gl


# ------------------------------------------------------------------ run
def main() -> int:
    results = [(w._label, w()) for w in RUNNERS]
    npass = sum(1 for _, ok in results if ok)
    print(f"\n{npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
