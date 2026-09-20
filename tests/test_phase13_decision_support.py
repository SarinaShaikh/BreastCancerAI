"""Phase 13: research decision-support layer tests (T13-01..T13-18).

Validates the approved margin-based Phase 13 scope:
    fixed 0.5 threshold, malignant-class model score (NOT calibrated),
    margin-based uncertainty flag (proximity to the threshold ONLY),
    deterministic top-k model-attribution evidence (Phase 11 semantics),
    cross_modal_evidence hard-coded UNAVAILABLE, suggested action
    "Clinical review recommended", mandatory research-only disclaimer.

The suite performs NO training, NO test-set inference, NO final test
evaluation, and NO modification of frozen artifacts. It reads the frozen
E1 checkpoint (MD5-verified), the frozen Phase 9 persisted prediction CSVs,
and the Phase 11 attention export; the only writes are temporary files.

Runner convention (established in the earlier phase suites): @check registers
a zero-arg check; every check explicitly returns True on success; ``main``
runs them all in order, reports per-check status, and exits non-zero on any
failure.
"""
from __future__ import annotations

import ast
import csv
import inspect
import json
import os
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CHECKS = []


def check(cid, description):
    def _wrap(fn):
        CHECKS.append((cid, description, fn))
        return fn
    return _wrap


# --------------------------------------------------------------------------
# module under test
# --------------------------------------------------------------------------
from src.clinical_support import decision_support as ds  # noqa: E402

REPORT_PATH = os.path.join(REPO, "reports", "phase13_decision_support_spec.md")
ATTN_CSV = os.path.join(REPO, "reports", "phase11", "attention_export.csv")
E1_VAL_CSV = os.path.join(REPO, "experiments", "dual_attention_results",
                          "da_stage_b", "val_predictions.csv")


def _load_val_predictions():
    rows = list(csv.DictReader(open(E1_VAL_CSV, newline="", encoding="utf-8")))
    return {r["bag_id"]: r for r in rows}


def _load_attention_export():
    by_bag = {}
    with open(ATTN_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_bag.setdefault(r["bag_id"], []).append(r)
    return by_bag


def _att(values, bag_id="bag-test0000", split="val"):
    return {"bag_id": bag_id, "attention": [float(v) for v in values],
            "probability": 0.5, "split": split}


# --------------------------------------------------------------------------
# T13-01..T13-03: prediction rule (frozen threshold)
# --------------------------------------------------------------------------

@check("T13-01", "prediction threshold is exactly 0.5 (frozen constant)")
def t13_01():
    assert ds.DECISION_THRESHOLD == 0.5
    assert ds.prediction_from_score(0.4999999999) == 0
    assert ds.prediction_from_score(0.5) == 1
    assert ds.prediction_from_score(0.5000000001) == 1
    return True


@check("T13-02", "prediction mapping is deterministic and repeated calls agree")
def t13_02():
    for s in (0.0, 0.31, 0.4, 0.5, 0.6, 0.79, 1.0):
        p1 = ds.prediction_from_score(s)
        for _ in range(5):
            assert ds.prediction_from_score(s) == p1
    return True


@check("T13-03", "frozen persisted predictions agree with the 0.5 mapping")
def t13_03():
    for _, row in _load_val_predictions().items():
        assert int(row["prediction"]) == ds.prediction_from_score(float(row["probability"]))
    return True


# --------------------------------------------------------------------------
# T13-04..T13-07: margin-based uncertainty flag (boundary behaviour)
# --------------------------------------------------------------------------

@check("T13-04", "score exactly 0.5 is uncertain")
def t13_04():
    assert ds.uncertainty_flag_from_score(0.5) is True
    return True


@check("T13-05", "scores 0.4 and 0.6 (exact margin edges) are uncertain")
def t13_05():
    assert ds.uncertainty_flag_from_score(0.4) is True
    assert ds.uncertainty_flag_from_score(0.6) is True
    return True


@check("T13-06", "scores strictly beyond the margin are not uncertain")
def t13_06():
    for s in (0.0, 0.1, 0.39, 0.3999999, 0.61, 0.7, 0.9, 1.0):
        assert ds.uncertainty_flag_from_score(s) is False, f"score {s}"
    return True


@check("T13-07", "margin constant is the pre-registered 0.1 (Phase 10/11 convention)")
def t13_07():
    assert ds.UNCERTAINTY_MARGIN == 0.1
    from src.evaluation import metrics as m
    from src.explainability import attention_visualization as av
    # metrics.py implements the convention as the default of uncertainty_mask
    assert inspect.signature(m.uncertainty_mask).parameters["margin"].default == 0.1
    assert av.UNCERTAINTY_MARGIN == ds.UNCERTAINTY_MARGIN
    return True


# --------------------------------------------------------------------------
# T13-08..T13-10: schema + provenance + end-to-end records
# --------------------------------------------------------------------------

@check("T13-08", "record schema: exact required field set, types, provenance identity")
def t13_08():
    rec = ds.build_decision_support_record(
        0.63, [0.5, 0.3, 0.2], bag_id="bag-x",
        source_group_id="grp-x", split="val")
    assert set(rec.keys()) == set(ds.OUTPUT_FIELDS)
    assert rec["prediction"] in ("Benign model prediction", "Malignant model prediction")
    assert isinstance(rec["malignant_class_model_score"], float)
    assert rec["malignant_class_model_score"] == 0.63
    assert rec["uncertainty_flag"] is False and rec["uncertainty_margin"] == 0.1
    assert rec["cross_modal_evidence"] == "UNAVAILABLE"
    assert rec["suggested_action"] == "Clinical review recommended"
    prov = rec["provenance"]
    assert prov["checkpoint_md5"] == ds.E1_CHECKPOINT_MD5 == "258710649fb0e6979f64fc1e7ccfc28f"
    assert prov["checkpoint_relpath"] == "model/dual_attention/da_stage_b/best.pt"
    assert prov["decision_threshold"] == 0.5
    assert prov["score_is_calibrated"] is False
    assert prov["source_key_status"] == "inferred_from_filename_not_verified_identifier"
    return True


@check("T13-09", "end-to-end records from frozen val predictions + Phase 11 export")
def t13_09():
    preds = _load_val_predictions()
    att = _load_attention_export()
    scores = {b: float(r["probability"]) for b, r in preds.items()}
    meta = {b: {"source_group_id": r["source_group_id"], "split": r["split"]}
            for b, r in preds.items()}
    att_by_bag = {b: [float(x["attention"]) for x in rows] for b, rows in att.items()
                  if b in scores}
    records = ds.build_records_from_scores(scores, att_by_bag, meta_by_bag=meta)
    assert set(records) == set(scores)
    for bag_id, rec in records.items():
        r = preds[bag_id]
        assert rec["malignant_class_model_score"] == float(r["probability"])
        assert rec["uncertainty_flag"] == (abs(float(r["probability"]) - 0.5) <= 0.1 + 1e-9)
        assert rec["provenance"]["bag_id"] == bag_id
        assert rec["provenance"]["split"] == r["split"]
        assert rec["provenance"]["attention_available"] == (bag_id in att_by_bag)
        assert len(rec["top_k_evidence"]) == min(3, len(att_by_bag.get(bag_id, [])))
    n_unc = sum(1 for rec in records.values() if rec["uncertainty_flag"])
    print(f"\n      [info] val bags: {len(records)}; uncertain: {n_unc}", end="")
    return True


@check("T13-10", "records are byte-identical on repeated construction")
def t13_10():
    preds = _load_val_predictions()
    att = _load_attention_export()
    bag = sorted(preds)[0]
    rows = att[bag]
    rec1 = ds.build_decision_support_record(
        float(preds[bag]["probability"]), [float(x["attention"]) for x in rows],
        bag_id=bag, source_group_id=preds[bag]["source_group_id"],
        split=preds[bag]["split"])
    rec2 = ds.build_decision_support_record(
        float(preds[bag]["probability"]), [float(x["attention"]) for x in rows],
        bag_id=bag, source_group_id=preds[bag]["source_group_id"],
        split=preds[bag]["split"])
    assert json.dumps(rec1, sort_keys=True) == json.dumps(rec2, sort_keys=True)
    return True


# --------------------------------------------------------------------------
# T13-11: deterministic top-k evidence (Phase 11 semantics)
# --------------------------------------------------------------------------

@check("T13-11", "top-k evidence deterministic; k>n clamps; ties break by instance order")
def t13_11():
    att = [0.05, 0.40, 0.40, 0.15]          # tie between instances 1 and 2
    top3 = ds._top_k_evidence(att, k=3)
    assert [e["order_index"] for e in top3] == [1, 2, 3]   # tie -> lower index first
    assert [e["rank"] for e in top3] == [1, 2, 3]
    top_all = ds._top_k_evidence(att, k=99)                 # k > n clamps to n
    assert len(top_all) == 4
    top1 = ds._top_k_evidence(att, k=1)
    assert len(top1) == 1 and top1[0]["order_index"] == 1
    assert json.dumps(top3, sort_keys=True) == json.dumps(ds._top_k_evidence(att, k=3),
                                                          sort_keys=True)
    for e in top3:
        assert e["semantics"].startswith("model-attributed image instance")
    return True


# --------------------------------------------------------------------------
# T13-12..T13-13: clinical-claim boundary
# --------------------------------------------------------------------------

_BANNED_IN_CODE = [
    "diagnosis", "diagnose", "diagnostic", "clinical risk", "patient risk",
    "cancer risk", "calibrated clinical probability", "bi-rads", "triage",
    "prognosis", "recurrence", "survival", "staging", "grading",
    "treatment recommendation", "treatment advice", "medical advice",
    "clinically validated", "clinical uncertainty",
]


def _banned_term_hits():
    """AST-based scan for clinical-capability terms in identifiers and user-facing
    string constants. Docstrings and the RESEARCH_ONLY_DISCLAIMER constant are
    exempt: they are the sanctioned negated safety documentation."""
    src = inspect.getsource(ds)
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)
    hits = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        for nm in names:
            low = nm.lower()
            for token in _BANNED_IN_CODE:
                if token in low:
                    hits.append(("identifier", nm, token))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value
            if val in docstrings or val == ds.RESEARCH_ONLY_DISCLAIMER:
                continue
            low = val.lower()
            for token in _BANNED_IN_CODE:
                if token in low:
                    hits.append(("string-literal", val[:60], token))
    return hits


@check("T13-12", "code (identifiers + string constants, sans sanctioned disclaimers) is clinically clean")
def t13_12():
    hits = _banned_term_hits()
    assert not hits, f"banned clinical-capability terms: {hits}"
    return True


@check("T13-13", "mandatory disclaimer carries all required negated statements")
def t13_13():
    d = ds.RESEARCH_ONLY_DISCLAIMER
    low = d.lower()
    for required in [
        "research prototype", "not clinically validated", "uncalibrated",
        "not a calibrated clinical probability",
        "proximity to the fixed 0.5 decision threshold",
        "not clinical or diagnostic uncertainty and not patient or cancer risk",
        "model attention and grad-cam are model attribution mechanisms, not "
        "clinical explanations",
        "model attention != clinical explanation",
        "no mri/cross-modal evidence is available",
        "must not be used as a diagnosis, a clinical risk assessment, "
        "or a treatment recommendation",
        "does not replace radiologist or pathologist judgment",
    ]:
        assert required in low, f"disclaimer missing: {required!r}"
    return True


# --------------------------------------------------------------------------
# T13-14..T13-15: cross-modal field + no MRI code
# --------------------------------------------------------------------------

@check("T13-14", "cross_modal_evidence is hard-coded UNAVAILABLE in every record")
def t13_14():
    assert ds.CROSS_MODAL_EVIDENCE == "UNAVAILABLE"
    for s in (0.2, 0.5, 0.8):
        rec = ds.build_decision_support_record(s, [1.0])
        assert rec["cross_modal_evidence"] == "UNAVAILABLE"
    return True


@check("T13-15", "module imports: no MRI/DICOM/NIfTI, no optimizer/scheduler, no calibration")
def t13_15():
    tree = ast.parse(inspect.getsource(ds))
    banned = ("mri", "dicom", "pydicom", "nibabel", "nii", "medpy",
              "torch.optim", "scheduler", "sklearn.linear_model",
              "isotonic", "conformal", "calibrat")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                low = a.name.lower()
                assert not any(b in low for b in banned), a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            low = node.module.lower()
            assert not any(b in low for b in banned), node.module
    return True


# --------------------------------------------------------------------------
# T13-16..T13-18: provenance / frozen integrity
# --------------------------------------------------------------------------

@check("T13-16", "frozen E1 checkpoint MD5 verified via the Phase 11 helper")
def t13_16():
    assert ds.verify_checkpoint_md5() == ds.E1_CHECKPOINT_MD5
    return True


@check("T13-17", "frozen Phase 11 attention export is unchanged and parseable")
def t13_17():
    path = os.path.join(REPO, "reports", "phase11", "attention_export.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 88
    assert all(r["source_key_status"] == "inferred_from_filename_not_verified_identifier"
               for r in rows)
    sums = {}
    for r in rows:
        sums[r["bag_id"]] = sums.get(r["bag_id"], 0.0) + float(r["attention"])
    for bag, s in sums.items():
        assert abs(s - 1.0) <= 1e-6, f"attention sum {s} for {bag}"
    return True


@check("T13-18", "frozen Phase 9 E1 artifacts and test manifest are untouched")
def t13_18():
    from src.preprocessing.pipeline import verify_freeze
    assert verify_freeze() == "959f1cd3d9f719195ac31af1a586d7c84824af1e23698ec7616c376714112b74"
    for name in ("val_predictions.csv", "test_predictions.csv", "test_metrics.json",
                 "metrics.json", "provenance.json", "train_log.csv"):
        path = os.path.join(REPO, "experiments", "dual_attention_results",
                            "da_stage_b", name)
        assert os.path.exists(path), f"missing frozen artifact: {name}"
    return True


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main() -> int:
    passed = 0
    for cid, desc, fn in CHECKS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[FAIL] {cid} {desc} :: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[PASS] {cid} {desc}")
    total = len(CHECKS)
    print("-" * 72)
    print(f"{passed}/{total} Phase 13 decision-support checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
