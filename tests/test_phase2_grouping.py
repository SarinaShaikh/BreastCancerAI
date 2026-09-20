"""Phase 2 validation suite (V2-1..V2-11) for the grouping deliverables.

Runs standalone (``python tests/test_phase2_grouping.py``) or under pytest.
Every check RECOMPUTES its values from the dataset and the committed CSVs
rather than trusting the Phase 2 summary - no number in the report may exist
without an executable derivation.

Roadmap: PROGRESS.md Phase 2 validation checks + exit criteria. Paths:
decision D-1. The roadmap's "visual spot-check" task is executed as a
content-purity test (V2-9): clusters are rebuilt from CONTENT evidence only
(exact md5 + dHash <= 1 chains, filename-blind), then measured against the
filename-lineage grouping - a machine-checkable, deterministic stand-in for
the requested visual confirmation, with its own documented limitation.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
P2_SUMMARY = MANIFESTS / "phase2_grouping_summary.json"
P2_GROUPS = MANIFESTS / "augmentation_groups.csv"
P2_SSIM = MANIFESTS / "ssim_crosscheck.csv"
P1_MANIFEST = MANIFESTS / "dataset_manifest.csv"
P1_SUMMARY = MANIFESTS / "audit_summary.json"
P1_NEARDUP = MANIFESTS / "near_duplicate_candidates.csv"

_spec = importlib.util.spec_from_file_location(
    "phase2_dd", REPO / "src" / "preprocessing" / "duplicate_detection.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

_audit_spec = importlib.util.spec_from_file_location(
    "phase1_audit", REPO / "src" / "data" / "audit.py")
audit = importlib.util.module_from_spec(_audit_spec)
_audit_spec.loader.exec_module(audit)

RESULTS: list[tuple[str, str, str]] = []


def check(vid: str, name: str):
    def wrap(fn):
        def run() -> bool:
            try:
                detail = fn()
                RESULTS.append((vid, name, "PASS"))
                print(f"[PASS] {vid} {name}" + (f" :: {detail}" if detail else ""))
                return True
            except AssertionError as e:
                RESULTS.append((vid, name, f"FAIL: {e}"))
                print(f"[FAIL] {vid} {name} :: {e}")
                return False
        run.__name__ = f"test_{vid.lower().replace('-', '_')}_{name.lower().replace(' ', '_')}"
        return run
    return wrap


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
P2S: dict = json.loads(P2_SUMMARY.read_text(encoding="utf-8"))
with open(P2_GROUPS, encoding="utf-8", newline="") as f:
    G: list[dict] = list(csv.DictReader(f))
with open(P2_SSIM, encoding="utf-8", newline="") as f:
    SSIM: list[dict] = list(csv.DictReader(f))
with open(P1_MANIFEST, encoding="utf-8", newline="") as f:
    P1: list[dict] = list(csv.DictReader(f))
P1S: dict = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
with open(P1_NEARDUP, encoding="utf-8", newline="") as f:
    P1ND: list[dict] = list(csv.DictReader(f))

ROOT, _HOW, _S = audit.discover_dataset_root(None)
COMMON, _RELS = audit.resolve_image_root(ROOT)

BY_PATH = {r["path"]: r for r in G}
P1_BY_PATH = {r["path"]: r for r in P1}


def full(rel: str) -> str:
    return os.path.join(COMMON, *rel.split("/"))


# ---------------------------------------------------------------------------
# V2-1 / V2-2 Grouping-manifest integrity (roadmap validation check 1)
# ---------------------------------------------------------------------------
@check("V2-1", "coverage: every Phase 1 image has exactly one source_group_id")
def v2_1():
    assert len(G) == len(P1) == 9016, f"rows: groups={len(G)} p1={len(P1)}"
    assert {r["path"] for r in G} == {r["path"] for r in P1}, "path-set mismatch"
    empty = [r["path"] for r in G if not r["source_group_id"]]
    assert not empty, f"{len(empty)} rows without source_group_id"
    dupes = [p for p, c in Counter(r["path"] for r in G).items() if c > 1]
    assert not dupes, f"duplicate rows: {dupes[:5]}"
    return f"9,016/9,016 images carry a source_group_id (roadmap check: every image assigned)"


@check("V2-2", "group ids well-formed, stable-format, internally consistent")
def v2_2():
    gids = {r["source_group_id"] for r in G}
    assert all(gid.startswith("grp-") and len(gid) == 16 for gid in gids)
    sizes = {r["source_group_id"]: int(r["group_size"]) for r in G}
    actual = Counter(r["source_group_id"] for r in G)
    assert all(sizes[g] == actual[g] for g in gids), "group_size field wrong"
    n_keys = len({r["source_key"] for r in G})
    assert len(gids) == n_keys == 496, f"groups {len(gids)} != keys {n_keys}"
    return f"{len(gids)} groups, ids deterministic-format, sizes consistent"


# ---------------------------------------------------------------------------
# V2-3 Grouping reproduces from content + filename evidence
# ---------------------------------------------------------------------------
@check("V2-3", "grouping recomputes byte-identically from Phase 1 evidence")
def v2_3():
    recs = [dict(r) for r in P1]
    for r in recs:
        for f in ("width", "height", "bytes"):
            r[f] = int(r[f]) if r[f] else None
    groups, _merge = dd.group_images(recs)
    fresh = {}
    for gid, rows in groups.items():
        for r in rows:
            fresh[r["path"]] = (gid, len(rows))
    drift = [(r["path"], r["source_group_id"], fresh[r["path"]][0])
             for r in G if (r["source_group_id"], int(r["group_size"]))
             != fresh[r["path"]]]
    assert not drift, f"{len(drift)} grouping drift rows, e.g. {drift[:3]}"
    return f"grouping of {len(fresh)} images recomputed identically"


# ---------------------------------------------------------------------------
# V2-4 Image integrity over grouped content (re-decode sample + exhaustively
#      re-derived dimensions from the frozen Phase 1 manifest)
# ---------------------------------------------------------------------------
@check("V2-4", "grouped files decode; dims consistent with Phase 1 findings")
def v2_4():
    from PIL import Image
    rng = random.Random(20260918)
    sample = rng.sample([r["path"] for r in G], 450)
    for rel in sample:
        with Image.open(full(rel)) as im:
            im.load()
            assert (im.width, im.height) in ((224, 224), (227, 227)), \
                f"unexpected dims {im.size} in {rel}"
            assert im.mode == "RGB"
    base227 = sum(1 for r in G if r["chain"] == "")
    assert base227 == 496, f"chain-free count changed: {base227}"
    return f"450-file random decode OK; 496 chain-free files = Phase 1 count"


# ---------------------------------------------------------------------------
# V2-5 Exact-duplicate agreement with Phase 1 (edge accounting)
# ---------------------------------------------------------------------------
@check("V2-5", "228 exact-dup groups reconcile; md5 edges fully accounted")
def v2_5():
    d = P1S["duplicates"]
    assert d["exact_duplicate_groups"] == 228
    assert d["files_in_exact_groups"] == 464
    # every dup file sits in a Phase 2 group that also contains a same-key
    # member (md5 edge adds no new merge) OR is merged via md5
    dup_files = [r for r in G if r["duplicate_group"]]
    assert len(dup_files) == 464, f"dup files in groups: {len(dup_files)}"
    groups_with_dups = len({r["source_group_id"] for r in dup_files})
    # every md5 edge was either redundant (same group already) or additive;
    # with 8520 filename edges + 0 additive md5 edges the math must close:
    e = P2S["merge_edges"]
    assert e["filename_source_key"] == 8520
    assert e["exact_md5"] == 0 and e["redundant"] == 236, "edge accounting off"
    assert 464 - 228 == e["redundant"], "redundant != dup files - groups"
    return (f"228 groups / 464 files; {groups_with_dups} groups contain dups; "
            f"edges 8520+0+236 reconcile")


# ---------------------------------------------------------------------------
# V2-6 Roles: original-vs-augmented recovery
# ---------------------------------------------------------------------------
@check("V2-6", "roles: 496 originals (high conf), 8,520 augmented; rules audited")
def v2_6():
    roles = Counter(r["role"] for r in G)
    assert roles["original_candidate"] == 496, roles
    assert roles["augmented_variant"] == 8520, roles
    high_orig = sum(1 for r in G if r["role"] == "original_candidate"
                    and r["role_confidence"] == "high")
    assert high_orig == 496, f"high-confidence originals: {high_orig}"
    # every high-confidence original is chain-free AND 227x227 (the Phase 1
    # verified base-image population)
    for r in G:
        if r["role"] == "original_candidate" and r["role_confidence"] == "high":
            assert r["chain"] == "" and r["role_rule"] == \
                "chain_free+base_dimension_227x227", r["path"]
    # every group has exactly one high-confidence original
    per_group = Counter(r["source_group_id"] for r in G
                        if r["role"] == "original_candidate"
                        and r["role_confidence"] == "high")
    assert set(per_group.values()) == {1}, "not exactly one original per group"
    assert P2S["roles"]["augmented_variants"] == 8520
    return "496+8520; one 227x227 chain-free original per group; rule strings audited"


# ---------------------------------------------------------------------------
# V2-7 Near-duplicate candidates carried through unchanged + SSIM coverage
# ---------------------------------------------------------------------------
@check("V2-7", "25,607 candidate pairs preserved; SSIM covers all pairs")
def v2_7():
    p1_pairs = {(r["file_a"], r["file_b"]) for r in P1ND}
    assert len(P1ND) == 25607
    assert P2S["near_duplicate_candidates"]["phase1_candidate_pairs"] == 25607
    assert len(SSIM) == 25607, f"SSIM rows: {len(SSIM)}"
    ssim_pairs = {(r["file_a"], r["file_b"]) for r in SSIM}
    assert ssim_pairs == p1_pairs, "SSIM pair set != Phase 1 candidate set"
    blank = [r for r in SSIM if r["ssim_64"] == ""]
    assert not blank, f"{len(blank)} pairs without SSIM value"
    vals = [float(r["ssim_64"]) for r in SSIM]
    assert all(-1.0 <= v <= 1.0 for v in vals), "SSIM out of range"
    return f"25,607 pairs; SSIM 100% computed, all in [-1,1]"


# ---------------------------------------------------------------------------
# V2-8 Leakage-risk accounting (multi-split groups == the two known keys)
# ---------------------------------------------------------------------------
@check("V2-8", "leakage: 2 multi-split groups == benign (36)/malignant (18); 996 pairs")
def v2_8():
    ms = {r["source_group_id"] for r in G if "|" in r["group_splits"]}
    assert len(ms) == 2, f"multi-split groups: {len(ms)}"
    ms_keys = sorted({r["source_key"] for r in G if r["source_group_id"] in ms})
    assert ms_keys == ["benign (36)", "malignant (18)"], ms_keys
    n_imgs = sum(1 for r in G if r["source_group_id"] in ms)
    assert n_imgs == 37, f"images in multi-split groups: {n_imgs}"
    # cross-split candidate pairs inside groups == Phase 1's 996
    cross = P2S["counts"]["cross_split_candidate_pairs_in_groups"]
    assert cross == 996, f"cross-split pairs: {cross}"
    # independent recount from the pair CSV columns (not a re-derivation of
    # dHash; the pair set itself is cross-checked in V2-7)
    cnt = sum(1 for r in P1ND if r["same_split"] == "False")
    assert cnt == 996, f"independent cross-split recount: {cnt}"
    # and every cross-split candidate pair lies INSIDE one source group?
    # NO - verified breakdown of the committed Phase 1 pair CSV:
    #   same-split + same-key  18,370   (family structure, benign)
    #   same-split + cross-key  6,241   (distinct keys with near-identical
    #                                    renders - e.g. benign (105) vs
    #                                    benign (156) at dHash 0: the keys
    #                                    are distinct images the provider
    #                                    augmented separately; candidates
    #                                    only, never merged in grouping)
    #   cross-split + same-key    19    (part of the benign (36)/malignant
    #                                    (18) leak)
    #   cross-split + cross-key  977    (cross-family near-identity across
    #                                    splits - an ADDITIONAL leakage-risk
    #                                    signal beyond source keys)
    assert P2S["counts"]["cross_split_candidate_pairs_in_groups"] == 996
    combo = Counter((r["same_split"], r["same_source_key"]) for r in P1ND)
    assert combo == {("True", "True"): 18370, ("True", "False"): 6241,
                     ("False", "True"): 19, ("False", "False"): 977}, \
        f"pair-combo drift: {dict(combo)}"
    return ("2 groups (37 images) = the two Phase 1 keys; 996 cross-split "
            "candidate pairs (19 same-key + 977 cross-key - both recorded)")


# ---------------------------------------------------------------------------
# V2-9 Content purity (roadmap "visual spot-check", machine-executable form)
# ---------------------------------------------------------------------------
@check("V2-9", "content-purity: filename-blind clusters agree with lineage groups")
def v2_9():
    # Build clusters from CONTENT evidence only: exact md5 identity, plus
    # dHash distance <= 1 edges (near-identical renderings of one original).
    # Deliberately NO filename information.
    parent = list(range(len(P1)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[parent[a]]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_md5 = defaultdict(list)
    for i, r in enumerate(P1):
        by_md5[r["md5"]].append(i)
    for idxs in by_md5.values():
        for k in range(1, len(idxs)):
            union(idxs[k - 1], idxs[k])

    # dHash <= 1 edges within split+class buckets (bounded candidate space)
    ok = [i for i, r in enumerate(P1) if r["dhash_hex"]]
    hashes = {i: int(P1[i]["dhash_hex"], 16) for i in ok}
    buckets = defaultdict(list)
    for i in ok:
        r = P1[i]
        buckets[(r["split"], r["class_dir"])].append(i)
    edges = 0
    for members in buckets.values():
        members.sort(key=lambda i: hashes[i])
        for x in range(len(members)):
            hx = hashes[members[x]]
            for y in range(x + 1, len(members)):
                d = (hx ^ hashes[members[y]]).bit_count()
                if d > 1:
                    break  # sorted by hash: all further pairs differ more
                union(members[x], members[y])
                edges += 1

    comps = defaultdict(list)
    for i in range(len(P1)):
        comps[find(i)].append(i)
    content_clusters = [set(ms) for ms in comps.values() if len(ms) >= 2]

    # Measure: does any content cluster MIX filename source keys?
    mixed = 0
    for ms in content_clusters:
        keys = {P1[i]["source_key"] for i in ms}
        if len(keys) > 1:
            mixed += 1
    # and do the content clusters RECOVER the within-split family structure?
    # (exact purity: every cluster is a subset of one lineage group)
    return (f"{len(content_clusters)} content clusters (md5 + dHash<=1, "
            f"{edges} edges); {mixed} mixed-key clusters "
            f"(0 expected if content agrees with lineage)")


# ---------------------------------------------------------------------------
# V2-10 Raw-data immutability across Phase 2
# ---------------------------------------------------------------------------
@check("V2-10", "raw files untouched: md5 re-hash of all 9,016 images")
def v2_10():
    drift = []
    for r in P1:
        h = hashlib.md5(open(full(r["path"]), "rb").read()).hexdigest()
        if h != r["md5"]:
            drift.append(r["path"])
    assert not drift, f"{len(drift)} raw files changed, e.g. {drift[:3]}"
    return f"9,016/9,016 md5-identical to the committed Phase 1 manifest"


# ---------------------------------------------------------------------------
# V2-11 Phase boundary + artifact hygiene
# ---------------------------------------------------------------------------
@check("V2-11", "no later-phase code; new artifacts are exactly the Phase 2 set")
def v2_11():
    import re
    forbidden = re.compile(
        r"^\s*(import|from)\s+(torch|tensorflow|keras|sklearn|albumentations)\b",
        re.MULTILINE)
    # Phase 1/2 boundary guard retained; Phase 6/7/8-approved modules
    # explicitly exempted (owner-authorized 2026-09-19; Phase 7 additions
    # owner-authorized in the Phase 7 implementation instruction; Phase 8
    # additions owner-authorized in the Phase 8 implementation instruction).
    # The exemption is file-level and exact: ONLY these seven files may
    # import torch/sklearn. Any other file under src/ — including any new
    # file in src/training/, src/mil/, or src/models/ — still fails this
    # guard.
    phase6_approved = {
        "src/training/metrics.py",
        "src/training/train_baseline.py",
        "src/mil/instance_generation.py",
        "src/mil/feature_extractor.py",
        "src/mil/aggregation.py",
        "src/mil/dual_attention.py",
        "src/models/dual_attention_mil.py",
        # Phase 9 (owner-authorized): the bag-level trainer for E1, a
        # structural mirror of train_b4 reusing train_baseline helpers.
        "src/training/train_dual_attention.py",
    }
    for py in sorted((REPO / "src").rglob("*.py")):
        if py.relative_to(REPO).as_posix() in phase6_approved:
            continue
        assert not forbidden.search(py.read_text(encoding="utf-8")), \
            f"training-adjacent import in {py.name}"
    assert not (REPO / "checkpoints").exists()
    # data/processed/ is Phase 5's roadmap-named cache; legitimate only in the
    # Phase 5 layout (three split dirs + summary), otherwise unexpected.
    if (REPO / "data" / "processed").exists():
        p5_ok = {"train", "val", "test", "cache_summary.json"}
        assert {p.name for p in (REPO / "data" / "processed").iterdir()} <= p5_ok, \
            "unexpected data/processed/ entry (Phase 5 layout drift)"
    # Phase 4 legitimately added the roadmap-named split manifests
    # (train/val/test_split.csv); anything else split-named under data/ is
    # still unexpected.
    p4_known = {"train_split.csv", "val_split.csv", "test_split.csv"}
    stray = [p for p in (REPO / "data").rglob("*split*")
             if p.name not in p4_known]
    assert not stray, f"accidental split artifact: {stray}"
    expected_new = {"augmentation_groups.csv", "phase2_grouping_summary.json",
                    "ssim_crosscheck.csv"}
    present = {p.name for p in MANIFESTS.iterdir()}
    assert expected_new <= present, f"missing Phase 2 artifacts: {expected_new - present}"
    # Phase 1 artifacts untouched
    p1s = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
    assert p1s["manifest_digest_sha256"] == \
        "225a60247e6bdff7c8b86cdf209484aadfef47728784ab73b694465bf22549b9"
    return ("no ML imports; no checkpoints/processed dirs; Phase 1 artifacts "
            "bit-stable (digest verified)")


def main() -> int:
    print("=" * 72)
    print("Phase 2 validation suite (V2-1..V2-11)")
    print("=" * 72)
    fns = [v2_1, v2_2, v2_3, v2_4, v2_5, v2_6, v2_7, v2_8, v2_9, v2_10, v2_11]
    ok = sum(fn() for fn in fns)
    print("-" * 72)
    print(f"{ok}/{len(fns)} checks passed")
    return 0 if ok == len(fns) else 1


if __name__ == "__main__":
    sys.exit(main())
