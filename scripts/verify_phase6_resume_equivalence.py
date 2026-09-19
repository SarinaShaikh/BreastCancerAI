"""Phase 6 reproducibility evidence: chunk-resume bit-exactness.

Executed check (referenced by reports/phase6_baseline_results.md):
a short training run executed in 1-epoch chunks must produce the SAME
train log and the SAME best-checkpoint weights as one uninterrupted run.

Run:  python scripts/verify_phase6_resume_equivalence.py
Exit 0 iff bit-identical.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import torch  # noqa: E402

from src.training import train_baseline as tb  # noqa: E402


def main() -> int:
    base_cfg = tb._load_baseline_config(_REPO)
    tiny = copy.deepcopy(base_cfg)
    with tempfile.TemporaryDirectory() as td:
        tiny["b1_simple_cnn"]["max_epochs"] = 2
        tiny["b1_simple_cnn"]["batch_size"] = 8
        tiny["b1_simple_cnn"]["checkpoint"] = os.path.join(td, "chunked.pt")
        # Chunked run: two 1-epoch chunks.
        cs = os.path.join(td, "chunk_state.pt")
        _, log_a, info_a = tb.train_b1(tiny, seed=20260918, chunk_epochs=1,
                                       chunk_state_path=cs)
        assert info_a.get("paused"), info_a
        _, log_a, info_a = tb.train_b1(tiny, seed=20260918, chunk_epochs=1,
                                       chunk_state_path=cs)
        assert not info_a.get("paused"), info_a
        # Uninterrupted run.
        tiny["b1_simple_cnn"]["checkpoint"] = os.path.join(td, "whole.pt")
        _, log_b, info_b = tb.train_b1(tiny, seed=20260918)
        # Compare logs exactly EXCLUDING the wall-clock field (epoch_seconds
        # is a timing measurement, not a training result).
        strip = lambda rows: [{k: v for k, v in r.items()
                               if k != "epoch_seconds"} for r in rows]
        la, lb = strip(log_a), strip(log_b)
        if la != lb:
            print("LOG MISMATCH")
            print("chunked:", json.dumps(la, indent=1))
            print("whole:  ", json.dumps(lb, indent=1))
            return 1
        # Compare best-checkpoint weights bit-exactly.
        ma, mb = tb.SimpleCNN(), tb.SimpleCNN()
        pa = torch.load(os.path.join(td, "chunked.pt"), map_location="cpu",
                        weights_only=False)["model_state_dict"]
        pb = torch.load(os.path.join(td, "whole.pt"), map_location="cpu",
                        weights_only=False)["model_state_dict"]
        ma.load_state_dict(pa)
        mb.load_state_dict(pb)
        for (ka, va), (kb, vb) in zip(pa.items(), pb.items()):
            assert ka == kb and torch.equal(va, vb), f"weight drift at {ka}"
        print("chunk-resume equivalence: BIT-EXACT")
        log_summary = [(r["epoch"], r["train_loss"], r["val_bag_roc_auc"])
                       for r in log_a]
        print("  log:", log_summary)
        print("  info chunked=%s" % info_a)
        print("  info whole  =%s" % info_b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
