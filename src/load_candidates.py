"""
load_candidates.py — load the candidate pool and run the cheap, deterministic
Stage A (consistency) + Stage B (JD disqualifiers) passes across all 100k,
caching results to disk so the timed ranking step never has to recompute them.

Usage:
    python src/load_candidates.py --candidates data/raw/candidates.jsonl.gz

This is NOT the timed ranking step — run this once, ahead of time, as part of
precomputation. rank.py loads the cached output instead of re-running this.
"""

from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from consistency_gate import score_candidate
from jd_disqualifiers import apply_disqualifiers


def load_jsonl_gz(path: str | Path) -> list[dict]:
    """Load candidates from either a .jsonl.gz or plain .jsonl file."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"

    candidates = []
    with opener(path, mode, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {line_num}: {e}", file=sys.stderr)
    return candidates


def precompute(candidates: list[dict]) -> dict:
    """Run Stage A + Stage B across the full pool. Cheap (no embeddings, no
    ML) — should take seconds to low minutes even on 100k rows, since every
    check is plain dict/string arithmetic.
    """
    results = {}
    t0 = time.time()

    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", f"UNKNOWN_{i}")
        consistency = score_candidate(c)
        disqualifiers = apply_disqualifiers(c)

        results[cid] = {
            "consistency_score": consistency.consistency_score,
            "consistency_flags": consistency.flags,
            "is_likely_honeypot": consistency.is_likely_honeypot,
            "jd_multiplier": disqualifiers.multiplier,
            "jd_triggered_rules": disqualifiers.triggered_rules,
        }

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  processed {i + 1}/{len(candidates)} ({elapsed:.1f}s elapsed)",
                  file=sys.stderr)

    elapsed = time.time() - t0
    print(f"Precompute done: {len(candidates)} candidates in {elapsed:.1f}s", file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True,
                         help="Path to candidates.jsonl.gz or candidates.jsonl")
    parser.add_argument("--out", default="data/processed/stage_ab_scores.json",
                         help="Where to write the cached Stage A+B results")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...", file=sys.stderr)
    candidates = load_jsonl_gz(args.candidates)
    print(f"Loaded {len(candidates)} candidates.", file=sys.stderr)

    if len(candidates) == 0:
        print("ERROR: zero candidates loaded — check the file path and format.",
              file=sys.stderr)
        sys.exit(1)

    results = precompute(candidates)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    n_honeypots = sum(1 for r in results.values() if r["is_likely_honeypot"])
    n_disqualified = sum(1 for r in results.values() if r["jd_multiplier"] < 0.3)
    print(f"Wrote {len(results)} cached results to {out_path}", file=sys.stderr)
    print(f"  likely honeypots flagged: {n_honeypots}", file=sys.stderr)
    print(f"  heavily JD-disqualified (multiplier < 0.3): {n_disqualified}", file=sys.stderr)


if __name__ == "__main__":
    main()