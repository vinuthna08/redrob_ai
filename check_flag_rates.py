"""
check_flag_rates.py — diagnostic only, not part of the pipeline.

Runs every individual Stage A check across the full candidate set and
prints a per-rule firing rate, so we can see which checks are dead vs
working as intended (rather than just the aggregate honeypot count).

Usage (from repo root):
    python check_flag_rates.py --candidates data/raw/candidates.jsonl
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from consistency_gate import CHECKS  # noqa: E402


def load_candidates(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--limit", type=int, default=None,
                         help="optional cap for a quick sample run")
    args = parser.parse_args()

    counts = {name: 0 for name, _, _, _ in CHECKS}
    examples = {name: [] for name, _, _, _ in CHECKS}
    total = 0

    for i, candidate in enumerate(load_candidates(args.candidates)):
        if args.limit and i >= args.limit:
            break
        total += 1
        for name, check_fn, _ , _ in CHECKS:
            violated, detail = check_fn(candidate)
            if violated:
                counts[name] += 1
                if len(examples[name]) < 3:
                    examples[name].append((candidate.get("candidate_id"), detail))

    print(f"Total candidates scanned: {total}\n")
    print(f"{'Rule':<28} {'Fired':>8} {'Rate':>8}")
    print("-" * 48)
    for name, _,_, _ in CHECKS:
        rate = counts[name] / total * 100 if total else 0
        print(f"{name:<28} {counts[name]:>8} {rate:>7.2f}%")

    print("\nSample fires per rule (up to 3 each):")
    for name, _,_, _ in CHECKS:
        print(f"\n[{name}]")
        if not examples[name]:
            print("  (none fired)")
        for cid, detail in examples[name]:
            print(f"  {cid}: {detail}")


if __name__ == "__main__":
    main()