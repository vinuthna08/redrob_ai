from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from consistency_gate import score_candidate 
from jd_disqualifiers import apply_disqualifiers  
from fit_scoring import score_all as fit_score_all, save_cache as save_fit_cache  


def load_candidates(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run_stage_ab(candidates: list[dict], out_path: str) -> None:
    """Stage A + Stage B: cheap, deterministic dict/string logic. Fast
    even at 100k (measured ~10-18s on this machine).
    """
    print("\n=== Stage A + B: consistency + JD disqualifiers ===", flush=True)
    t0 = time.time()
    results = {}
    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", f"UNKNOWN_{i}")
        consistency = score_candidate(c)
        disqualifier = apply_disqualifiers(c)
        results[cid] = {
            "consistency_score": consistency.consistency_score,
            "consistency_flags": consistency.flags,
            "is_likely_honeypot": consistency.is_likely_honeypot,
            "jd_multiplier": disqualifier.multiplier,
            "jd_triggered_rules": disqualifier.triggered_rules,
        }
        if (i + 1) % 10000 == 0:
            print(f"  processed {i+1}/{len(candidates)} ({time.time()-t0:.1f}s elapsed)", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    elapsed = time.time() - t0
    honeypots = sum(1 for r in results.values() if r["is_likely_honeypot"])
    disqualified = sum(1 for r in results.values() if r["jd_multiplier"] < 0.3)
    print(f"  Done in {elapsed:.1f}s. Wrote {len(results)} results to {out_path}")
    print(f"  likely honeypots flagged: {honeypots}")
    print(f"  heavily JD-disqualified (multiplier < 0.3): {disqualified}")


def run_stage_c(candidates: list[dict], out_path: str) -> None:
    """Stage C: embedding-based fit scoring. SLOW -- measured 73-127
    minutes on the real 100k dataset, CPU-only, on this machine. This is
    the step the spec explicitly allows to exceed the ranking-step
    compute budget, as long as it happens here, not inside rank.py.
    """
    print("\n=== Stage C: embedding + structured fit scoring ===", flush=True)
    print("  WARNING: this step is slow. Expect well over an hour on CPU.", flush=True)
    t0 = time.time()
    results = fit_score_all(candidates, show_progress=True)
    save_fit_cache(results, out_path)
    elapsed = time.time() - t0
    fit_scores = [r.fit_score for r in results.values()]
    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min). Wrote {len(results)} results to {out_path}")
    print(f"  fit_score range: min={min(fit_scores):.4f} max={max(fit_scores):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all precomputation (Stage A+B+C) and cache results to data/processed/."
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--stage-ab-out", default="data/processed/stage_ab_scores.json")
    parser.add_argument("--stage-c-out", default="data/processed/fit_scores.json")
    parser.add_argument("--skip-stage-ab", action="store_true",
                         help="Skip Stage A/B (use if cache already exists and is current)")
    parser.add_argument("--skip-stage-c", action="store_true",
                         help="Skip Stage C (use if cache already exists and is current -- "
                              "saves over an hour on re-runs where only Stage A/B logic changed)")
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...", flush=True)
    candidates = list(load_candidates(args.candidates))
    print(f"Loaded {len(candidates)} candidates.")

    overall_start = time.time()

    if not args.skip_stage_ab:
        run_stage_ab(candidates, args.stage_ab_out)
    else:
        print("\nSkipping Stage A/B (--skip-stage-ab).")

    if not args.skip_stage_c:
        run_stage_c(candidates, args.stage_c_out)
    else:
        print("\nSkipping Stage C (--skip-stage-c).")

    total_elapsed = time.time() - overall_start
    print(f"\n=== Precompute complete in {total_elapsed/60:.1f} min ===")
    print(f"Caches written to {args.stage_ab_out} and {args.stage_c_out}")
    print(f"Now run: python rank.py --candidates {args.candidates} --out submission.csv")


if __name__ == "__main__":
    main()