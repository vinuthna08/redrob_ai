import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fit_scoring import score_all, save_cache, RELEVANT_TITLE_MARKERS 


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
    parser.add_argument("--out", default="data/processed/fit_scores.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading candidates from {args.candidates}...")
    candidates = list(load_candidates(args.candidates))
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"Loaded {len(candidates)} candidates.")

    results = score_all(candidates, show_progress=True)

    print(f"\nWriting cache to {args.out}...")
    save_cache(results, args.out)

    fit_scores = [r.fit_score for r in results.values()]
    fractions = [r.relevant_fraction for r in results.values()]

    print(f"\n=== Fit score distribution ===")
    print(f"  min={min(fit_scores):.4f}  max={max(fit_scores):.4f}  "
          f"mean={sum(fit_scores)/len(fit_scores):.4f}")

    zero_frac = sum(1 for f in fractions if f == 0.0)
    full_frac = sum(1 for f in fractions if f >= 0.999)
    partial_frac = len(fractions) - zero_frac - full_frac
    print(f"\n=== relevant_fraction breakdown ===")
    print(f"  fraction == 0.0  (no relevant career evidence): {zero_frac} ({zero_frac/len(fractions)*100:.1f}%)")
    print(f"  fraction == 1.0  (entire career relevant):       {full_frac} ({full_frac/len(fractions)*100:.1f}%)")
    print(f"  0 < fraction < 1 (partial/mixed career):         {partial_frac} ({partial_frac/len(fractions)*100:.1f}%)")

    
    title_relevant_hits = defaultdict(int)
    for c in candidates:
        for h in c.get("career_history", []) or []:
            title = (h.get("title") or "").lower()
            if any(m in title for m in RELEVANT_TITLE_MARKERS):
                title_relevant_hits[h.get("title", "")] += 1

    print(f"\n=== Titles matching RELEVANT_TITLE_MARKERS (and how often) ===")
    for title, count in sorted(title_relevant_hits.items(), key=lambda x: -x[1]):
        print(f"  {count:>6}  {title}")

    top_candidates = sorted(results.values(), key=lambda r: -r.fit_score)[:10]
    print(f"\n=== Top 10 candidates by blended fit score ===")
    for r in top_candidates:
        print(f"  {r.candidate_id}: fit={r.fit_score:.4f} "
              f"(cosine={r.cosine_score:.4f}, relevant_fraction={r.relevant_fraction:.2f})")


if __name__ == "__main__":
    main()