# Redrob Hackathon — Candidate Ranking System

## Status
- [x] Stage A: consistency gate (`src/consistency_gate.py`) — tested
- [x] Stage B: JD-specific disqualifiers (`src/jd_disqualifiers.py`) — tested
- [ ] Stage C: embedding-based fit scoring (owner: Person B)
- [ ] JSONL loader for candidates.jsonl.gz
- [ ] Score combination + CSV writer (must match validate_submission.py exactly)
- [ ] Reasoning string generator (template off real feature values, no LLM calls)
- [ ] Sandbox deployment (HF Spaces / Colab)

## Design principles (defend these in the Stage 5 interview)
1. Consistency (Stage A) and JD-fit-disqualifiers (Stage B) are kept SEPARATE
   from fit scoring (Stage C), not blended into one score. A well-faked
   honeypot should never be able to mathematically outscore a real candidate
   by looking good on fit alone — it has to survive A and B first.
2. Stage B rules are pulled verbatim from explicit statements in
   job_description.md, not inferred. Cite the JD line for each rule.
3. No LLM calls anywhere in the timed ranking step (5min/16GB/CPU-only/no
   network, per submission_spec.md Section 3). Reasoning strings are
   templated from the same feature values that drove the score — this also
   means they can never contradict the rank (Stage 4 check).
4. Every rule/check has a unit test with a passing fixture and a failing
   fixture. See tests/test_stage_a_b.py for the pattern — keep using it.

## Validator gotchas (from validate_submission.py — read literally, not from
the prose spec)
- Tie-break on equal scores is candidate_id ASCENDING ONLY. Don't rely on a
  secondary model signal for ties unless you also sort by candidate_id after.
- `rank` must serialize as a clean int string ("1" not "1.0" or "01"). If
  writing from pandas, explicitly cast the rank column to int before
  `to_csv()` — pandas will silently float-cast otherwise and every row fails.
- Exactly 100 data rows, ranks 1-100 each exactly once, header must match
  `candidate_id,rank,score,reasoning` exactly.

## Run tests
```
python tests/test_stage_a_b.py
```

## Next steps
1. Person A: write `src/load_candidates.py` (load candidates.jsonl.gz, run
   Stage A + Stage B across all 100k, cache results).
2. Person B: write `src/fit_scoring.py` — local embedding model (CPU-friendly,
   e.g. sentence-transformers all-MiniLM-L6-v2), JD-vs-candidate similarity,
   plus the "concept translation layer" for JD-stated equivalences (e.g.
   "recommendation system" == "embeddings/retrieval experience" — see the
   JD's own Tier-5 example in job_description.md's closing section).
3. Person C: write `src/rank.py` (combine A+B+C scores, write CSV per spec,
   generate reasoning strings), wire up `validate_submission.py` as a CI/local
   check, deploy sandbox.