


# redrob_hackathon — Intelligent Candidate Discovery & Ranking

Ranks the top 100 candidates from a 100K-candidate pool against the
"Senior AI Engineer — Founding Team" job description, for the Redrob
INDIA RUNS hackathon (Intelligent Candidate Discovery & Ranking Challenge).

## Architecture

Three independent, explainable stages, each with its own design rationale
documented in its module docstring:

- **Stage A — `src/consistency_gate.py`**: detects honeypot/implausible
  profiles (e.g. an 11-year experience gap, "expert" skills with zero
  months used). Six checks, tiered hard/soft: a single "hard" flag
  (deterministic, no-benign-explanation evidence) is sufficient for
  honeypot classification on its own; "soft" flags (circumstantial,
  individually forgivable) require accumulation past a score threshold.

- **Stage B — `src/jd_disqualifiers.py`**: seven rules pulled directly
  from explicit statements in the JD's "things we explicitly do NOT
  want" / "disqualifiers we actually apply" sections. Multiplicative
  penalties, each independently explainable.

- **Stage C — `src/fit_scoring.py`**: blends sentence-transformers
  embedding similarity (candidate career narrative vs JD substance) with
  a structured `relevant_career_fraction` signal (share of career months
  in genuinely ML/AI/retrieval-titled roles, from trusted title/industry
  fields). Both signals are needed -- embedding similarity alone was
  tested and found to score generic "I'm curious about ChatGPT"
  candidates uncomfortably close to genuine ML engineers; the structured
  signal fixes this. Full reasoning in the module docstring.

A key design decision threading all three stages: **`career_history[].description`
is never used for scoring or embedding anywhere in this codebase.**
`check_description_shuffle.py` found that ~50%+ of candidates' description
text doesn't match their own title/company -- it's drawn from a shared
pool independent of the actual role. There's no field marking which
descriptions are "real," so any use of this field would inject coin-flip
noise into scoring. `profile.summary`, `profile.headline`, and
career_history `title`/`company`/`industry`/`duration_months` are NOT
shuffled and are the only text fields used.

## Compute constraint (read this before running)

Per `submission_spec.docx` Section 3, the ranking step has a hard budget:
**<=5 minutes wall-clock, CPU-only, no GPU, no network calls.** This is
why the pipeline is split into two stages:

1. **`precompute.py`** -- slow, run once, NOT subject to the 5-minute
   budget. Runs Stage A+B (~10-20s) and Stage C embeddings (measured
   **73-127 minutes** on a CPU-only machine for the real 100K dataset).
   Writes results to `data/processed/`.
2. **`rank.py`** -- fast, this IS what gets reproduced at Stage 3. Loads
   the precomputed caches, combines scores, sorts, writes the submission
   CSV. Measured **~5 seconds** end-to-end, comfortably under budget.

This split is explicitly permitted by the spec: *"If your system requires
pre-computation..., pre-computation may exceed the 5-minute window, but
the ranking step that produces the CSV must complete within it."*

## How to reproduce the submission

```bash
pip install -r requirements.txt

# Step 1: precompute (slow -- budget 1.5-2+ hours on CPU)
python precompute.py --candidates data/raw/candidates.jsonl

# Step 2: rank (fast -- completes in seconds)
python rank.py --candidates data/raw/candidates.jsonl --out submission.csv

# Step 3: validate
python validate_submission.py submission.csv
```

If you already have current caches in `data/processed/` (from a prior
run) and only changed `rank.py` itself, skip the slow step:

```bash
python precompute.py --candidates data/raw/candidates.jsonl --skip-stage-ab --skip-stage-c
python rank.py --candidates data/raw/candidates.jsonl --out submission.csv
```

##Repo layout

```
redrob_hackathon/
├── README.md
├── requirements.txt
├── submission_metadata.yaml
├── precompute.py              # Stage A+B+C precomputation (slow, run once)
├── rank.py                    # Final ranking + CSV writer (fast, Stage-3-reproduced)
├── validate_submission.py     # Organizer-provided validator
│
├── data/
│   ├── raw/                   # Hackathon bundle, untouched (gitignored: candidates.jsonl)
│   ├── processed/             # Precompute cache (gitignored, regenerable)
│   └── manual_labels/         # Hand-labeled ground truth for calibration
│
├── src/
│   ├── consistency_gate.py    # Stage A
│   ├── jd_disqualifiers.py    # Stage B
│   ├── fit_scoring.py         # Stage C
│   └── load_candidates.py     # Standalone Stage A/B runner (used by precompute.py logic)
│
├── tests/
│   ├── test_stage_a_b.py
│   └── test_fit_scoring_sanity.py
│
└── check_*.py, score_fit.py   # Diagnostic scripts used during development
                                 # (not part of the reproduction path, kept
                                 # for methodology transparency -- see git
                                 # history for the findings each one produced)
```

##Methodology notes / known limitations

Documented honestly rather than hidden, since Stage 4/5 review weighs
this:

- **Stage A:** `title_history_mismatch` and `closed_source_no_validation`
  fire at or near 0% on the real dataset -- confirmed via
  `check_dead_rules_vocab.py` to be correctly dormant (their trigger
  vocabulary doesn't exist in this dataset), not broken logic.
- **Stage B:** `senior_no_recent_code` fires 0% -- confirmed the dataset's
  44 distinct titles never include "architect"/"tech lead"/"principal."
  `pure_research_no_production` is rare by design (research-marker base
  rate is 0.14% of candidates).
- **Stage C:** `RELEVANT_TITLE_MARKERS` excludes "Data Engineer" and
  "Data Scientist" after checking real data -- both were initially
  included as plausible-sounding guesses, but a full 100K run found
  "Data Engineer" alone accounted for 73% of all relevant-fraction hits
  despite the JD never naming data engineering as a relevant function.
  Removed after verification; see git history for the before/after
  numbers.
- Weights (`CONSISTENCY_WEIGHT`/`FIT_WEIGHT` in `rank.py`,
  `COSINE_WEIGHT`/`STRUCTURED_WEIGHT` in `fit_scoring.py`) are reasoned
  choices, not empirically tuned against ground truth -- we don't have
  labeled relevance tiers to tune against. `data/manual_labels/` is
  intended for this once populated.

## AI tool usage


AI tools were used as development assistants for debugging, troubleshooting, code review, and documentation support. All core design decisions, ranking methodology, scoring logic, validation procedures, and final implementation choices were developed and verified by the team.

