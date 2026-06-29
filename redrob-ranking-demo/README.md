---
title: RedRob Ranking Demo
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.19.0"
python_version: "3.10"
app_file: app.py
pinned: false
---


# Redrob Hackathon — Candidate Ranking Demo

Upload a JSON file of up to 100 candidate objects from the Redrob hackathon
dataset. The pipeline runs all three ranking stages and returns a spec-compliant
ranked CSV.

## What this demo shows

- **Stage A** — honeypot/consistency detection (tiered hard/soft evidence model)
- **Stage B** — JD-specific disqualifier rules (7 rules from the JD's explicit "do not want" section)
- **Stage C** — blended fit score (sentence-transformers cosine similarity + structured career-relevance fraction)

## How to use

1. Upload a JSON file containing a list of candidate objects (same schema as `candidates.jsonl`)
2. Click Submit
3. Download the ranked CSV

Use `sample_candidates.json` from the hackathon bundle (50 candidates) as a quick test input.

## Full pipeline

The full 100K ranking is reproduced from the GitHub repository at Stage 3.
This Space demonstrates small-sample end-to-end reproducibility as required
by `submission_spec.docx` Section 10.5.
