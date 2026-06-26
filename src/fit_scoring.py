"""
fit_scoring.py — Stage C of the ranking pipeline.

COMBINES TWO INDEPENDENT SIGNALS, not one:

  1. Embedding cosine similarity (sentence-transformers, all-MiniLM-L6-v2)
     between candidate career-narrative text and the JD's substance text.
     Good at coarse TOPIC matching -- correctly separates a genuine ML
     engineer (Ela Singh-style) from a candidate with zero AI exposure by
     a wide margin (validated: 0.75 vs 0.29 on real sample candidates).

  2. relevant_career_fraction: a STRUCTURED, data-derived signal -- the
     fraction of a candidate's total career months spent in roles/
     industries that plausibly involve applied ML/AI/retrieval work,
     computed directly from career_history title/industry fields (NOT
     embeddings, NOT prose).

WHY BOTH ARE NEEDED (this was tested, not assumed):
  An earlier version relied on cosine similarity alone. Sanity testing
  against real candidates (test_fit_scoring_sanity.py) found it correctly
  ranked a genuine ML engineer highest, but ALSO gave meaningfully high
  scores (~0.57-0.61) to generic non-technical candidates (Operations
  Manager, Marketing Manager) whose summaries casually mention "AI tools"
  or "ChatGPT" without any real technical ownership -- almost as high as
  candidates with real ML backgrounds, and far higher than the gap should
  be per the JD's own explicit framing of who is NOT a fit.

  We then rewrote the JD text to explicitly state the negative case
  ("not a fit for someone who is merely curious about AI tools...") and
  re-ran the same test. The output was IDENTICAL to four decimal places.
  This is a known, real limitation of single-vector sentence embeddings:
  they are dominated by which topical/content words are present, not by
  negation or fine-grained qualitative distinctions ("built it" vs "is
  aware of it" reads as topically similar even when explicitly contrasted
  in the text). Verified by direct experiment, not assumed from theory.

  Since rephrasing the JD text provably did not fix this, the fix is
  structural: add a second signal that does not rely on prose semantics
  at all. relevant_career_fraction directly counts months spent in
  ML/AI/retrieval-relevant titles and industries vs unrelated ones, using
  trustworthy structured fields (title, industry -- NOT the shuffled
  description field, see DESIGN NOTE below). A candidate now needs BOTH
  topical proximity AND genuine structural career evidence to score
  highly -- this directly suppresses the "mentions ChatGPT casually"
  false-positive pattern without requiring the embedding model to detect
  subtle negation, which we've now shown it cannot reliably do.

DESIGN NOTE — why career_history[].description is excluded:
  check_description_shuffle.py (diagnostic, repo root) found that ~50%+ of
  candidates' career_history description text does NOT match their own
  title/company -- it appears to be drawn from a shared pool independent
  of the actual role (e.g. a "Marketing Manager" entry with a description
  about Kafka pipelines). There is no field that flags which descriptions
  are "real" vs shuffled, so using description text anywhere in Stage C
  would inject coin-flip noise into the single most consequential score
  in the pipeline. profile.summary, profile.headline, and career_history
  title/company/industry/duration_months are NOT shuffled (verified by
  manual inspection across dozens of real candidates) and are the only
  fields used here.

DESIGN NOTE — why this is NOT just skill-keyword matching:
  job_description.docx is explicit: "The right answer is not 'find
  candidates whose skills section contains the most AI keywords.' That's
  a trap we've explicitly built into the dataset... A candidate who has
  all the AI keywords listed as skills but whose title is 'Marketing
  Manager' is not a fit." Accordingly, neither signal in this module uses
  skills[] at all. Skills are Stage B's explicit-rule territory
  (jd_disqualifiers.py). Stage C measures whether the candidate's actual
  career SHAPE (narrative + structured role history) matches the role's
  substance -- title and industry, not self-reported skill lists.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"

# Weight given to the embedding cosine signal vs the structured
# relevant-career-fraction signal in the final blended score. Weighted
# toward the structured signal (0.4 cosine / 0.6 structured) because the
# structured signal is what actually discriminates the false-positive
# pattern we found; cosine alone was proven insufficient on its own.
COSINE_WEIGHT = 0.4
STRUCTURED_WEIGHT = 0.6

# The JD's substance -- what the role actually is -- deliberately
# excluding the disqualifier sections (those are Stage B's job already).
JD_FIT_TEXT = """
Senior AI Engineer, founding team at an AI-native talent intelligence
platform. Owns the intelligence layer of the product: ranking, retrieval,
and matching systems. Will audit existing BM25 and rule-based scoring,
ship a v2 ranking system using embeddings, hybrid retrieval, and
LLM-based re-ranking, and build offline/online evaluation infrastructure.
Drives long-term architecture for candidate-JD matching at scale and
mentors a growing engineering team.

The ideal candidate has roughly 6-8 years of total experience, with 4-5
years in applied ML or AI roles at product companies, not pure IT
services firms. They have shipped at least one end-to-end ranking,
search, or recommendation system to real users at meaningful scale. They
have hands-on production experience with embeddings-based retrieval,
vector databases or hybrid search infrastructure, and designing
evaluation frameworks for ranking systems. They have strong, defensible
opinions about retrieval, evaluation methodology, and when to fine-tune
versus prompt an LLM, grounded in systems they actually built. Production
experience with recommendation systems, search ranking, information
retrieval, or personalization at a product company is a strong signal
even without recent LLM-specific keywords. A scrappy, ship-first
product-engineering attitude is more important than research depth.
""".strip()

# Structured taxonomy for relevant_career_fraction. Title/industry
# keywords that plausibly indicate applied ML/AI/retrieval/search work.
# Scoped to TITLE and INDUSTRY fields only (trustworthy, not shuffled).
#
# NOTE: "Data Engineer" / "Senior Data Engineer" were REMOVED after
# checking real data. They were included as a guess ("adjacent infra
# role"), but a full 100k run found they accounted for 1,637 + 1,543 =
# 3,180 of the 4,338 total relevant_fraction title-matches -- 73% of the
# entire signal -- despite the JD never naming data engineering as a
# relevant function. The JD explicitly lists what counts even without
# LLM-specific keywords: "recommendation systems, search ranking,
# information retrieval, or personalization." Data engineering (ETL,
# warehouse, pipeline orchestration) is infrastructure-ADJACENT to ML but
# is a different job per the JD's own specific, named list. Keeping it
# would have meant the single largest driver of Stage C's structured
# signal was an unverified guess, not something grounded in the JD text.
RELEVANT_TITLE_MARKERS = (
    "recommendation systems engineer", "search engineer", "nlp engineer",
    "applied ml engineer", "machine learning engineer", "ml engineer",
    "ai engineer", "data scientist", "research engineer",
)
RELEVANT_INDUSTRY_MARKERS = (
    "ai/ml", "ai", "machine learning",
)
# Company size/context that signals "product company" per the JD's own
# explicit framing (product company vs pure IT services firm).
SERVICES_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}


@dataclass
class FitScoreResult:
    candidate_id: str
    fit_score: float          # final blended 0-1 score
    cosine_score: float       # raw embedding similarity (diagnostic/debug)
    relevant_fraction: float  # raw structured signal (diagnostic/debug)


def _career_history_sentence(entry: dict) -> str:
    """Synthesize one clean sentence per role from trusted structured
    fields only. Deliberately omits entry.get('description').
    """
    title = entry.get("title", "")
    company = entry.get("company", "")
    industry = entry.get("industry", "")
    size = entry.get("company_size", "")
    months = entry.get("duration_months", 0) or 0

    if not title or not company:
        return ""

    parts = [f"{title} at {company}"]
    if industry:
        parts.append(f"({industry}")
        if size:
            parts[-1] += f", {size} employees)"
        else:
            parts[-1] += ")"
    if months:
        years = months / 12.0
        parts.append(f"for {years:.1f} years" if years >= 1 else f"for {months} months")

    return " ".join(parts) + "."


def build_candidate_text(candidate: dict) -> str:
    """Build the embedding input text from trusted, non-shuffled fields
    only: profile.summary, profile.headline, and synthesized
    career_history sentences (NOT description).
    """
    profile = candidate.get("profile", {})
    parts = []

    headline = profile.get("headline", "")
    if headline:
        parts.append(headline)

    summary = profile.get("summary", "")
    if summary:
        parts.append(summary)

    for entry in candidate.get("career_history", []) or []:
        sentence = _career_history_sentence(entry)
        if sentence:
            parts.append(sentence)

    return " ".join(parts)


def compute_relevant_career_fraction(candidate: dict) -> float:
    """Fraction of total career months spent in roles whose TITLE or
    INDUSTRY plausibly indicates applied ML/AI/retrieval/search work,
    using only structured fields (not description, not skills).

    A role counts as relevant if its title OR industry matches the
    taxonomy above. Returns 0.0-1.0. A candidate with no career_history
    or zero total months returns 0.0 (no evidence, no credit).
    """
    history = candidate.get("career_history", []) or []
    total_months = 0
    relevant_months = 0

    for entry in history:
        months = entry.get("duration_months", 0) or 0
        if months <= 0:
            continue
        total_months += months

        title = (entry.get("title") or "").lower()
        industry = (entry.get("industry") or "").lower()

        is_relevant = (
            any(m in title for m in RELEVANT_TITLE_MARKERS)
            or any(m in industry for m in RELEVANT_INDUSTRY_MARKERS)
        )
        if is_relevant:
            relevant_months += months

    if total_months == 0:
        return 0.0
    return relevant_months / total_months


def score_all(
    candidates: list[dict],
    model: SentenceTransformer | None = None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> dict[str, FitScoreResult]:
    """Embed the JD once, batch-embed all candidates, blend cosine
    similarity with the structured relevant-career-fraction signal.
    """
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    jd_embedding = model.encode(JD_FIT_TEXT, convert_to_tensor=True)

    candidate_ids = [c.get("candidate_id", f"UNKNOWN_{i}") for i, c in enumerate(candidates)]
    candidate_texts = [build_candidate_text(c) for c in candidates]
    relevant_fractions = [compute_relevant_career_fraction(c) for c in candidates]

    t0 = time.time()
    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_tensor=True,
    )
    elapsed = time.time() - t0
    if show_progress:
        print(f"Encoded {len(candidates)} candidates in {elapsed:.1f}s", flush=True)

    cosine_scores = util.cos_sim(candidate_embeddings, jd_embedding).squeeze(-1).tolist()

    results = {}
    for cid, cos, frac in zip(candidate_ids, cosine_scores, relevant_fractions):
        cos = float(cos)
        blended = (COSINE_WEIGHT * cos) + (STRUCTURED_WEIGHT * frac)
        results[cid] = FitScoreResult(
            candidate_id=cid,
            fit_score=blended,
            cosine_score=cos,
            relevant_fraction=frac,
        )
    return results


def save_cache(results: dict[str, FitScoreResult], out_path: str) -> None:
    serializable = {
        cid: {"fit_score": r.fit_score, "cosine_score": r.cosine_score,
              "relevant_fraction": r.relevant_fraction}
        for cid, r in results.items()
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_cache(path: str) -> dict[str, FitScoreResult]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        cid: FitScoreResult(
            candidate_id=cid,
            fit_score=v["fit_score"],
            cosine_score=v["cosine_score"],
            relevant_fraction=v["relevant_fraction"],
        )
        for cid, v in raw.items()
    }