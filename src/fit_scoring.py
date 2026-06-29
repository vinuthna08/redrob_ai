from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"

COSINE_WEIGHT = 0.4
STRUCTURED_WEIGHT = 0.6

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

RELEVANT_TITLE_MARKERS = (
    "recommendation systems engineer", "search engineer", "nlp engineer",
    "applied ml engineer", "machine learning engineer", "ml engineer",
    "ai engineer", "data scientist", "research engineer",
)
RELEVANT_INDUSTRY_MARKERS = (
    "ai/ml", "ai", "machine learning",
)
SERVICES_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}


@dataclass
class FitScoreResult:
    candidate_id: str
    fit_score: float          
    cosine_score: float       
    relevant_fraction: float  


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