"""
app.py — Redrob Hackathon sandbox demo.

Accepts up to 100 candidates as a JSON file upload, runs the full
Stage A+B+C ranking pipeline end-to-end, and returns a ranked CSV.

This is the small-sample sandbox required by submission_spec.docx
Section 10.5. The full 100K precomputation (which takes 1.5-2+ hours
on CPU) is NOT reproduced here -- Stage 3 reproduction uses the full
codebase from the GitHub repo. This Space demonstrates that the ranking
logic runs correctly end-to-end on a small sample within the 5-minute
compute budget.
"""

import csv
import io
import json
import time
from dataclasses import dataclass, field
from datetime import date, datetime

import gradio as gr
from sentence_transformers import SentenceTransformer, util

# ── Model (loaded once at startup) ──────────────────────────────────────────
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Stage C constants ────────────────────────────────────────────────────────
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
even without recent LLM-specific keywords.
""".strip()

RELEVANT_TITLE_MARKERS = (
    "recommendation systems engineer", "search engineer", "nlp engineer",
    "applied ml engineer", "machine learning engineer", "ml engineer",
    "ai engineer", "research engineer",
)
RELEVANT_INDUSTRY_MARKERS = ("ai/ml", "ai", "machine learning")
SERVICES_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini",
}

CONSISTENCY_WEIGHT = 0.3
FIT_WEIGHT = 0.7

# ── Stage A: consistency gate ─────────────────────────────────────────────────

def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def check_experience_arithmetic(c):
    claimed = c.get("profile", {}).get("years_of_experience", 0) or 0
    history = c.get("career_history", []) or []
    actual_months = sum(h.get("duration_months", 0) or 0 for h in history)
    actual_years = actual_months / 12.0
    gap = claimed - actual_years
    if claimed > 1 and gap > max(2.0, claimed * 0.4):
        return True, f"claimed {claimed:.1f}y experience but career_history sums to only {actual_years:.1f}y"
    return False, ""


def check_skill_vs_assessment(c):
    skills = c.get("skills", []) or []
    assessments = (c.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}
    contradictions = []
    for s in skills:
        if s.get("proficiency") == "expert":
            score = assessments.get(s.get("name", ""))
            if score is not None and score < 50:
                contradictions.append(f"{s.get('name')} (expert, scored {score:.0f}/100)")
    zero_dur = [s.get("name") for s in skills
                if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0]
    if zero_dur:
        contradictions.append(f"{len(zero_dur)} expert skill(s) with 0 months used")
    if contradictions:
        return True, "skill/assessment contradiction: " + "; ".join(contradictions)
    return False, ""


def check_endorsement_inflation(c):
    signals = c.get("redrob_signals", {}) or {}
    endorsements = signals.get("endorsements_received", 0) or 0
    connections = signals.get("connection_count", 0) or 0
    if connections > 0 and endorsements / connections > 3.0:
        return True, f"endorsement ratio {endorsements/connections:.2f} (>{3.0})"
    return False, ""


def check_activity_vs_availability(c):
    signals = c.get("redrob_signals", {}) or {}
    open_to_work = signals.get("open_to_work_flag", False)
    last_active = _parse_date(signals.get("last_active_date"))
    response_rate = signals.get("recruiter_response_rate", 0) or 0
    flags = []
    if last_active:
        days_stale = (date.today() - last_active).days
        if open_to_work and days_stale > 180:
            flags.append(f"open_to_work=True but inactive {days_stale}d")
    if response_rate < 0.05:
        flags.append(f"response_rate={response_rate:.2f}")
    if flags:
        return True, "; ".join(flags)
    return False, ""


def check_verification_baseline(c):
    signals = c.get("redrob_signals", {}) or {}
    verified = [signals.get("verified_email"), signals.get("verified_phone"),
                signals.get("linkedin_connected")]
    if not any(verified):
        return True, "no verification signals"
    return False, ""


def check_title_mismatch(c):
    profile = c.get("profile", {})
    history = c.get("career_history", []) or []
    current = next((h for h in history if h.get("is_current")), None)
    if not current:
        return False, ""
    if current.get("title") != profile.get("current_title") or \
       current.get("company") != profile.get("current_company"):
        return True, f"profile title/company doesn't match current career_history entry"
    return False, ""


CHECKS = [
    ("experience_arithmetic", check_experience_arithmetic, 30, "hard"),
    ("skill_vs_assessment", check_skill_vs_assessment, 30, "hard"),
    ("endorsement_inflation", check_endorsement_inflation, 15, "soft"),
    ("activity_vs_availability", check_activity_vs_availability, 10, "soft"),
    ("verification_baseline", check_verification_baseline, 5, "soft"),
    ("title_history_mismatch", check_title_mismatch, 10, "soft"),
]


def score_consistency(c):
    score = 100.0
    flags = []
    hard_flag = False
    for name, fn, penalty, tier in CHECKS:
        violated, detail = fn(c)
        if violated:
            score -= penalty
            flags.append(f"[{name}] {detail}")
            if tier == "hard":
                hard_flag = True
    score = max(0.0, score)
    return {"consistency_score": score, "consistency_flags": flags,
            "is_likely_honeypot": hard_flag or score < 40}


# ── Stage B: JD disqualifiers ────────────────────────────────────────────────

def apply_disqualifiers(c):
    history = c.get("career_history", []) or []
    multiplier = 1.0
    triggered = []

    # pure services career
    companies = [(h.get("company") or "").lower() for h in history]
    if companies and all(any(sf in co for sf in SERVICES_FIRMS) for co in companies if co):
        multiplier *= 0.05
        triggered.append("[pure_services_career] entire career at services firms only")

    # title chasing
    if len(history) >= 3:
        short = sum(1 for h in history if (h.get("duration_months") or 0) <= 18)
        seniority = {"senior", "staff", "principal", "lead"}
        escalating = sum(1 for h in history
                         if any(w in (h.get("title") or "").lower() for w in seniority))
        if short >= len(history) * 0.6 and escalating >= 2:
            multiplier *= 0.5
            triggered.append("[title_chasing_pattern]")

    return {"multiplier": multiplier, "jd_triggered_rules": triggered}


# ── Stage C: fit scoring ──────────────────────────────────────────────────────

def build_candidate_text(c):
    profile = c.get("profile", {})
    parts = []
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])
    for h in c.get("career_history", []) or []:
        title = h.get("title", "")
        company = h.get("company", "")
        industry = h.get("industry", "")
        months = h.get("duration_months", 0) or 0
        if title and company:
            sentence = f"{title} at {company}"
            if industry:
                sentence += f" ({industry})"
            if months:
                sentence += f" for {months/12:.1f} years" if months >= 12 else f" for {months} months"
            parts.append(sentence + ".")
    return " ".join(parts)


def compute_relevant_fraction(c):
    history = c.get("career_history", []) or []
    total = relevant = 0
    for h in history:
        months = h.get("duration_months", 0) or 0
        if months <= 0:
            continue
        total += months
        title = (h.get("title") or "").lower()
        industry = (h.get("industry") or "").lower()
        if any(m in title for m in RELEVANT_TITLE_MARKERS) or \
           any(m in industry for m in RELEVANT_INDUSTRY_MARKERS):
            relevant += months
    return relevant / total if total > 0 else 0.0


def rank_candidates(candidates, model):
    jd_emb = model.encode(JD_FIT_TEXT, convert_to_tensor=True)
    texts = [build_candidate_text(c) for c in candidates]
    embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = util.cos_sim(embs, jd_emb).squeeze(-1).tolist()

    scored = []
    for c, cos in zip(candidates, cosine_scores):
        cid = c.get("candidate_id", "UNKNOWN")
        consistency = score_consistency(c)
        disqualifier = apply_disqualifiers(c)

        if consistency["is_likely_honeypot"]:
            continue

        frac = compute_relevant_fraction(c)
        fit_score = (COSINE_WEIGHT * float(cos)) + (STRUCTURED_WEIGHT * frac)
        final = ((CONSISTENCY_WEIGHT * consistency["consistency_score"] / 100.0) +
                 (FIT_WEIGHT * fit_score)) * disqualifier["multiplier"]

        profile = c.get("profile", {})
        signals = c.get("redrob_signals", {}) or {}
        response_rate = signals.get("recruiter_response_rate", 0) or 0
        reasoning_parts = [
            f"{profile.get('current_title','?')} at {profile.get('current_company','?')} "
            f"with {profile.get('years_of_experience',0):.1f}y experience."
        ]
        if frac >= 0.99:
            reasoning_parts.append("Entire career in ML/AI/retrieval-relevant roles.")
        elif frac > 0:
            reasoning_parts.append(f"{frac*100:.0f}% of career in ML/AI/retrieval-relevant roles.")
        else:
            reasoning_parts.append("No ML/AI/retrieval-relevant career history.")
        if disqualifier["jd_triggered_rules"]:
            reasoning_parts.append(f"Concern: {disqualifier['jd_triggered_rules'][0].split(']')[0].lstrip('[').replace('_',' ')}.")
        if response_rate >= 0.5:
            reasoning_parts.append(f"Strong recruiter response rate ({response_rate:.2f}).")
        elif response_rate < 0.1:
            reasoning_parts.append(f"Low recruiter response rate ({response_rate:.2f}).")

        scored.append((cid, round(final, 4), " ".join(reasoning_parts)))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:100]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def run_ranking(json_file):
    if json_file is None:
        return "Please upload a JSON file.", None

    t0 = time.time()
    try:
        file_path = json_file if isinstance(json_file, str) else json_file.name

        with open(file_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except Exception as e:
        return f"Error reading JSON: {e}", None

    if not isinstance(candidates, list):
        return "JSON must be a list of candidate objects.", None
    if len(candidates) > 100:
        return f"This demo accepts up to 100 candidates. Got {len(candidates)}.", None

    try:
        ranked = rank_candidates(candidates, MODEL)
    except Exception as e:
        return f"Scoring error: {e}", None

    # Write CSV to an in-memory string, then to a temp file Gradio can serve
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for rank, (cid, score, reasoning) in enumerate(ranked, start=1):
        writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    tmp_path = "/tmp/submission_sample.csv"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(output.getvalue())

    elapsed = time.time() - t0
    status = (
        f"Ranked {len(candidates)} candidates in {elapsed:.1f}s.\n"
        f"Excluded likely honeypots: {len(candidates) - len(ranked)}\n"
        f"Top candidate: {ranked[0][0]} (score {ranked[0][1]:.4f})\n"
        f"Output: {len(ranked)} ranked candidates."
    )
    return status, tmp_path


demo = gr.Interface(
    fn=run_ranking,
    inputs="file",
    outputs=["text", "file"],
)

if __name__ == "__main__":
    demo.launch()
