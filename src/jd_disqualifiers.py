"""
jd_disqualifiers.py — Stage B of the ranking pipeline.

These rules are pulled VERBATIM from explicit statements in job_description.md
("Things we explicitly do NOT want" + "the disqualifiers we actually apply").
This is the highest-leverage module in the whole system: the JD told us almost
exactly what disqualifies a candidate. Most teams will treat the JD as
unstructured text to embed; we treat it as a literal rulebook.

Each rule returns (triggered: bool, weight: float, detail: str). Weight is a
multiplicative dampener on the final fit score (1.0 = no penalty, 0.0 = hard
exclude). Keeping these as explicit, named, multiplicative penalties (not a
black-box score) is what makes this defensible in the Stage 5 interview.
"""

from __future__ import annotations
from dataclasses import dataclass, field

# Companies explicitly called out as "pure services" in the JD.
SERVICES_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}

# Keyword sets used for cheap heuristic classification of career history text.
RESEARCH_ONLY_MARKERS = {"research scientist", "research intern", "phd", "postdoc",
                          "research fellow", "academic"}
PRODUCTION_MARKERS = {"shipped", "deployed", "production", "scaled", "launched",
                       "built and deployed", "serving", "users", "live system"}
LANGCHAIN_WRAPPER_MARKERS = {"langchain", "openai api", "chatgpt wrapper", "gpt wrapper",
                              "prompt engineering"}
PRE_LLM_ML_MARKERS = {"recommendation system", "search ranking", "information retrieval",
                       "ranking model", "click-through", "ctr prediction", "search relevance",
                       "personalization", "collaborative filtering", "learning to rank"}
ARCHITECTURE_TITLE_MARKERS = {"architect", "tech lead", "engineering manager", "principal"}
CV_SPEECH_ROBOTICS_MARKERS = {"computer vision", "speech recognition", "robotics",
                               "image classification", "object detection", "autonomous vehicle"}
NLP_IR_MARKERS = {"nlp", "natural language", "retrieval", "embeddings", "ranking",
                   "search", "ir ", "information retrieval", "rag", "semantic search"}
EXTERNAL_VALIDATION_MARKERS = {"paper", "publication", "talk", "conference", "open source",
                                "open-source", "github.com", "blog post", "arxiv"}


@dataclass
class DisqualifierResult:
    candidate_id: str
    multiplier: float = 1.0
    triggered_rules: list[str] = field(default_factory=list)


def _all_text(candidate: dict) -> str:
    """Concatenate every free-text field we might want to keyword-scan."""
    parts = [
        candidate.get("profile", {}).get("headline", ""),
        candidate.get("profile", {}).get("summary", ""),
    ]
    for h in candidate.get("career_history", []) or []:
        parts.append(h.get("title", ""))
        parts.append(h.get("description", ""))
        parts.append(h.get("company", ""))
    for c in candidate.get("certifications", []) or []:
        parts.append(c.get("name", ""))
    return " ".join(p.lower() for p in parts if p)


def rule_pure_research_no_production(candidate: dict) -> tuple[bool, float, str]:
    """'If you've spent your career in pure research environments... without
    any production deployment — we will not move forward.' (hard exclude)
    """
    text = _all_text(candidate)
    has_research = any(m in text for m in RESEARCH_ONLY_MARKERS)
    has_production = any(m in text for m in PRODUCTION_MARKERS)
    if has_research and not has_production:
        return True, 0.05, "pure-research background with no production deployment evidence"
    return False, 1.0, ""


def rule_recent_langchain_only(candidate: dict) -> tuple[bool, float, str]:
    """'If your AI experience consists primarily of recent (<12mo) projects
    using LangChain to call OpenAI — we will probably not move forward,
    unless substantial pre-LLM-era ML production experience exists.'
    """
    history = candidate.get("career_history", []) or []
    if not history:
        return False, 1.0, ""

    text = _all_text(candidate)
    has_wrapper_signal = any(m in text for m in LANGCHAIN_WRAPPER_MARKERS)
    has_pre_llm_depth = any(m in text for m in PRE_LLM_ML_MARKERS)

    # Total experience as a rough proxy for "recent <12mo" — if total relevant
    # AI-tagged experience is shallow and wrapper-shaped, and there's no
    # pre-LLM ML depth elsewhere in the history, down-weight.
    total_months = sum(h.get("duration_months", 0) or 0 for h in history)
    if has_wrapper_signal and not has_pre_llm_depth and total_months < 24:
        return True, 0.3, "LangChain/API-wrapper-shaped AI experience with no pre-LLM ML depth"
    return False, 1.0, ""


def rule_senior_no_recent_code(candidate: dict) -> tuple[bool, float, str]:
    """'If you are a senior engineer who hasn't written production code in
    the last 18 months because you've moved into architecture/tech-lead
    roles — we will probably not move forward. This role writes code.'
    """
    history = candidate.get("career_history", []) or []
    current = next((h for h in history if h.get("is_current")), None)
    if not current:
        return False, 1.0, ""

    title = (current.get("title") or "").lower()
    duration = current.get("duration_months", 0) or 0
    if any(m in title for m in ARCHITECTURE_TITLE_MARKERS) and duration >= 18:
        return True, 0.4, (
            f"current role '{current.get('title')}' is architecture/lead-shaped "
            f"and has run {duration}mo — role explicitly writes code"
        )
    return False, 1.0, ""


def rule_pure_services_career(candidate: dict) -> tuple[bool, float, str]:
    """'People who have only worked at consulting firms... in their entire
    career' — hard exclude, UNLESS currently at one but with prior
    product-company experience (explicit exception in the JD).
    """
    history = candidate.get("career_history", []) or []
    if not history:
        return False, 1.0, ""

    companies = [(h.get("company") or "").lower() for h in history]
    all_services = all(any(sf in c for sf in SERVICES_FIRMS) for c in companies if c)

    if all_services and companies:
        return True, 0.05, (
            f"entire career_history at services firms only: {set(companies)}"
        )
    return False, 1.0, ""


def rule_cv_speech_robotics_no_nlp(candidate: dict) -> tuple[bool, float, str]:
    """'People whose primary expertise is computer vision, speech, or
    robotics without significant NLP/IR exposure.'
    """
    text = _all_text(candidate)
    has_cv_speech_robo = any(m in text for m in CV_SPEECH_ROBOTICS_MARKERS)
    has_nlp_ir = any(m in text for m in NLP_IR_MARKERS)
    if has_cv_speech_robo and not has_nlp_ir:
        return True, 0.3, "CV/speech/robotics background with no NLP/IR exposure evident"
    return False, 1.0, ""


def rule_closed_source_no_validation(candidate: dict) -> tuple[bool, float, str]:
    """'People whose work has been entirely on closed-source proprietary
    systems for 5+ years without external validation (papers, talks,
    open-source).'

    IMPORTANT: this rule requires a POSITIVE closed-source/proprietary signal
    (explicit github_activity_score == -1, i.e. no GitHub linked at all, AND
    an explicit 'proprietary'/'closed-source'/'internal only' marker in the
    text) — NOT merely the absence of validation keywords. Most candidates in
    a structured-fields dataset simply won't mention papers/talks/OSS even if
    they have them; punishing absence-of-evidence as if it were
    evidence-of-absence would wrongly tank strong, normal candidates. This
    keeps the rule narrow and matched to what the JD actually describes:
    a specific, identifiable pattern, not a default penalty.
    """
    total_months = sum(h.get("duration_months", 0) or 0
                        for h in candidate.get("career_history", []) or [])
    text = _all_text(candidate)
    certs = candidate.get("certifications", []) or []
    signals = candidate.get("redrob_signals", {})

    has_external_validation = any(m in text for m in EXTERNAL_VALIDATION_MARKERS) or len(certs) > 0
    no_github = signals.get("github_activity_score", 0) == -1
    explicit_proprietary_marker = any(
        m in text for m in ("proprietary", "closed-source", "closed source", "internal tooling only")
    )

    if (total_months >= 60 and not has_external_validation
            and no_github and explicit_proprietary_marker):
        return True, 0.6, (
            f"{total_months/12:.1f}y experience, no GitHub linked, explicit "
            f"proprietary/closed-source language, and no external validation signal"
        )
    return False, 1.0, ""


def rule_title_chasing_pattern(candidate: dict) -> tuple[bool, float, str]:
    """'If your career trajectory shows you optimizing for "Senior" -> "Staff"
    -> "Principal" titles by switching companies every 1.5 years.'
    """
    history = candidate.get("career_history", []) or []
    if len(history) < 3:
        return False, 1.0, ""

    short_stints = sum(1 for h in history if (h.get("duration_months", 0) or 0) <= 18)
    seniority_words = {"senior", "staff", "principal", "lead"}
    escalating_titles = sum(
        1 for h in history if any(w in (h.get("title") or "").lower() for w in seniority_words)
    )

    if short_stints >= len(history) * 0.6 and escalating_titles >= 2:
        return True, 0.5, (
            f"{short_stints}/{len(history)} roles <=18mo with escalating seniority "
            f"titles — job-hop-for-title pattern"
        )
    return False, 1.0, ""


# All rules, in order. Multipliers compound (multiply together), so multiple
# triggered rules stack — a candidate can be down-weighted by several
# independent JD-stated reasons simultaneously.
RULES = [
    ("pure_research_no_production", rule_pure_research_no_production),
    ("recent_langchain_only", rule_recent_langchain_only),
    ("senior_no_recent_code", rule_senior_no_recent_code),
    ("pure_services_career", rule_pure_services_career),
    ("cv_speech_robotics_no_nlp", rule_cv_speech_robotics_no_nlp),
    ("closed_source_no_validation", rule_closed_source_no_validation),
    ("title_chasing_pattern", rule_title_chasing_pattern),
]


def apply_disqualifiers(candidate: dict) -> DisqualifierResult:
    cid = candidate.get("candidate_id", "UNKNOWN")
    multiplier = 1.0
    triggered = []

    for name, rule_fn in RULES:
        fired, weight, detail = rule_fn(candidate)
        if fired:
            multiplier *= weight
            triggered.append(f"[{name}] {detail}")

    return DisqualifierResult(candidate_id=cid, multiplier=multiplier, triggered_rules=triggered)


def apply_all(candidates: list[dict]) -> dict[str, DisqualifierResult]:
    return {c.get("candidate_id", f"UNKNOWN_{i}"): apply_disqualifiers(c)
            for i, c in enumerate(candidates)}