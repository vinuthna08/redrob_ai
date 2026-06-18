"""
consistency_gate.py — Stage A of the ranking pipeline.

Detects candidates whose profile fields are individually schema-valid but
*jointly implausible* (honeypots, keyword-stuffers, inflated profiles).

Design principle (defend this in the interview):
  We treat "is this profile real/plausible" as ORTHOGONAL to "is this a good
  fit for the JD." A honeypot with perfect keyword overlap should not be
  rankable purely on fit-similarity — it needs to be caught before fit
  scoring ever runs, otherwise a well-faked profile can mathematically
  outscore a real, slightly-weaker candidate.

Each check below returns a (violated: bool, detail: str) pair so that every
flag can be surfaced verbatim in the reasoning column later — no hidden
logic, nothing the team can't explain line-by-line in a defend-your-work
interview.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class ConsistencyResult:
    candidate_id: str
    consistency_score: float  # 0-100, higher = more plausible
    flags: list[str] = field(default_factory=list)
    is_likely_honeypot: bool = False


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def check_experience_arithmetic(candidate: dict) -> tuple[bool, str]:
    """Claimed years_of_experience vs sum of career_history durations.

    A large gap (claimed >> actual) is a hard, cheap, deterministic
    honeypot signal — e.g. 'CAND claims 8 years, history sums to <3'.
    """
    claimed = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    history = candidate.get("career_history", []) or []
    actual_months = sum(h.get("duration_months", 0) or 0 for h in history)
    actual_years = actual_months / 12.0

    # Allow slack for overlap-free estimation error / concurrent roles.
    gap = claimed - actual_years
    if claimed > 1 and gap > max(2.0, claimed * 0.4):
        return True, (
            f"claimed {claimed:.1f}y experience but career_history sums to "
            f"only {actual_years:.1f}y (gap {gap:.1f}y)"
        )
    return False, ""


def check_company_founding_vs_tenure(candidate: dict) -> tuple[bool, str]:
    """Flags impossible tenure: e.g. 8 years at a company founded 3 years ago.

    The schema doesn't include company founding year directly, so we use
    this as a placeholder hook — wire in a company-founding-year lookup
    table if/when available (e.g. derived from a small curated list of
    well-known company founding years for plausibility spot-checks).
    This function is intentionally conservative: it only fires when we
    have external founding-year data, never guesses.
    """
    # No founding-year data in candidate_schema.json — left as an extension
    # point. Real implementation would join against a company->founded_year
    # reference table built from public data for at least well-known names.
    return False, ""


def check_skill_claim_vs_assessment(candidate: dict) -> tuple[bool, str]:
    """'Expert' self-claimed skill with a low (or missing) platform-verified
    assessment score is a direct self-report vs verified-evidence contradiction.
    This is one of the cleanest honeypot signals in the dataset.
    """
    skills = candidate.get("skills", []) or []
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}

    contradictions = []
    expert_count = 0
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        if prof == "expert":
            expert_count += 1
            score = assessments.get(name)
            if score is not None and score < 50:
                contradictions.append(f"{name} (claimed expert, scored {score:.0f}/100)")

    # Also flag implausibly many "expert" claims with near-zero usage duration.
    zero_duration_experts = [
        s.get("name", "") for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0
    ]
    if zero_duration_experts:
        contradictions.append(
            f"{len(zero_duration_experts)} skill(s) claimed 'expert' with 0 months used: "
            f"{', '.join(zero_duration_experts[:3])}"
        )

    if contradictions:
        return True, "skill/assessment contradiction: " + "; ".join(contradictions)
    return False, ""


def check_endorsement_inflation(candidate: dict) -> tuple[bool, str]:
    """Disproportionate endorsements relative to network size.

    Threshold calibrated against the real dataset, not guessed: across all
    100k candidates, connection_count never drops below 10 (so a
    'connections < 10' branch is dead code on this data) and the
    endorsements/connections ratio tops out at 5.0, with p99=1.73 and
    p99.9=3.62. A ratio > 3.0 sits clearly above the p99.9 tail without
    being pinned to the single most extreme observed case, making it a
    defensible "implausible relative to nearly everyone else" cutoff.
    """
    signals = candidate.get("redrob_signals", {})
    endorsements = signals.get("endorsements_received", 0) or 0
    connections = signals.get("connection_count", 0) or 0

    if connections > 0 and endorsements / connections > 3.0:
        ratio = endorsements / connections
        return True, (
            f"endorsement-to-connection ratio implausibly high "
            f"({endorsements} endorsements / {connections} connections, "
            f"ratio {ratio:.2f})"
        )
    return False, ""


def check_activity_vs_availability(candidate: dict) -> tuple[bool, str]:
    """open_to_work=True but stale activity, or near-zero recruiter response
    rate — per the signals doc's own framing, this candidate is 'not
    actually available' for hiring purposes regardless of skill fit.
    """
    signals = candidate.get("redrob_signals", {})
    open_to_work = signals.get("open_to_work_flag", False)
    last_active = _parse_date(signals.get("last_active_date"))
    response_rate = signals.get("recruiter_response_rate", 0) or 0

    flags = []
    if last_active:
        days_stale = (date.today() - last_active).days
        if open_to_work and days_stale > 180:
            flags.append(f"open_to_work=True but inactive {days_stale}d")
    if response_rate < 0.05:
        flags.append(f"recruiter_response_rate={response_rate:.2f} (near-zero)")

    if flags:
        return True, "; ".join(flags)
    return False, ""


def check_verification_baseline(candidate: dict) -> tuple[bool, str]:
    """All verification signals false on an otherwise 'perfect' profile is
    itself a trust flag — maps directly to the JD's literal ask: 'a
    shortlist a recruiter can trust.'
    """
    signals = candidate.get("redrob_signals", {})
    verified = [
        signals.get("verified_email", False),
        signals.get("verified_phone", False),
        signals.get("linkedin_connected", False),
    ]
    if not any(verified):
        return True, "no verification signals present (email/phone/linkedin all unverified)"
    return False, ""


def check_title_vs_current_company_history(candidate: dict) -> tuple[bool, str]:
    """current_title/current_company should match the most recent,
    is_current=True entry in career_history. Mismatch = data integrity
    red flag, often present in fabricated/templated profiles.
    """
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    current_entries = [h for h in history if h.get("is_current")]

    if not current_entries:
        if history:
            return True, "no career_history entry marked is_current=True"
        return False, ""

    entry = current_entries[0]
    if entry.get("title") != profile.get("current_title") or \
       entry.get("company") != profile.get("current_company"):
        return True, (
            f"profile.current_title/company ('{profile.get('current_title')}' @ "
            f"'{profile.get('current_company')}') does not match career_history "
            f"current entry ('{entry.get('title')}' @ '{entry.get('company')}')"
        )
    return False, ""


# Replace the CHECKS list and score_candidate function with the versions below.

# Each check now carries a severity tier alongside its point penalty:
#   "hard"  -> deterministic, no-benign-explanation evidence. A single hard
#              flag is sufficient on its own to call a candidate a likely
#              honeypot, regardless of the numeric score.
#   "soft"  -> circumstantial evidence that's individually forgivable (lots
#              of real, non-fraudulent candidates trip these) but still
#              informative in combination with other flags.
#
# This directly encodes the design judgment: an 11-year experience gap or
# an "expert" skill claim with zero months of usage needs no corroboration
# to be disqualifying. An unverified email or a quiet job-seeker does.

CHECKS = [
    ("experience_arithmetic", check_experience_arithmetic, 30, "hard"),
    ("skill_vs_assessment", check_skill_claim_vs_assessment, 30, "hard"),
    ("endorsement_inflation", check_endorsement_inflation, 15, "soft"),
    ("activity_vs_availability", check_activity_vs_availability, 10, "soft"),
    ("verification_baseline", check_verification_baseline, 5, "soft"),
    ("title_history_mismatch", check_title_vs_current_company_history, 10, "soft"),
]


def score_candidate(candidate: dict) -> ConsistencyResult:
    """Run all checks, deduct weighted penalties, return a 0-100 plausibility score.

    Honeypot classification uses two independent paths, not just the score:
      1. ANY single "hard" flag (experience_arithmetic, skill_vs_assessment)
         is sufficient on its own -> is_likely_honeypot = True, regardless
         of the numeric score. These are deterministic, no-benign-
         explanation signals; they don't need corroboration.
      2. Otherwise, fall back to the aggregate score threshold (< 40) as
         before -> catches cases where multiple "soft" signals stack up
         even though none alone is damning.

    Threshold guidance (tune empirically against your hand-labeled set):
      score >= 70  -> plausible, pass to fit scoring normally
      40-70        -> plausible but flagged, fit score gets dampened
      < 40         -> likely honeypot, excluded from top 100 entirely
      (any hard flag) -> likely honeypot, excluded, irrespective of score
    """
    cid = candidate.get("candidate_id", "UNKNOWN")
    score = 100.0
    flags: list[str] = []
    hard_flag_fired = False

    for name, check_fn, penalty, tier in CHECKS:
        violated, detail = check_fn(candidate)
        if violated:
            score -= penalty
            flags.append(f"[{name}] {detail}")
            if tier == "hard":
                hard_flag_fired = True

    score = max(0.0, score)
    is_honeypot = hard_flag_fired or score < 40

    return ConsistencyResult(
        candidate_id=cid,
        consistency_score=score,
        flags=flags,
        is_likely_honeypot=is_honeypot,
    )


def score_all(candidates: list[dict]) -> dict[str, ConsistencyResult]:
    return {c.get("candidate_id", f"UNKNOWN_{i}"): score_candidate(c)
            for i, c in enumerate(candidates)}