

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from consistency_gate import score_candidate
from jd_disqualifiers import apply_disqualifiers


def make_base_candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": "Senior ML Engineer",
            "summary": "Built and deployed production recommendation systems at scale.",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 7,
            "current_title": "Senior ML Engineer",
            "current_company": "ProductCo",
            "current_company_size": "201-500",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "ProductCo", "title": "Senior ML Engineer",
                "start_date": "2022-01-01", "end_date": None,
                "duration_months": 36, "is_current": True,
                "industry": "Technology", "company_size": "201-500",
                "description": "Shipped and deployed a production recommendation system serving millions of users.",
            },
            {
                "company": "EarlierCo", "title": "ML Engineer",
                "start_date": "2018-01-01", "end_date": "2021-12-31",
                "duration_months": 48, "is_current": False,
                "industry": "Technology", "company_size": "51-200",
                "description": "Built search ranking and information retrieval systems.",
            },
        ],
        "education": [],
        "skills": [
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 60},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 5, "duration_months": 36},
        ],
        "certifications": [],
        "redrob_signals": {
            "profile_completeness_score": 90,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-10",
            "open_to_work_flag": True,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.7,
            "avg_response_time_hours": 5,
            "skill_assessment_scores": {"Python": 85, "Embeddings": 75},
            "connection_count": 300,
            "endorsements_received": 15,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 45},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 60,
            "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 3,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    base.update(overrides)
    return base


def make_honeypot_candidate():
    """8 years experience claimed, career_history sums to far less.
    Expert-claimed skills with 0 duration. Endorsement inflation.
    """
    c = make_base_candidate(candidate_id="CAND_0000002")
    c["profile"]["years_of_experience"] = 8
    c["career_history"] = [
        {
            "company": "TinyStartup", "title": "ML Engineer",
            "start_date": "2025-06-01", "end_date": None,
            "duration_months": 12, "is_current": True,
            "industry": "Technology", "company_size": "1-10",
            "description": "Worked on AI projects.",
        }
    ]
    c["skills"] = [
        {"name": "RAG", "proficiency": "expert", "endorsements": 80, "duration_months": 0},
        {"name": "Pinecone", "proficiency": "expert", "endorsements": 80, "duration_months": 0},
    ]
    c["redrob_signals"]["connection_count"] = 5
    c["redrob_signals"]["endorsements_received"] = 160
    c["redrob_signals"]["skill_assessment_scores"] = {}
    return c


def make_services_only_candidate():
    c = make_base_candidate(candidate_id="CAND_0000003")
    c["career_history"] = [
        {
            "company": "TCS", "title": "Senior Developer",
            "start_date": "2020-01-01", "end_date": None,
            "duration_months": 60, "is_current": True,
            "industry": "IT Services", "company_size": "10001+",
            "description": "Worked on client projects.",
        },
        {
            "company": "Infosys", "title": "Developer",
            "start_date": "2017-01-01", "end_date": "2019-12-31",
            "duration_months": 36, "is_current": False,
            "industry": "IT Services", "company_size": "10001+",
            "description": "Worked on enterprise systems.",
        },
    ]
    c["profile"]["current_title"] = "Senior Developer"
    c["profile"]["current_company"] = "TCS"
    return c


def run():
    print("=== Test 1: Clean, plausible candidate ===")
    good = make_base_candidate()
    result_a = score_candidate(good)
    result_b = apply_disqualifiers(good)
    print(f"Consistency score: {result_a.consistency_score} (flags: {result_a.flags})")
    print(f"JD multiplier: {result_b.multiplier} (triggered: {result_b.triggered_rules})")
    assert result_a.consistency_score >= 70, "Clean candidate should pass consistency gate"
    assert result_b.multiplier == 1.0, "Clean candidate should trigger no JD disqualifiers"
    print("PASS\n")

    print("=== Test 2: Honeypot candidate ===")
    honeypot = make_honeypot_candidate()
    result_a = score_candidate(honeypot)
    print(f"Consistency score: {result_a.consistency_score} (flags: {result_a.flags})")
    print(f"Is likely honeypot: {result_a.is_likely_honeypot}")
    assert result_a.is_likely_honeypot, "Honeypot should be flagged"
    print("PASS\n")

    print("=== Test 3: Pure-services-career candidate ===")
    services = make_services_only_candidate()
    result_b = apply_disqualifiers(services)
    print(f"JD multiplier: {result_b.multiplier} (triggered: {result_b.triggered_rules})")
    assert result_b.multiplier < 1.0, "Pure-services candidate should be down-weighted"
    print("PASS\n")
    print("=== Test 4: Single hard flag (experience_arithmetic), otherwise clean ===")
candidate_4 = {
    "candidate_id": "CAND_TEST_0004",
    "profile": {
        "years_of_experience": 12.0,
        "current_title": "Engineering Manager",
        "current_company": "ProductCo",
    },
    "career_history": [
        {
            "company": "ProductCo",
            "title": "Engineering Manager",
            "duration_months": 18,
            "is_current": True,
        }
    ],
    "skills": [
        {"name": "Python", "proficiency": "advanced", "duration_months": 24}
    ],
    "redrob_signals": {
        "endorsements_received": 20,
        "connection_count": 400,
        "open_to_work_flag": True,
        "last_active_date": "2026-06-10",
        "recruiter_response_rate": 0.5,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
        "skill_assessment_scores": {},
    },
}
result_4 = score_candidate(candidate_4)
print(f"Consistency score: {result_4.consistency_score} (flags: {result_4.flags})")
print(f"Is likely honeypot: {result_4.is_likely_honeypot}")
assert result_4.is_likely_honeypot is True, (
    "Expected a single hard flag (12y claimed vs 1.5y actual) to be "
    "sufficient for honeypot classification on its own, regardless of "
    "the remaining score."
)
assert result_4.consistency_score == 70.0, (
    f"Expected score 70.0 (100 - 30 for the one hard flag), got "
    f"{result_4.consistency_score}"
)
print("PASS\n")

print("=== Test 5: Only soft flags, but enough to cross the score threshold ===")
candidate_5 = {
    "candidate_id": "CAND_TEST_0005",
    "profile": {
        "years_of_experience": 5.0,
        "current_title": "Sales Executive",
        "current_company": "Acme Corp",
    },
    "career_history": [
        {
            "company": "Acme Corp",
            "title": "Sales Executive",
            "duration_months": 60,
            "is_current": True,
        }
    ],
    "skills": [
        {"name": "Salesforce CRM", "proficiency": "intermediate", "duration_months": 30}
    ],
    "redrob_signals": {
        "endorsements_received": 45,
        "connection_count": 10,
        "open_to_work_flag": True,
        "last_active_date": "2025-01-01",
        "recruiter_response_rate": 0.02,
        "verified_email": False,
        "verified_phone": False,
        "linkedin_connected": False,
        "skill_assessment_scores": {},
    },
}
result_5 = score_candidate(candidate_5)
print(f"Consistency score: {result_5.consistency_score} (flags: {result_5.flags})")
print(f"Is likely honeypot: {result_5.is_likely_honeypot}")
assert result_5.is_likely_honeypot is False, (
    "Three soft flags totaling 30 points should NOT cross the honeypot "
    "bar on their own (score 70, still >= 40) -- soft signals require "
    "more accumulation than a single hard flag to trigger exclusion."
)
print("PASS\n")



print("All sanity tests passed.")


if __name__ == "__main__":
    run()