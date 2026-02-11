"""
Serialize agent types to JSON-safe dicts for the web UI.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Lazy imports in routes to avoid loading agent/test_problems at module level
# when only serving static files or health.


def world_state_to_dict(state: Any) -> List[Dict[str, Any]]:
    """Serialize WorldState facts to a list of dicts."""
    if state is None:
        return []
    facts_out = []
    for fact in state.facts.values():
        facts_out.append({
            "predicate": fact.predicate,
            "args": list(fact.args),
            "confidence": round(float(fact.confidence), 2),
        })
    return facts_out


def plan_to_dict(plan: Any) -> Dict[str, Any] | None:
    """Serialize Plan to a dict (steps, cost)."""
    if plan is None:
        return None
    return {
        "steps": [
            {"name": a.name, "cost": a.cost, "reversible": a.reversible}
            for a in plan.steps
        ],
        "estimated_cost": plan.estimated_cost,
        "total_cost": plan.total_cost(),
    }


def verification_to_dict(v: Any) -> Dict[str, Any] | None:
    """Serialize VerificationResult to a dict."""
    if v is None:
        return None
    return {
        "satisfied": v.satisfied,
        "confidence": round(float(v.confidence), 2),
        "summary": v.summary(),
        "check_results": [
            {"passed": r.passed, "message": r.message}
            for r in v.check_results
        ],
        "failures": [{"message": r.message} for r in v.failures],
    }


def lessons_to_dict(lessons: List[Any]) -> List[Dict[str, Any]]:
    """Serialize list of Lesson to dicts."""
    return [
        {
            "situation": l.situation,
            "insight": l.insight,
            "outcome": l.outcome,
            "confidence": round(float(l.confidence), 2),
        }
        for l in lessons
    ]


def agent_result_to_dict(result: Any) -> Dict[str, Any]:
    """Serialize AgentResult to a dict for API response."""
    return {
        "status": result.status,
        "message": result.message or "",
        "success": result.success,
        "plan": plan_to_dict(result.plan),
        "state": world_state_to_dict(result.state) if result.state else [],
        "verification": verification_to_dict(result.verification) if result.verification else None,
        "questions": list(result.questions),
        "concerns": list(result.concerns),
        "lessons": lessons_to_dict(result.lessons),
        "failure_reason": result.failure_reason,
        "suggested_next_steps": list(result.suggested_next_steps or []),
    }
