"""
Tool-required and accuracy scenarios for the app and tools comparison.

These scenarios require real tool execution (web_search, run_snippet) to succeed.
They are returned by get_tool_scenarios() and included in get_scenarios_for_run().
"""

from __future__ import annotations

import re
from typing import List

from tests.test_problems import Difficulty, Scenario


def _make_tool_required_scenario() -> Scenario:
    """
    Scenario that REQUIRES tools: the action has require_tool_execution=True, so
    SimpleLLMExecutor does not apply its effects (state unchanged -> goal not met).
    Only ToolCapableExecutor can add search_done via LLM tool call + new_facts.
    """
    from deliberative_agent import WorldState, Goal, Action, Fact, Confidence, ConfidenceSource, VerificationPlan
    from deliberative_agent.verification import PredicateCheck

    state = WorldState()
    state.add_fact(Fact("task_started", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))
    action = Action(
        name="do_web_research",
        description=(
            "You must call the web_search tool with params {\"query\": \"Python asyncio tutorial\"}. "
            "After you receive the tool result, respond with {\"done\": true, \"effects\": [\"Searched the web\"], "
            "\"new_facts\": [{\"predicate\": \"search_done\", \"args\": []}]}. "
            "Do not respond with done until you have called the tool."
        ),
        preconditions=[lambda s: s.has_fact("task_started") is not None],
        effects=[Fact("search_done", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=2.0,
        reversible=False,
        require_tool_execution=True,
    )
    goal = Goal(
        id="research_done",
        description="search_done fact must be present (only achievable by executor adding it via tool)",
        predicate=lambda s: s.has_fact("search_done") is not None,
        verification=VerificationPlan(checks=[
            PredicateCheck(
                predicate=lambda s: s.has_fact("search_done") is not None,
                description="search_done fact present",
            )
        ]),
    )
    return Scenario(
        name="Web research (tools REQUIRED)",
        description="Goal only reachable if executor calls web_search and adds new_facts",
        difficulty=Difficulty.HARD,
        initial_state=state,
        goal=goal,
        available_actions=[action],
        success_predicate=goal.predicate,
        max_steps=5,
    )


def _make_accuracy_web_version_scenario() -> Scenario:
    """
    Accuracy scenario: goal requires a fact whose value must come from real search results.
    Verification checks version format.
    """
    from deliberative_agent import WorldState, Goal, Action, Fact, Confidence, ConfidenceSource, VerificationPlan
    from deliberative_agent.verification import PredicateCheck

    state = WorldState()
    state.add_fact(Fact("task_started", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))
    version_pattern = re.compile(r"^\d+\.\d+(\.\d+)?$")

    def has_valid_version(s):
        facts = s.get_facts_by_predicate("python_latest_version")
        if not facts:
            return False
        args = getattr(facts[0], "args", ()) or ()
        if not args:
            return False
        return bool(version_pattern.match(str(args[0]).strip()))

    action = Action(
        name="fetch_python_version",
        description=(
            "Call web_search with params {\"query\": \"Python 3 latest stable version number\"}. "
            "From the results, extract the current stable version (e.g. 3.12 or 3.12.1). "
            "Respond with {\"done\": true, \"effects\": [\"Fetched version\"], "
            "\"new_facts\": [{\"predicate\": \"python_latest_version\", \"args\": [\"<version>\"]]} "
            "using the actual version string you found. Do not respond with done until you called the tool."
        ),
        preconditions=[lambda s: s.has_fact("task_started") is not None],
        effects=[Fact("python_latest_version", ("0.0",), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=2.0,
        reversible=False,
        require_tool_execution=True,
    )
    goal = Goal(
        id="version_known",
        description="python_latest_version fact present with valid version format (from real search)",
        predicate=has_valid_version,
        verification=VerificationPlan(checks=[
            PredicateCheck(predicate=has_valid_version, description="python_latest_version has valid version format"),
        ]),
    )
    return Scenario(
        name="Python version from web (accuracy)",
        description="Fetch real version from search; verification checks format",
        difficulty=Difficulty.HARD,
        initial_state=state,
        goal=goal,
        available_actions=[action],
        success_predicate=has_valid_version,
        max_steps=5,
    )


def _make_multi_step_tool_chain_scenario() -> Scenario:
    """
    Multi-step scenario: two actions that both require tools (search, then snippet).
    """
    from deliberative_agent import WorldState, Goal, Action, Fact, Confidence, ConfidenceSource, VerificationPlan
    from deliberative_agent.verification import PredicateCheck

    state = WorldState()
    state.add_fact(Fact("task_started", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))

    def both_facts(s):
        return s.has_fact("search_done") is not None and s.has_fact("snippet_done") is not None

    action_search = Action(
        name="do_web_search",
        description=(
            "Call web_search with params {\"query\": \"Python asyncio\"}. "
            "Then respond with {\"done\": true, \"effects\": [\"Searched\"], "
            "\"new_facts\": [{\"predicate\": \"search_done\", \"args\": []}]}. "
            "Do not respond with done until you have called the tool."
        ),
        preconditions=[lambda s: s.has_fact("task_started") is not None],
        effects=[Fact("search_done", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=2.0,
        reversible=False,
        require_tool_execution=True,
    )
    action_snippet = Action(
        name="run_code_step",
        description=(
            "Call run_snippet with params {\"language\": \"python\", \"code\": \"print(2+2)\"}. "
            "After you receive the tool result, respond with {\"done\": true, \"effects\": [\"Ran snippet\"], "
            "\"new_facts\": [{\"predicate\": \"snippet_done\", \"args\": []}]}. "
            "Do not respond with done until you have called the tool."
        ),
        preconditions=[lambda s: s.has_fact("search_done") is not None],
        effects=[Fact("snippet_done", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=1.0,
        reversible=False,
        require_tool_execution=True,
    )
    goal = Goal(
        id="chain_done",
        description="Both search_done and snippet_done (multi-step tool chain)",
        predicate=both_facts,
        verification=VerificationPlan(checks=[
            PredicateCheck(predicate=both_facts, description="search_done and snippet_done present"),
        ]),
    )
    return Scenario(
        name="Multi-step: search then snippet",
        description="Two tool-required actions in sequence",
        difficulty=Difficulty.HARD,
        initial_state=state,
        goal=goal,
        available_actions=[action_search, action_snippet],
        success_predicate=both_facts,
        max_steps=5,
    )


def _make_snippet_accuracy_scenario() -> Scenario:
    """
    Accuracy scenario: goal requires a fact whose value is the actual stdout of a snippet.
    """
    from deliberative_agent import WorldState, Goal, Action, Fact, Confidence, ConfidenceSource, VerificationPlan
    from deliberative_agent.verification import PredicateCheck

    state = WorldState()
    state.add_fact(Fact("task_started", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))

    def has_snippet_result_42(s):
        facts = s.get_facts_by_predicate("snippet_result")
        if not facts:
            return False
        args = getattr(facts[0], "args", ()) or ()
        return len(args) >= 1 and "42" in str(args[0])

    action = Action(
        name="run_snippet_and_capture",
        description=(
            "Call run_snippet with params {\"language\": \"python\", \"code\": \"print(42)\"}. "
            "From the tool result, take the stdout value. "
            "Respond with {\"done\": true, \"effects\": [\"Ran snippet\"], "
            "\"new_facts\": [{\"predicate\": \"snippet_result\", \"args\": [\"<stdout from result>\"]]}. "
            "Do not respond with done until you have called the tool."
        ),
        preconditions=[lambda s: s.has_fact("task_started") is not None],
        effects=[Fact("snippet_result", ("42",), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=1.0,
        reversible=False,
        require_tool_execution=True,
    )
    goal = Goal(
        id="snippet_captured",
        description="snippet_result fact present with actual snippet stdout containing 42",
        predicate=has_snippet_result_42,
        verification=VerificationPlan(checks=[
            PredicateCheck(predicate=has_snippet_result_42, description="snippet_result contains 42"),
        ]),
    )
    return Scenario(
        name="Snippet output (accuracy)",
        description="Run snippet and verify output content",
        difficulty=Difficulty.HARD,
        initial_state=state,
        goal=goal,
        available_actions=[action],
        success_predicate=has_snippet_result_42,
        max_steps=5,
    )


def get_tool_scenarios() -> List[Scenario]:
    """Return all tool-required and accuracy scenarios (same shape as test_problems.Scenario)."""
    return [
        _make_tool_required_scenario(),
        _make_accuracy_web_version_scenario(),
        _make_multi_step_tool_chain_scenario(),
        _make_snippet_accuracy_scenario(),
    ]
