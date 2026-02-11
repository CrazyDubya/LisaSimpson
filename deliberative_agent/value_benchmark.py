"""
Real-world value benchmark: Deliberative Agent vs "Ralph Wiggum" baseline.

Compares planning + verification + learning against blind random-order execution
to quantify whether we add real value (success rate, steps to goal, cost).
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .actions import Action
from .agent import DeliberativeAgent
from .core import Confidence, ConfidenceSource, WorldState
from .execution import ActionExecutor, ExecutionStepResult, Executor
from .goals import Goal
from .planning import Planner
from .verification import Check, CheckResult, VerificationPlan


@dataclass
class RealWorldScenario:
    """A single real-world-style scenario (goal + state + actions)."""

    id: str
    name: str
    description: str
    initial_state: WorldState
    goal: Goal
    actions: List[Action]
    max_steps: int = 15
    # Optimal number of steps when planned (for reporting)
    optimal_steps: Optional[int] = None


@dataclass
class ValueRunResult:
    """Result of one run (deliberative or baseline)."""

    scenario_id: str
    run_type: str  # "deliberative" | "baseline"
    success: bool
    steps: int
    total_cost: float
    duration_seconds: float
    failure_reason: Optional[str] = None


@dataclass
class ValueReport:
    """Aggregate comparison report."""

    scenarios: List[str]
    deliberative_results: List[ValueRunResult] = field(default_factory=list)
    baseline_results: List[ValueRunResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Human-readable summary."""
        lines = [
            "",
            "=" * 70,
            "VALUE BENCHMARK: Deliberative vs Ralph Wiggum Baseline",
            "=" * 70,
        ]
        for sid in self.scenarios:
            delibs = [r for r in self.deliberative_results if r.scenario_id == sid]
            bases = [r for r in self.baseline_results if r.scenario_id == sid]
            d = delibs[0] if delibs else None
            b_list = bases
            n_baseline = len(b_list)
            n_ok = sum(1 for r in b_list if r.success)
            avg_steps_b = sum(r.steps for r in b_list) / n_baseline if n_baseline else 0
            avg_cost_b = sum(r.total_cost for r in b_list) / n_baseline if n_baseline else 0
            lines.append(f"\n  Scenario: {sid}")
            if d:
                lines.append(
                    f"    Deliberative: success={d.success}, steps={d.steps}, "
                    f"cost={d.total_cost:.1f}, time={d.duration_seconds:.2f}s"
                )
            lines.append(
                f"    Baseline (n={n_baseline}): success_rate={n_ok}/{n_baseline}, "
                f"avg_steps={avg_steps_b:.1f}, avg_cost={avg_cost_b:.1f}"
            )
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)


class BaselineRunner:
    """
    "Ralph Wiggum" baseline: no planning, no verification, no memory.

    Repeatedly picks a random applicable action until goal is satisfied or max_steps.
    Shows what happens without GOAP-style planning (wrong order, wasted steps).
    """

    def __init__(self, executor: ActionExecutor, max_steps: int = 20, seed: Optional[int] = None):
        self.executor = executor
        self.max_steps = max_steps
        self._rng = random.Random(seed)

    async def run(
        self,
        scenario: RealWorldScenario,
    ) -> ValueRunResult:
        state = scenario.initial_state.copy()
        steps = 0
        total_cost = 0.0
        start = time.perf_counter()
        failure_reason: Optional[str] = None

        while steps < min(scenario.max_steps, self.max_steps):
            if scenario.goal.is_satisfied(state):
                break
            applicable = [a for a in scenario.actions if a.applicable(state)]
            if not applicable:
                failure_reason = "no applicable action (wrong order or stuck)"
                break
            action = self._rng.choice(applicable)
            step_result = await self.executor.execute(action, state)
            if not step_result.success:
                failure_reason = str(step_result.error) if step_result.error else "execution failed"
                break
            state = step_result.new_state
            steps += 1
            total_cost += action.cost

        duration = time.perf_counter() - start
        success = scenario.goal.is_satisfied(state)
        return ValueRunResult(
            scenario_id=scenario.id,
            run_type="baseline",
            success=success,
            steps=steps,
            total_cost=total_cost,
            duration_seconds=duration,
            failure_reason=failure_reason if not success else None,
        )


def _make_scenario_safe_deploy() -> RealWorldScenario:
    """Safe production deployment: strict order (tests -> backup -> deploy -> verify)."""
    from .core import Confidence, ConfidenceSource, Fact

    state = WorldState()
    state.add_fact(Fact("environment", ("production",), Confidence(1.0, ConfidenceSource.OBSERVATION)))

    run_tests = Action(
        name="run_tests",
        description="Run test suite",
        preconditions=[],
        effects=[Fact("tests_run", (), Confidence(0.95, ConfidenceSource.VERIFICATION))],
        cost=2.0,
        reversible=False,
    )
    create_backup = Action(
        name="create_backup",
        description="Create backup",
        preconditions=[lambda s: s.has_fact("tests_run") is not None],
        effects=[Fact("backup_created", (), Confidence(0.99, ConfidenceSource.VERIFICATION))],
        cost=3.0,
        reversible=False,
    )
    deploy = Action(
        name="deploy",
        description="Deploy to production",
        preconditions=[
            lambda s: s.has_fact("tests_run") is not None and s.has_fact("backup_created") is not None
        ],
        effects=[Fact("deployed", (), Confidence(0.85, ConfidenceSource.INFERENCE))],
        cost=5.0,
        reversible=True,
    )
    verify = Action(
        name="verify_deployment",
        description="Verify deployment",
        preconditions=[lambda s: s.has_fact("deployed") is not None],
        effects=[Fact("deployment_verified", (), Confidence(0.95, ConfidenceSource.VERIFICATION))],
        cost=2.0,
        reversible=False,
    )

    goal = Goal(
        id="safe_deploy",
        description="Safe production deployment with tests and backup",
        predicate=lambda s: s.has_fact("deployment_verified") is not None,
        verification=VerificationPlan([]),
    )

    return RealWorldScenario(
        id="safe_deploy",
        name="Safe Production Deployment",
        description="Deploy with tests and backup; order must be tests -> backup -> deploy -> verify",
        initial_state=state,
        goal=goal,
        actions=[run_tests, create_backup, deploy, verify],
        max_steps=10,
        optimal_steps=4,
    )


def _make_scenario_ci_pipeline() -> RealWorldScenario:
    """CI pipeline: code -> tests -> lint -> artifact. Strict linear order."""
    from .core import Confidence, ConfidenceSource, Fact

    state = WorldState()
    state.add_fact(Fact("repo_ready", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))

    write_code = Action(
        name="write_code",
        description="Write feature code",
        preconditions=[lambda s: s.has_fact("repo_ready") is not None],
        effects=[Fact("code_written", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=3.0,
        reversible=True,
    )
    write_tests = Action(
        name="write_tests",
        description="Write unit tests",
        preconditions=[lambda s: s.has_fact("code_written") is not None],
        effects=[Fact("tests_written", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=2.0,
        reversible=True,
    )
    run_tests = Action(
        name="run_tests",
        description="Run tests",
        preconditions=[lambda s: s.has_fact("tests_written") is not None],
        effects=[Fact("tests_passed", (), Confidence(0.95, ConfidenceSource.VERIFICATION))],
        cost=1.0,
        reversible=False,
    )
    lint = Action(
        name="lint",
        description="Run linter",
        preconditions=[lambda s: s.has_fact("code_written") is not None],
        effects=[Fact("lint_ok", (), Confidence(0.9, ConfidenceSource.VERIFICATION))],
        cost=1.0,
        reversible=False,
    )
    build_artifact = Action(
        name="build_artifact",
        description="Build release artifact",
        preconditions=[
            lambda s: s.has_fact("tests_passed") is not None and s.has_fact("lint_ok") is not None
        ],
        effects=[Fact("artifact_built", (), Confidence(0.9, ConfidenceSource.VERIFICATION))],
        cost=2.0,
        reversible=False,
    )

    goal = Goal(
        id="ci_pipeline",
        description="Complete CI pipeline with tests and lint",
        predicate=lambda s: s.has_fact("artifact_built") is not None,
        verification=VerificationPlan([]),
    )

    return RealWorldScenario(
        id="ci_pipeline",
        name="CI Pipeline",
        description="Linear pipeline: code -> tests + lint -> artifact; order matters",
        initial_state=state,
        goal=goal,
        actions=[write_code, write_tests, run_tests, lint, build_artifact],
        max_steps=12,
        optimal_steps=5,
    )


def _make_scenario_backend_with_deps() -> RealWorldScenario:
    """Backend dev: requirements -> (db design, api design) -> impl db -> impl api -> tests."""
    from .core import Confidence, ConfidenceSource, Fact

    state = WorldState()
    state.add_fact(Fact("requirements_defined", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))

    design_db = Action(
        name="design_database",
        description="Design database schema",
        preconditions=[lambda s: s.has_fact("requirements_defined") is not None],
        effects=[Fact("database_designed", (), Confidence(0.85, ConfidenceSource.INFERENCE))],
        cost=4.0,
        reversible=True,
    )
    design_api = Action(
        name="design_api",
        description="Design API",
        preconditions=[lambda s: s.has_fact("requirements_defined") is not None],
        effects=[Fact("api_designed", (), Confidence(0.85, ConfidenceSource.INFERENCE))],
        cost=4.0,
        reversible=True,
    )
    impl_db = Action(
        name="implement_database",
        description="Implement database",
        preconditions=[lambda s: s.has_fact("database_designed") is not None],
        effects=[Fact("database_implemented", (), Confidence(0.9, ConfidenceSource.VERIFICATION))],
        cost=6.0,
        reversible=True,
    )
    impl_api = Action(
        name="implement_api",
        description="Implement API",
        preconditions=[
            lambda s: s.has_fact("api_designed") is not None
            and s.has_fact("database_implemented") is not None
        ],
        effects=[Fact("api_implemented", (), Confidence(0.9, ConfidenceSource.VERIFICATION))],
        cost=6.0,
        reversible=True,
    )
    run_tests = Action(
        name="run_tests",
        description="Run backend tests",
        preconditions=[lambda s: s.has_fact("api_implemented") is not None],
        effects=[Fact("tests_passed", (), Confidence(0.95, ConfidenceSource.VERIFICATION))],
        cost=2.0,
        reversible=False,
    )

    goal = Goal(
        id="backend_done",
        description="Backend implemented and tested",
        predicate=lambda s: s.has_fact("tests_passed") is not None,
        verification=VerificationPlan([]),
    )

    return RealWorldScenario(
        id="backend_with_deps",
        name="Backend with Dependencies",
        description="DAG: requirements -> design(db+api) -> impl db -> impl api -> tests",
        initial_state=state,
        goal=goal,
        actions=[design_db, design_api, impl_db, impl_api, run_tests],
        max_steps=12,
        optimal_steps=5,
    )


class _FactPresentCheck(Check):
    """Verification check that passes only if a fact exists (for value benchmark)."""

    def __init__(self, predicate: str, *args: Any):
        self._predicate = predicate
        self._args = args

    async def run(self, state: WorldState) -> CheckResult:
        fact = state.has_fact(self._predicate, *self._args)
        passed = fact is not None
        conf = Confidence(0.95 if passed else 0.0, ConfidenceSource.VERIFICATION)
        return CheckResult(
            passed=passed,
            confidence=conf,
            message=f"Fact {self._predicate}{self._args} present" if passed else f"Missing {self._predicate}{self._args}",
        )


def _make_scenario_verified_deploy() -> RealWorldScenario:
    """Deploy then verify; goal includes VerificationPlan so semantic verification is run."""
    from .core import Fact

    state = WorldState()
    deploy = Action(
        name="deploy",
        description="Deploy to staging",
        preconditions=[],
        effects=[Fact("deployed", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=3.0,
        reversible=True,
    )
    verify = Action(
        name="verify_deploy",
        description="Verify deployment",
        preconditions=[lambda s: s.has_fact("deployed") is not None],
        effects=[Fact("verified", (), Confidence(0.95, ConfidenceSource.VERIFICATION))],
        cost=1.0,
        reversible=False,
    )
    goal = Goal(
        id="verified_deploy",
        description="Deploy and verify (verification plan required)",
        predicate=lambda s: s.has_fact("deployed") is not None and s.has_fact("verified") is not None,
        verification=VerificationPlan(checks=[_FactPresentCheck("verified")]),
    )
    return RealWorldScenario(
        id="verified_deploy",
        name="Verified Deploy (with VerificationPlan)",
        description="Deploy then verify; goal uses semantic verification, not just predicate",
        initial_state=state,
        goal=goal,
        actions=[deploy, verify],
        max_steps=5,
        optimal_steps=2,
    )


def _make_scenario_bugfix_regression() -> RealWorldScenario:
    """Bug fix with regression test: reproduce -> fix -> add test -> run tests."""
    from .core import Confidence, ConfidenceSource, Fact

    state = WorldState()
    state.add_fact(Fact("bug_reported", (), Confidence(1.0, ConfidenceSource.OBSERVATION)))

    reproduce = Action(
        name="reproduce_bug",
        description="Reproduce the bug",
        preconditions=[lambda s: s.has_fact("bug_reported") is not None],
        effects=[Fact("bug_reproduced", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=2.0,
        reversible=False,
    )
    fix = Action(
        name="write_fix",
        description="Write the fix",
        preconditions=[lambda s: s.has_fact("bug_reproduced") is not None],
        effects=[Fact("fix_written", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=3.0,
        reversible=True,
    )
    add_test = Action(
        name="add_regression_test",
        description="Add regression test",
        preconditions=[lambda s: s.has_fact("fix_written") is not None],
        effects=[Fact("regression_test_added", (), Confidence(0.9, ConfidenceSource.INFERENCE))],
        cost=2.0,
        reversible=True,
    )
    run_tests = Action(
        name="run_tests",
        description="Run tests",
        preconditions=[lambda s: s.has_fact("regression_test_added") is not None],
        effects=[Fact("tests_passed", (), Confidence(0.95, ConfidenceSource.VERIFICATION))],
        cost=1.0,
        reversible=False,
    )

    goal = Goal(
        id="bugfix_done",
        description="Bug fixed with regression test and tests passing",
        predicate=lambda s: s.has_fact("tests_passed") is not None,
        verification=VerificationPlan([]),
    )

    return RealWorldScenario(
        id="bugfix_regression",
        name="Bug Fix with Regression Test",
        description="Reproduce -> fix -> add regression test -> run tests",
        initial_state=state,
        goal=goal,
        actions=[reproduce, fix, add_test, run_tests],
        max_steps=8,
        optimal_steps=4,
    )


def get_real_world_scenarios() -> List[RealWorldScenario]:
    """Return real-world-style scenarios where planning clearly adds value."""
    return [
        _make_scenario_safe_deploy(),
        _make_scenario_ci_pipeline(),
        _make_scenario_backend_with_deps(),
        _make_scenario_verified_deploy(),
        _make_scenario_bugfix_regression(),
    ]


class MockExecutor(ActionExecutor):
    """Deterministic executor that just applies action effects (no LLM). For fast value benchmark."""

    async def execute(self, action: Action, state: WorldState) -> ExecutionStepResult:
        new_state = action.apply(state)
        return ExecutionStepResult.successful(new_state=new_state, effects=[action.description])


async def run_deliberative(
    scenario: RealWorldScenario,
    executor: ActionExecutor,
) -> ValueRunResult:
    """Run one scenario with the full deliberative agent (plan then execute)."""
    agent = DeliberativeAgent(actions=scenario.actions, action_executor=executor)
    start = time.perf_counter()
    result = await agent.achieve(scenario.goal, scenario.initial_state.copy())
    duration = time.perf_counter() - start
    steps = len(result.plan.steps) if result.plan else 0
    total_cost = sum(s.cost for s in result.plan.steps) if result.plan else 0.0
    return ValueRunResult(
        scenario_id=scenario.id,
        run_type="deliberative",
        success=result.success,
        steps=steps,
        total_cost=total_cost,
        duration_seconds=duration,
        failure_reason=None if result.success else (result.message or "unknown"),
    )


async def run_value_benchmark(
    scenarios: Optional[List[RealWorldScenario]] = None,
    executor: Optional[ActionExecutor] = None,
    baseline_runs_per_scenario: int = 10,
    baseline_seed: Optional[int] = 42,
) -> ValueReport:
    """
    Run deliberative vs baseline on each scenario; return comparison report.

    Uses MockExecutor if executor is None (fast, no LLM). Baseline runs multiple
    times per scenario to average over random order.
    """
    scenarios = scenarios or get_real_world_scenarios()
    executor = executor or MockExecutor()

    report = ValueReport(scenarios=[s.id for s in scenarios])
    baseline_runner = BaselineRunner(executor=executor, max_steps=25, seed=baseline_seed)

    for scenario in scenarios:
        # One deliberative run (deterministic)
        d_result = await run_deliberative(scenario, executor)
        report.deliberative_results.append(d_result)

        # Multiple baseline runs (random order)
        for i in range(baseline_runs_per_scenario):
            runner = BaselineRunner(
                executor=executor,
                max_steps=25,
                seed=baseline_seed + i if baseline_seed is not None else None,
            )
            b_result = await runner.run(scenario)
            report.baseline_results.append(b_result)

    # Summary stats
    d_ok = sum(1 for r in report.deliberative_results if r.success)
    d_steps = [r.steps for r in report.deliberative_results if r.success]
    b_all = report.baseline_results
    b_ok = sum(1 for r in b_all if r.success)
    b_steps = [r.steps for r in b_all if r.success]
    report.summary = {
        "deliberative_success_rate": d_ok / len(report.deliberative_results) if report.deliberative_results else 0,
        "deliberative_avg_steps_when_success": sum(d_steps) / len(d_steps) if d_steps else 0,
        "baseline_success_rate": b_ok / len(b_all) if b_all else 0,
        "baseline_avg_steps_when_success": sum(b_steps) / len(b_steps) if b_steps else 0,
        "baseline_runs_per_scenario": baseline_runs_per_scenario,
    }
    return report


@dataclass
class MemoryRunPair:
    """Result of running the same scenario twice with shared agent memory."""

    scenario_id: str
    run1_success: bool
    run1_steps: int
    run2_success: bool
    run2_steps: int


async def run_memory_value_demo(
    scenarios: Optional[List[RealWorldScenario]] = None,
    executor: Optional[ActionExecutor] = None,
) -> List[MemoryRunPair]:
    """
    Run each scenario twice with a single agent (shared memory). Reports steps for run1 vs run2.
    Shows that the same agent can re-attempt with memory of past episodes/lessons.
    """
    from .memory import Memory

    scenarios = scenarios or get_real_world_scenarios()
    executor = executor or MockExecutor()
    shared_memory = Memory()
    results: List[MemoryRunPair] = []
    for scenario in scenarios:
        agent = DeliberativeAgent(
            actions=scenario.actions,
            action_executor=executor,
            memory=shared_memory,
        )
        # Run 1
        r1 = await agent.achieve(scenario.goal, scenario.initial_state.copy())
        steps1 = len(r1.plan.steps) if r1.plan else 0
        # Run 2 (new agent instance, same memory, fresh initial state)
        agent2 = DeliberativeAgent(
            actions=scenario.actions,
            action_executor=executor,
            memory=shared_memory,
        )
        r2 = await agent2.achieve(scenario.goal, scenario.initial_state.copy())
        steps2 = len(r2.plan.steps) if r2.plan else 0
        results.append(
            MemoryRunPair(
                scenario_id=scenario.id,
                run1_success=r1.success,
                run1_steps=steps1,
                run2_success=r2.success,
                run2_steps=steps2,
            )
        )
    return results
