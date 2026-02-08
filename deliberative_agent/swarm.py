"""
Swarm orchestration for hierarchical goal execution.

Provides:
- Goal DAG construction from Goal dependencies and CompositeGoal groups
- Goal decomposition via HierarchicalPlanner decomposers
- Capability-aware agent assignment
- Parallel branch execution with dependency joins
- Snapshot data for dashboard/observability layers
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

from .agent import AgentResult, DeliberativeAgent
from .core import WorldState
from .goals import CompositeGoal, Goal
from .planning import HierarchicalPlanner
from .verification import VerificationPlan


@dataclass
class SwarmMember:
    """A specialized agent participating in a swarm run."""

    name: str
    agent: DeliberativeAgent
    capabilities: Set[str] = field(default_factory=set)

    def can_handle(self, goal: Goal) -> bool:
        """Check if this member can handle the goal's capability requirements."""
        required = set(goal.metadata.get("required_capabilities", []))
        required.update(goal.metadata.get("required_tools", []))
        if not required:
            return True
        return required.issubset(self.capabilities)


@dataclass
class GoalNode:
    """A node in the executable goal DAG."""

    goal: Goal
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


@dataclass
class SwarmNodeResult:
    """Execution outcome for a single goal node."""

    goal_id: str
    goal_description: str
    status: str
    agent_name: str
    message: str = ""
    concerns: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_ms: float = 0.0
    state: Optional[WorldState] = None


@dataclass
class SwarmRunResult:
    """Result for an entire swarm execution."""

    run_id: str
    status: str
    node_results: Dict[str, SwarmNodeResult]
    final_state: WorldState
    failed_nodes: List[str] = field(default_factory=list)
    skipped_nodes: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == "success"


@dataclass
class SwarmDashboardNode:
    """Dashboard-friendly view of one node."""

    goal_id: str
    description: str
    status: str
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: str = ""


@dataclass
class SwarmDashboardSnapshot:
    """Snapshot for GUI/observability polling."""

    run_id: str
    timestamp: datetime
    total_nodes: int
    pending_nodes: int
    running_nodes: int
    completed_nodes: int
    failed_nodes: int
    nodes: List[SwarmDashboardNode] = field(default_factory=list)


@dataclass
class TodoItem:
    """
    A multi-phase todo item represented as a Goal node.

    Dependencies encode phase ordering and branch joins.
    """

    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    phase: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    priority: int = 0


def build_todo_goal_graph(items: Sequence[TodoItem]) -> List[Goal]:
    """
    Build Goal objects from todo items.

    Each generated goal uses the fact `todo_done(<item_id>)` as its satisfaction
    predicate, allowing execution systems to mark completion explicitly.
    """
    by_id: Dict[str, Goal] = {}

    for item in items:
        by_id[item.id] = Goal(
            id=item.id,
            description=item.description,
            predicate=(
                lambda state, item_id=item.id: state.has_fact("todo_done", item_id)
                is not None
            ),
            verification=VerificationPlan(checks=[]),
            priority=item.priority,
            metadata={
                "type": "todo_item",
                "phase": item.phase,
                "required_capabilities": list(item.required_capabilities),
            },
        )

    for item in items:
        goal = by_id[item.id]
        goal.dependencies = [by_id[dep_id] for dep_id in item.dependencies if dep_id in by_id]

    return list(by_id.values())


class SwarmManager:
    """
    Orchestrates a hierarchical swarm of agents across a goal DAG.

    This manager composes existing primitives (Goal, CompositeGoal,
    HierarchicalPlanner, DeliberativeAgent) into a parallel execution runtime.
    """

    def __init__(
        self,
        hierarchical_planner: HierarchicalPlanner,
        members: Optional[List[SwarmMember]] = None,
        max_parallelism: int = 4,
    ):
        if max_parallelism < 1:
            raise ValueError("max_parallelism must be >= 1")

        self.hierarchical_planner = hierarchical_planner
        self.members: List[SwarmMember] = members or []
        self.max_parallelism = max_parallelism
        self._latest_snapshot: Optional[SwarmDashboardSnapshot] = None

    def register_member(self, member: SwarmMember) -> None:
        """Add a swarm member."""
        self.members.append(member)

    def get_dashboard_snapshot(self) -> Optional[SwarmDashboardSnapshot]:
        """Get the most recent execution snapshot for GUI consumers."""
        return self._latest_snapshot

    def build_goal_dag(
        self,
        goals: Sequence[Goal | CompositeGoal],
    ) -> Dict[str, GoalNode]:
        """
        Build an acyclic goal graph.

        Includes:
        - Explicit Goal dependencies
        - CompositeGoal expansion (`require_all=True`)
        - Decomposed subgoals for goals with registered decomposers
        """
        nodes: Dict[str, GoalNode] = {}
        visited: Set[str] = set()
        decomposition_cache: Dict[str, List[Goal]] = {}

        def ensure_node(goal: Goal) -> None:
            if goal.id not in nodes:
                nodes[goal.id] = GoalNode(goal=goal)

        def add_edge(dep: Goal, target: Goal) -> None:
            ensure_node(dep)
            ensure_node(target)
            nodes[target.id].dependencies.add(dep.id)
            nodes[dep.id].dependents.add(target.id)

        def visit(goal: Goal) -> None:
            ensure_node(goal)

            for dep in goal.dependencies:
                add_edge(dep, goal)
                if dep.id not in visited:
                    visit(dep)

            subgoals = self._decompose_goal(goal, decomposition_cache)
            previous: Optional[Goal] = None
            for subgoal in subgoals:
                add_edge(subgoal, goal)
                if previous is not None:
                    # Match HierarchicalPlanner sequence semantics.
                    add_edge(previous, subgoal)
                if subgoal.id not in visited:
                    visit(subgoal)
                previous = subgoal

            visited.add(goal.id)

        for top_level in goals:
            if isinstance(top_level, CompositeGoal):
                composite_goal = top_level.to_goal()
                visit(composite_goal)
                if top_level.require_all:
                    for subgoal in top_level.subgoals:
                        add_edge(subgoal, composite_goal)
                        if subgoal.id not in visited:
                            visit(subgoal)
            else:
                visit(top_level)

        self._validate_acyclic(nodes)
        return nodes

    async def execute(
        self,
        goals: Sequence[Goal | CompositeGoal],
        initial_state: WorldState,
    ) -> SwarmRunResult:
        """
        Execute goals in dependency order with parallel branch scheduling.
        """
        if not self.members:
            raise ValueError("SwarmManager requires at least one SwarmMember")

        run_id = f"swarm-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        nodes = self.build_goal_dag(goals)

        status_by_goal = {goal_id: "pending" for goal_id in nodes}
        assigned_agents: Dict[str, str] = {}
        node_results: Dict[str, SwarmNodeResult] = {}
        node_states: Dict[str, WorldState] = {}
        completion_order: List[str] = []

        assignment_counts = {member.name: 0 for member in self.members}
        running: Dict[asyncio.Task, Tuple[str, datetime, str]] = {}

        self._update_snapshot(run_id, nodes, status_by_goal, assigned_agents)

        while True:
            pending = [goal_id for goal_id, status in status_by_goal.items() if status == "pending"]

            if not pending and not running:
                break

            ready = [
                goal_id
                for goal_id in pending
                if all(status_by_goal[dep] == "success" for dep in nodes[goal_id].dependencies)
            ]

            for goal_id in ready:
                if len(running) >= self.max_parallelism:
                    break

                member = self._select_member(nodes[goal_id].goal, assignment_counts)
                assignment_counts[member.name] += 1
                assigned_agents[goal_id] = member.name
                status_by_goal[goal_id] = "running"

                state_for_goal = self._state_for_goal(
                    goal_id=goal_id,
                    nodes=nodes,
                    node_states=node_states,
                    initial_state=initial_state,
                )

                start = datetime.now()
                task = asyncio.create_task(member.agent.achieve(nodes[goal_id].goal, state_for_goal))
                running[task] = (goal_id, start, member.name)

            self._update_snapshot(run_id, nodes, status_by_goal, assigned_agents)

            if not running:
                break

            done, _ = await asyncio.wait(
                list(running.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                goal_id, started_at, member_name = running.pop(task)
                finished_at = datetime.now()
                duration_ms = (finished_at - started_at).total_seconds() * 1000.0
                goal = nodes[goal_id].goal

                try:
                    result: AgentResult = task.result()
                except Exception as exc:
                    status_by_goal[goal_id] = "failed"
                    node_results[goal_id] = SwarmNodeResult(
                        goal_id=goal.id,
                        goal_description=goal.description,
                        status="failed",
                        agent_name=member_name,
                        message=f"Unhandled execution error: {exc}",
                        started_at=started_at,
                        finished_at=finished_at,
                        duration_ms=duration_ms,
                    )
                    continue

                if result.success:
                    status_by_goal[goal_id] = "success"
                    if result.state is not None:
                        node_states[goal_id] = result.state
                    completion_order.append(goal_id)
                elif result.needs_input:
                    status_by_goal[goal_id] = "needs_input"
                else:
                    status_by_goal[goal_id] = "failed"

                node_results[goal_id] = SwarmNodeResult(
                    goal_id=goal.id,
                    goal_description=goal.description,
                    status=status_by_goal[goal_id],
                    agent_name=member_name,
                    message=result.message,
                    concerns=list(result.concerns),
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    state=result.state,
                )

            self._update_snapshot(run_id, nodes, status_by_goal, assigned_agents)

        skipped_nodes: List[str] = []
        for goal_id, status in status_by_goal.items():
            if status != "pending":
                continue
            dependency_statuses = {status_by_goal[d] for d in nodes[goal_id].dependencies}
            if "failed" in dependency_statuses or "needs_input" in dependency_statuses:
                status_by_goal[goal_id] = "skipped"
                skipped_nodes.append(goal_id)
                node_results[goal_id] = SwarmNodeResult(
                    goal_id=goal_id,
                    goal_description=nodes[goal_id].goal.description,
                    status="skipped",
                    agent_name="",
                    message="Skipped because a dependency failed or needs input.",
                )
            else:
                status_by_goal[goal_id] = "blocked"
                node_results[goal_id] = SwarmNodeResult(
                    goal_id=goal_id,
                    goal_description=nodes[goal_id].goal.description,
                    status="blocked",
                    agent_name="",
                    message="Blocked with no ready schedulable dependencies.",
                )

        successful_states = [
            node_states[goal_id]
            for goal_id in completion_order
            if goal_id in node_states
        ]
        final_state = self._merge_states([initial_state, *successful_states])

        failed_nodes = [
            goal_id
            for goal_id, status in status_by_goal.items()
            if status in {"failed", "needs_input"}
        ]

        if failed_nodes:
            overall_status = "partial_failure"
        elif any(status in {"blocked", "skipped"} for status in status_by_goal.values()):
            overall_status = "blocked"
        else:
            overall_status = "success"

        self._update_snapshot(run_id, nodes, status_by_goal, assigned_agents)

        return SwarmRunResult(
            run_id=run_id,
            status=overall_status,
            node_results=node_results,
            final_state=final_state,
            failed_nodes=failed_nodes,
            skipped_nodes=skipped_nodes,
        )

    def _decompose_goal(
        self,
        goal: Goal,
        cache: Dict[str, List[Goal]],
    ) -> List[Goal]:
        """Resolve subgoals from registered decomposers."""
        if goal.id in cache:
            return cache[goal.id]

        goal_type = goal.metadata.get("type", "")
        decomposer = self.hierarchical_planner.decomposers.get(goal_type)
        if decomposer is None:
            cache[goal.id] = []
            return []

        subgoals = decomposer(goal)
        cache[goal.id] = subgoals
        return subgoals

    def _validate_acyclic(self, nodes: Dict[str, GoalNode]) -> None:
        """Raise if graph contains cycles."""
        indegree = {goal_id: len(node.dependencies) for goal_id, node in nodes.items()}
        ready = [goal_id for goal_id, degree in indegree.items() if degree == 0]
        visited = 0

        while ready:
            current = ready.pop()
            visited += 1
            for dependent in nodes[current].dependents:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if visited != len(nodes):
            raise ValueError("Goal graph contains a cycle and cannot be scheduled as a DAG")

    def _select_member(
        self,
        goal: Goal,
        assignment_counts: Dict[str, int],
    ) -> SwarmMember:
        """Select the least-loaded capable member for a goal."""
        capable = [member for member in self.members if member.can_handle(goal)]
        candidates = capable or self.members
        return min(candidates, key=lambda member: (assignment_counts.get(member.name, 0), member.name))

    def _state_for_goal(
        self,
        goal_id: str,
        nodes: Dict[str, GoalNode],
        node_states: Dict[str, WorldState],
        initial_state: WorldState,
    ) -> WorldState:
        """Compose dependency outputs into the input state for a goal."""
        dep_states = [
            node_states[dep_id]
            for dep_id in sorted(nodes[goal_id].dependencies)
            if dep_id in node_states
        ]
        return self._merge_states([initial_state, *dep_states])

    def _merge_states(self, states: Sequence[WorldState]) -> WorldState:
        """Merge states with later states winning on key conflicts."""
        if not states:
            return WorldState()

        merged = states[0].copy()
        for state in states[1:]:
            merged.facts.update(state.facts)
            merged.files.update(state.files)
            merged.test_results.update(state.test_results)
            merged.metadata.update(state.metadata)

        return merged

    def _update_snapshot(
        self,
        run_id: str,
        nodes: Dict[str, GoalNode],
        status_by_goal: Dict[str, str],
        assigned_agents: Dict[str, str],
    ) -> None:
        dashboard_nodes = [
            SwarmDashboardNode(
                goal_id=goal_id,
                description=node.goal.description,
                status=status_by_goal.get(goal_id, "pending"),
                dependencies=sorted(node.dependencies),
                assigned_agent=assigned_agents.get(goal_id, ""),
            )
            for goal_id, node in sorted(nodes.items(), key=lambda item: item[0])
        ]

        self._latest_snapshot = SwarmDashboardSnapshot(
            run_id=run_id,
            timestamp=datetime.now(),
            total_nodes=len(nodes),
            pending_nodes=sum(1 for s in status_by_goal.values() if s == "pending"),
            running_nodes=sum(1 for s in status_by_goal.values() if s == "running"),
            completed_nodes=sum(1 for s in status_by_goal.values() if s == "success"),
            failed_nodes=sum(1 for s in status_by_goal.values() if s in {"failed", "needs_input"}),
            nodes=dashboard_nodes,
        )
