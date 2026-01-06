"""
Verification system for the Deliberative Agent.

Provides multi-level verification against specifications, not just string matching.
This is a key differentiator from "Ralph Wiggum" style approaches.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from .core import Confidence, ConfidenceSource, WorldState

if TYPE_CHECKING:
    pass


@dataclass
class CheckResult:
    """Result of running a verification check."""

    passed: bool
    confidence: Confidence
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"CheckResult({status}, {self.confidence}, {self.message!r})"


class Check(ABC):
    """
    Base class for verification checks.

    Checks are the atomic units of verification. They can be:
    - Static analysis (type checking, linting)
    - Dynamic testing (unit tests, integration tests)
    - Semantic verification (LLM-based property checking)
    """

    @abstractmethod
    async def run(self, state: WorldState) -> CheckResult:
        """
        Run the check against the given world state.

        Args:
            state: Current world state

        Returns:
            CheckResult with pass/fail status and confidence
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable name for the check."""
        return self.__class__.__name__


class TypeCheck(Check):
    """
    Runs type checking on specified files.

    Supports multiple type checkers (pyright, mypy, etc.).
    """

    def __init__(
        self,
        paths: List[Path],
        tool: str = "pyright",
        strict: bool = False
    ):
        """
        Initialize the type check.

        Args:
            paths: Files or directories to type check
            tool: Type checker to use (pyright, mypy)
            strict: Whether to use strict mode
        """
        self.paths = paths
        self.tool = tool
        self.strict = strict

    async def run(self, state: WorldState) -> CheckResult:
        """Run type checking."""
        try:
            cmd = [self.tool]
            if self.strict and self.tool == "mypy":
                cmd.append("--strict")

            cmd.extend(str(p) for p in self.paths)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            passed = process.returncode == 0
            output = stdout.decode() + stderr.decode()

            return CheckResult(
                passed=passed,
                confidence=Confidence(
                    0.95 if passed else 0.1,
                    ConfidenceSource.VERIFICATION,
                    evidence=[f"Type check with {self.tool}"]
                ),
                message="Type check passed" if passed else f"Type errors found",
                details={
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "returncode": process.returncode,
                    "tool": self.tool
                }
            )
        except FileNotFoundError:
            return CheckResult(
                passed=False,
                confidence=Confidence(0.0, ConfidenceSource.VERIFICATION),
                message=f"Type checker '{self.tool}' not found",
                details={"error": "tool_not_found"}
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                confidence=Confidence(0.0, ConfidenceSource.VERIFICATION),
                message=f"Type check failed to run: {e}",
                details={"error": str(e)}
            )

    @property
    def name(self) -> str:
        return f"TypeCheck({self.tool})"


class TestCheck(Check):
    """
    Runs a test suite and reports results.

    Can run any test command (pytest, unittest, npm test, etc.).
    """

    def __init__(
        self,
        test_command: List[str],
        timeout: float = 300.0,
        working_dir: Optional[Path] = None
    ):
        """
        Initialize the test check.

        Args:
            test_command: Command to run tests (e.g., ["pytest", "-v"])
            timeout: Maximum time to wait for tests
            working_dir: Directory to run tests from
        """
        self.test_command = test_command
        self.timeout = timeout
        self.working_dir = working_dir

    async def run(self, state: WorldState) -> CheckResult:
        """Run the test suite."""
        try:
            process = await asyncio.create_subprocess_exec(
                *self.test_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return CheckResult(
                    passed=False,
                    confidence=Confidence(0.0, ConfidenceSource.VERIFICATION),
                    message=f"Tests timed out after {self.timeout}s",
                    details={"error": "timeout"}
                )

            passed = process.returncode == 0

            return CheckResult(
                passed=passed,
                confidence=Confidence(
                    0.9 if passed else 0.05,
                    ConfidenceSource.VERIFICATION,
                    evidence=[f"Test suite: {' '.join(self.test_command)}"]
                ),
                message="All tests passed" if passed else "Test failures",
                details={
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "returncode": process.returncode,
                    "command": self.test_command
                }
            )
        except FileNotFoundError:
            return CheckResult(
                passed=False,
                confidence=Confidence(0.0, ConfidenceSource.VERIFICATION),
                message=f"Test command '{self.test_command[0]}' not found",
                details={"error": "command_not_found"}
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                confidence=Confidence(0.0, ConfidenceSource.VERIFICATION),
                message=f"Tests failed to run: {e}",
                details={"error": str(e)}
            )

    @property
    def name(self) -> str:
        return f"TestCheck({self.test_command[0]})"


class LintCheck(Check):
    """
    Runs a linter on the codebase.

    Supports various linters (ruff, pylint, eslint, etc.).
    """

    def __init__(
        self,
        paths: List[Path],
        tool: str = "ruff",
        config: Optional[Path] = None
    ):
        """
        Initialize the lint check.

        Args:
            paths: Files or directories to lint
            tool: Linter to use
            config: Optional config file
        """
        self.paths = paths
        self.tool = tool
        self.config = config

    async def run(self, state: WorldState) -> CheckResult:
        """Run the linter."""
        try:
            cmd = [self.tool, "check"]
            if self.config:
                cmd.extend(["--config", str(self.config)])
            cmd.extend(str(p) for p in self.paths)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            passed = process.returncode == 0

            return CheckResult(
                passed=passed,
                confidence=Confidence(
                    0.85 if passed else 0.2,
                    ConfidenceSource.VERIFICATION,
                    evidence=[f"Lint check with {self.tool}"]
                ),
                message="No lint issues" if passed else "Lint issues found",
                details={
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "tool": self.tool
                }
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                confidence=Confidence(0.0, ConfidenceSource.VERIFICATION),
                message=f"Lint check failed: {e}",
                details={"error": str(e)}
            )


class FileExistsCheck(Check):
    """Checks that required files exist."""

    def __init__(self, paths: List[Path]):
        """
        Initialize the file existence check.

        Args:
            paths: Paths that must exist
        """
        self.paths = paths

    async def run(self, state: WorldState) -> CheckResult:
        """Check that all required files exist."""
        missing = [p for p in self.paths if not p.exists()]

        if missing:
            return CheckResult(
                passed=False,
                confidence=Confidence(1.0, ConfidenceSource.OBSERVATION),
                message=f"Missing files: {missing}",
                details={"missing": [str(p) for p in missing]}
            )

        return CheckResult(
            passed=True,
            confidence=Confidence(1.0, ConfidenceSource.OBSERVATION),
            message="All required files exist",
            details={"checked": [str(p) for p in self.paths]}
        )


class PredicateCheck(Check):
    """
    Checks an arbitrary predicate against the world state.

    Useful for custom verification logic.
    """

    def __init__(
        self,
        predicate: Callable[[WorldState], bool],
        description: str,
        confidence_if_true: float = 0.9,
        confidence_if_false: float = 0.1
    ):
        """
        Initialize the predicate check.

        Args:
            predicate: Function that returns True if check passes
            description: Human-readable description
            confidence_if_true: Confidence when predicate is True
            confidence_if_false: Confidence when predicate is False
        """
        self.predicate = predicate
        self.description = description
        self.confidence_if_true = confidence_if_true
        self.confidence_if_false = confidence_if_false

    async def run(self, state: WorldState) -> CheckResult:
        """Run the predicate check."""
        try:
            result = self.predicate(state)
            return CheckResult(
                passed=result,
                confidence=Confidence(
                    self.confidence_if_true if result else self.confidence_if_false,
                    ConfidenceSource.VERIFICATION
                ),
                message=self.description + (" satisfied" if result else " not satisfied")
            )
        except Exception as e:
            return CheckResult(
                passed=False,
                confidence=Confidence(0.0, ConfidenceSource.VERIFICATION),
                message=f"Predicate check failed: {e}",
                details={"error": str(e)}
            )


class SemanticCheck(Check):
    """
    Uses LLM to verify semantic properties.

    This is a key differentiator from "Ralph Wiggum" style string matching.
    Instead of checking for magic completion strings, we verify that
    semantic properties actually hold.

    NOTE: This is a placeholder - real implementation would integrate
    with an LLM provider.
    """

    def __init__(
        self,
        property_description: str,
        context_files: List[Path],
        llm_provider: Optional[Any] = None
    ):
        """
        Initialize the semantic check.

        Args:
            property_description: Description of the property to verify
            context_files: Files to provide as context
            llm_provider: Optional LLM provider for verification
        """
        self.property = property_description
        self.context_files = context_files
        self.llm_provider = llm_provider

    async def run(self, state: WorldState) -> CheckResult:
        """
        Run semantic verification.

        In a real implementation, this would:
        1. Gather context from files
        2. Construct a verification prompt
        3. Query an LLM with structured output
        4. Parse and validate the response
        """
        # Placeholder - in production, this calls an LLM
        if self.llm_provider is None:
            return CheckResult(
                passed=True,
                confidence=Confidence(
                    0.5,
                    ConfidenceSource.ASSUMPTION,
                    evidence=["SemanticCheck: LLM provider not configured"]
                ),
                message="Semantic check skipped (no LLM provider)",
                details={"property": self.property, "skipped": True}
            )

        # Real implementation would be:
        # context = self._gather_context()
        # prompt = self._build_verification_prompt(context)
        # response = await self.llm_provider.complete(prompt)
        # return self._parse_verification_response(response)

        return CheckResult(
            passed=True,
            confidence=Confidence(0.5, ConfidenceSource.INFERENCE),
            message="Semantic verification placeholder"
        )

    def _gather_context(self) -> str:
        """Gather context from files for the LLM."""
        context_parts = []
        for path in self.context_files:
            if path.exists():
                try:
                    content = path.read_text()
                    context_parts.append(f"=== {path} ===\n{content}")
                except Exception:
                    pass
        return "\n\n".join(context_parts)


@dataclass
class VerificationResult:
    """Result of running a full verification plan."""

    satisfied: bool
    confidence: Confidence
    check_results: List[CheckResult]
    failures: List[CheckResult]

    @property
    def all_passed(self) -> bool:
        """Check if all individual checks passed."""
        return all(r.passed for r in self.check_results)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        passed = sum(1 for r in self.check_results if r.passed)
        total = len(self.check_results)
        return f"{passed}/{total} checks passed, confidence: {float(self.confidence):.2f}"


@dataclass
class VerificationPlan:
    """
    A plan for verifying that a goal has been achieved.

    Combines multiple checks and requires a minimum confidence level.
    """

    checks: List[Check]
    required_confidence: float = 0.8
    require_all_pass: bool = True

    async def verify(self, state: WorldState) -> VerificationResult:
        """
        Execute all checks and aggregate results.

        Args:
            state: Current world state to verify against

        Returns:
            VerificationResult with aggregated outcomes
        """
        if not self.checks:
            return VerificationResult(
                satisfied=True,
                confidence=Confidence(1.0, ConfidenceSource.ASSUMPTION),
                check_results=[],
                failures=[]
            )

        # Run all checks concurrently
        results = await asyncio.gather(*[check.run(state) for check in self.checks])

        # Aggregate results
        failures = [r for r in results if not r.passed]
        all_passed = len(failures) == 0

        # Compute aggregate confidence
        if results:
            min_confidence = min(r.confidence for r in results)
        else:
            min_confidence = Confidence(1.0, ConfidenceSource.ASSUMPTION)

        # Determine if verification is satisfied
        if self.require_all_pass:
            satisfied = all_passed and float(min_confidence) >= self.required_confidence
        else:
            # Allow some failures if confidence is high enough
            passed_ratio = (len(results) - len(failures)) / len(results)
            satisfied = (
                passed_ratio >= 0.8 and
                float(min_confidence) >= self.required_confidence
            )

        return VerificationResult(
            satisfied=satisfied,
            confidence=min_confidence,
            check_results=list(results),
            failures=failures
        )

    def add_check(self, check: Check) -> VerificationPlan:
        """Add a check to the plan (returns self for chaining)."""
        self.checks.append(check)
        return self

    def __repr__(self) -> str:
        check_names = [c.name for c in self.checks]
        return f"VerificationPlan({check_names}, required={self.required_confidence})"
