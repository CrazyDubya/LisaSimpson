# Next Steps (after this testing pass)

Suggestions derived from making the test suite pass and from the value benchmark. Use this as a backlog, not a commitment.

---

## 1. Verification value (show semantic checks beat string matching)

**Learned:** Value benchmark shows planning clearly helps (100% vs ~48% success, fewer steps). Verification is not yet demonstrated.

**Suggestions:**

- Add a **value scenario** where the executor can “claim” success but the goal has a real `VerificationPlan` (e.g. `TestCheck` on a file). If the executor didn’t actually run tests, verification fails and the run is not considered success.
- Optionally a **baseline** that “succeeds” on a magic string; compare to deliberative + verification so we report “verification caught N bad outcomes”.
- Keep scenarios fast: e.g. a small test file that must exist and pass, or a mock check that fails when a fact is missing.

---

## 2. Memory / learning value (show second run improves)

**Learned:** Memory and lessons are wired in; we don’t yet measure “run same (or similar) scenario twice and see improvement”.

**Suggestions:**

- **Re-run value scenario:** Run the same deliberative scenario twice with the same agent (shared memory). Compare steps, cost, or time on run 2 vs run 1 (expect same or better).
- **Lesson injection test:** In a test, add a lesson that says “action X often fails when Y”; check that the agent’s confidence or plan changes when that lesson is present.
- **Channel policy:** Add a small test or scenario that uses different memory channels (e.g. “recent” vs “project”) and checks that retrieval differs by channel.

---

## 3. Harder / expert value scenarios

**Learned:** Current value scenarios are 4–5 steps, linear or simple DAGs. Baseline still wins ~48% of the time with 12 runs/scenario.

**Suggestions:**

- Add **one expert scenario**: more steps (e.g. 8–10), branching (e.g. “run tests on both frontend and backend then merge”), or a rollback path. Measure baseline success rate vs deliberative.
- Optionally **cost variance:** some actions much cheaper than others so that “wrong order” is obviously more expensive even when the baseline eventually succeeds.

---

## 4. CI and stability

**Learned:** Full pytest run is 79 passed, 4 skipped; value benchmark is deterministic with mock executor.

**Suggestions:**

- Add **CI** (e.g. GitHub Actions) that runs:
  - `uv run pytest tests/ -v --ignore=tests/test_llm_providers.py` (no API keys)
  - `uv run python run_value_benchmark.py --baseline-runs 5 --quiet`
- Optionally **pin** or gate on: “value benchmark baseline success rate &lt; 60%” and “deliberative success rate = 100%” so regressions (e.g. planner or baseline behavior) are caught.
- Consider **marking** LLM tests (e.g. `@pytest.mark.llm`) and running them in CI only when an API key is present (or in a separate scheduled job).

---

## 5. Value benchmark with real LLM

**Learned:** Value benchmark is currently run with `MockExecutor` by default (fast, no API). With `--llm`, same scenarios use the real executor; we didn’t systematically compare deliberative vs baseline *with* LLM.

**Suggestions:**

- Run **deliberative vs baseline with `--llm`** (e.g. OpenRouter) on 2–3 scenarios with a small `--baseline-runs` (e.g. 3) and record success rate and steps in the session log or a one-off report.
- If the LLM sometimes “fails” an action (e.g. timeout, bad response), check that the **agent** handles it (e.g. execution failure, no plan found) and that the **value report** still compares fairly (e.g. count only runs where the executor was actually invoked).

---

## 6. Docs and discovery

**Learned:** TESTING.md and session_log are the main “what to run” and “what we learned” docs.

**Suggestions:**

- In **README**, add a one-line “Run value benchmark” under a “Testing” or “Benchmarks” subsection that links to TESTING.md.
- In **TESTING.md**, add a short “Known limitations” line: e.g. value benchmark baseline is random-order only (no “single monolithic prompt” baseline yet).
- If you add more entry points (e.g. `run_verification_benchmark.py`), list them in TESTING.md and in session_log when they’re first run.

---

## 7. Small cleanups (optional)

- **Dataclass names:** Pytest warns about `TestProblem`, `TestResult`, `TestSummary` (look like test classes). Renaming to e.g. `Scenario`, `ScenarioResult`, `ScenarioSummary` would remove the warnings; low priority.
- **test_llm_providers.py:** Still ignored in the main “full suite” command; either integrate (e.g. same `client` fixture and skip-if-no-key) or document why it’s excluded.

---

## Summary table

| Area              | Priority | Effort | Notes                                      |
|-------------------|----------|--------|--------------------------------------------|
| Verification value| High     | Medium | One scenario + optional baseline          |
| Memory value      | Medium   | Medium | Re-run scenario + lesson injection         |
| Harder scenarios  | Medium   | Low    | One 8–10 step or branching scenario       |
| CI                | High     | Low    | pytest + value benchmark                   |
| Value + LLM       | Medium   | Low    | Few runs with `--llm`, document results   |
| README/TESTING    | Low      | Low    | One-line link, limitations                |
| Dataclass renames | Low      | Low    | Removes pytest warnings                    |

Use this as a living list: after doing a chunk of work, update or tick items and add new ones to the session log.
