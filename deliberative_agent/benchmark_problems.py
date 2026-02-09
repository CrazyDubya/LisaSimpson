"""
Test Problems for LLM Benchmark Testing.

Provides a comprehensive set of medium and hard problems to test
LLM capabilities including:
- Reasoning and logic
- Code generation and debugging
- Mathematical problem solving
- Multi-step planning
- Edge case handling
- Knowledge integration
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class Difficulty(str, Enum):
    """Problem difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Category(str, Enum):
    """Problem categories."""
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    CODE_DEBUGGING = "code_debugging"
    MATHEMATICS = "mathematics"
    LOGIC_PUZZLE = "logic_puzzle"
    PLANNING = "planning"
    KNOWLEDGE = "knowledge"
    MULTI_STEP = "multi_step"
    EDGE_CASES = "edge_cases"
    SEMANTIC = "semantic"


@dataclass
class TestProblem:
    """A single test problem for LLM evaluation."""
    id: str
    name: str
    description: str
    prompt: str
    difficulty: Difficulty
    category: Category
    expected_patterns: List[str] = field(default_factory=list)
    validation_fn: Optional[Callable[[str], bool]] = None
    hints: List[str] = field(default_factory=list)
    time_limit_seconds: float = 120.0
    tags: List[str] = field(default_factory=list)
    reference_solution: Optional[str] = None

    def validate_response(self, response: str) -> bool:
        """
        Validate if the response is correct.

        Uses pattern matching and/or custom validation function.
        """
        if self.validation_fn:
            try:
                return self.validation_fn(response)
            except Exception:
                return False

        if self.expected_patterns:
            response_lower = response.lower()
            return any(
                re.search(pattern, response_lower, re.IGNORECASE | re.DOTALL)
                for pattern in self.expected_patterns
            )

        return True  # No validation criteria = always pass


@dataclass
class TestResult:
    """Result of running a test problem."""
    problem_id: str
    provider: str
    model: str
    response: str
    is_correct: bool
    latency_ms: float
    tokens_used: int
    error: Optional[str] = None


# =============================================================================
# MEDIUM DIFFICULTY PROBLEMS
# =============================================================================

MEDIUM_PROBLEMS: List[TestProblem] = [
    # 1. Reasoning - Syllogism
    TestProblem(
        id="medium_reasoning_01",
        name="Syllogism Reasoning",
        description="Test basic logical reasoning with syllogisms",
        prompt="""All programmers are logical thinkers.
Some logical thinkers are mathematicians.
Maria is a programmer.

Based on these statements, which of the following MUST be true?
A) Maria is a mathematician
B) Maria is a logical thinker
C) Some programmers are mathematicians
D) All logical thinkers are programmers

Provide your answer as just the letter (A, B, C, or D) followed by a brief explanation.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.REASONING,
        expected_patterns=[r"\bB\b", r"logical thinker"],
        tags=["logic", "syllogism"]
    ),

    # 2. Code Generation - FizzBuzz Variant
    TestProblem(
        id="medium_code_01",
        name="FizzBuzz Variant",
        description="Generate code for FizzBuzz with twist",
        prompt="""Write a Python function called `fizz_buzz_plus(n)` that:
- For numbers 1 to n:
  - Prints "Fizz" if divisible by 3
  - Prints "Buzz" if divisible by 5
  - Prints "FizzBuzz" if divisible by both 3 and 5
  - Prints "Prime" if the number is prime AND not divisible by 3 or 5
  - Otherwise prints the number

Return the count of prime numbers found.

Provide only the Python function code.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.CODE_GENERATION,
        expected_patterns=[r"def fizz_buzz_plus", r"prime", r"return"],
        validation_fn=lambda r: "def fizz_buzz_plus" in r and "prime" in r.lower(),
        tags=["python", "algorithms"]
    ),

    # 3. Mathematics - Probability
    TestProblem(
        id="medium_math_01",
        name="Probability Problem",
        description="Calculate probability with conditional events",
        prompt="""A bag contains 5 red balls, 3 blue balls, and 2 green balls.
You draw two balls WITHOUT replacement.

What is the probability that:
1. Both balls are the same color?
2. At least one ball is red?

Show your work and provide final answers as fractions or decimals.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.MATHEMATICS,
        expected_patterns=[r"14/45|0\.31", r"7/9|0\.78"],
        tags=["probability", "combinatorics"]
    ),

    # 4. Logic Puzzle - Knights and Knaves
    TestProblem(
        id="medium_logic_01",
        name="Knights and Knaves",
        description="Solve a classic knights and knaves puzzle",
        prompt="""On an island, knights always tell the truth and knaves always lie.

You meet two people, A and B.
A says: "At least one of us is a knave."

Determine whether A and B are knights or knaves. Explain your reasoning.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.LOGIC_PUZZLE,
        expected_patterns=[r"A.*knight", r"B.*knave"],
        tags=["logic", "puzzle"]
    ),

    # 5. Code Debugging
    TestProblem(
        id="medium_debug_01",
        name="Bug Fix - Off-by-One",
        description="Find and fix an off-by-one error",
        prompt="""The following Python code should find all pairs of numbers in a list that sum to a target.
However, it has bugs. Find and fix them:

```python
def find_pairs(numbers, target):
    pairs = []
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if numbers[i] + numbers[j] == target:
                pairs.append((numbers[i], numbers[j]))
    return pairs

# Should return [(1,4), (2,3)] for [1,2,3,4] and target 5
# But currently returns duplicates and self-pairs
```

Provide the corrected code.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.CODE_DEBUGGING,
        expected_patterns=[r"j.*=.*i.*\+.*1|range\(i\s*\+\s*1", r"for j in range"],
        tags=["python", "debugging"]
    ),

    # 6. Planning - Task Dependency
    TestProblem(
        id="medium_planning_01",
        name="Task Scheduling",
        description="Solve a task scheduling problem with dependencies",
        prompt="""Given these tasks and their dependencies, provide a valid execution order:

Task A: No dependencies, takes 2 hours
Task B: Depends on A, takes 3 hours
Task C: Depends on A, takes 1 hour
Task D: Depends on B and C, takes 2 hours
Task E: Depends on C, takes 4 hours
Task F: Depends on D and E, takes 1 hour

1. What is a valid execution order?
2. What is the minimum total time to complete all tasks if tasks can run in parallel when dependencies allow?

Show your reasoning.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.PLANNING,
        expected_patterns=[r"A.*B.*D|A.*C.*D", r"10|11|12"],
        tags=["scheduling", "graphs"]
    ),

    # 7. Multi-step Reasoning
    TestProblem(
        id="medium_multi_01",
        name="River Crossing",
        description="Solve a multi-step river crossing puzzle",
        prompt="""A farmer needs to cross a river with a wolf, a goat, and a cabbage.
The boat can only carry the farmer and one item at a time.
If left alone:
- The wolf will eat the goat
- The goat will eat the cabbage

Provide the minimum number of crossings and list each step.
Format: "Trip N: Farmer takes X (leaving Y, Z on original side)"
""",
        difficulty=Difficulty.MEDIUM,
        category=Category.MULTI_STEP,
        expected_patterns=[r"7|seven", r"goat"],
        tags=["puzzle", "planning"]
    ),

    # 8. Edge Cases
    TestProblem(
        id="medium_edge_01",
        name="Edge Case Handling",
        description="Write code that handles edge cases properly",
        prompt="""Write a Python function `safe_divide(a, b, default=0)` that:
- Returns a/b if b is not zero
- Returns default if b is zero
- Handles cases where a or b might be None, strings, or other non-numeric types
- Returns the default value for any invalid input

The function should NEVER raise an exception.

Provide the complete function code.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.EDGE_CASES,
        expected_patterns=[r"def safe_divide", r"try|except|isinstance"],
        tags=["python", "error-handling"]
    ),

    # 9. String Manipulation
    TestProblem(
        id="medium_string_01",
        name="Palindrome Permutation",
        description="Check if any permutation of string can form palindrome",
        prompt="""Write a Python function `can_form_palindrome(s)` that:
- Returns True if any permutation of the string s could form a palindrome
- Returns False otherwise
- Ignores spaces and is case-insensitive

Examples:
- "Tact Coa" -> True (can form "taco cat" or "atco cta")
- "hello" -> False
- "aab" -> True (can form "aba")

Provide the function with O(n) time complexity.""",
        difficulty=Difficulty.MEDIUM,
        category=Category.CODE_GENERATION,
        expected_patterns=[r"def can_form_palindrome", r"count|Counter|dict"],
        tags=["python", "strings", "algorithms"]
    ),

    # 10. Semantic Understanding
    TestProblem(
        id="medium_semantic_01",
        name="Code Intent Analysis",
        description="Understand what code is trying to accomplish",
        prompt="""Analyze this code and explain what it's trying to accomplish.
What edge cases might it miss?

```python
def mystery(data):
    if not data:
        return []
    result = [data[0]]
    for item in data[1:]:
        if item != result[-1]:
            result.append(item)
    return result
```

Provide:
1. A one-sentence description of what this function does
2. A better name for the function
3. Two edge cases it might not handle well""",
        difficulty=Difficulty.MEDIUM,
        category=Category.SEMANTIC,
        expected_patterns=[r"consecutive|duplicate|adjacent", r"dedupe|unique|compress"],
        tags=["code-analysis", "semantic"]
    ),
]


# =============================================================================
# HARD DIFFICULTY PROBLEMS
# =============================================================================

HARD_PROBLEMS: List[TestProblem] = [
    # 1. Complex Reasoning Chain
    TestProblem(
        id="hard_reasoning_01",
        name="Complex Deduction",
        description="Multi-step logical deduction with constraints",
        prompt="""Five houses in a row are painted different colors. In each house lives a person with a different nationality. Each person drinks a different beverage, smokes a different brand of cigar, and owns a different pet.

Clues:
1. The Brit lives in the red house.
2. The Swede keeps dogs.
3. The Dane drinks tea.
4. The green house is immediately to the left of the white house.
5. The owner of the green house drinks coffee.
6. The person who smokes Pall Mall keeps birds.
7. The owner of the yellow house smokes Dunhill.
8. The man living in the center house drinks milk.
9. The Norwegian lives in the first house.
10. The man who smokes Blend lives next to the one who keeps cats.
11. The man who keeps horses lives next to the man who smokes Dunhill.
12. The man who smokes Blue Master drinks beer.
13. The German smokes Prince.
14. The Norwegian lives next to the blue house.
15. The man who smokes Blend has a neighbor who drinks water.

Who owns the fish?

Provide your answer with key reasoning steps.""",
        difficulty=Difficulty.HARD,
        category=Category.REASONING,
        expected_patterns=[r"german", r"fish"],
        tags=["logic", "constraint-satisfaction"]
    ),

    # 2. Algorithm Design - LRU Cache
    TestProblem(
        id="hard_code_01",
        name="LRU Cache Implementation",
        description="Implement an LRU cache with O(1) operations",
        prompt="""Implement an LRU (Least Recently Used) cache in Python with these requirements:

1. `LRUCache(capacity)` - Initialize with positive capacity
2. `get(key)` - Return value if key exists, else return -1. Mark as recently used.
3. `put(key, value)` - Update/insert value. If at capacity, evict least recently used.

Both operations must be O(1) time complexity.

Provide the complete implementation using only standard library (no functools.lru_cache).
Include the class definition with all methods.""",
        difficulty=Difficulty.HARD,
        category=Category.CODE_GENERATION,
        expected_patterns=[r"class LRUCache", r"OrderedDict|dict.*list|doubly"],
        validation_fn=lambda r: "class LRUCache" in r and ("get" in r and "put" in r),
        tags=["python", "data-structures", "algorithms"]
    ),

    # 3. Advanced Mathematics
    TestProblem(
        id="hard_math_01",
        name="Dynamic Programming Optimization",
        description="Solve optimization with dynamic programming",
        prompt="""You have a rod of length n and a price table where price[i] is the price of a rod of length i+1.

Determine the maximum revenue obtainable by cutting up the rod and selling the pieces.

For n=8 with prices [1, 5, 8, 9, 10, 17, 17, 20]:
- price[0]=1 means rod of length 1 sells for $1
- price[1]=5 means rod of length 2 sells for $5
- etc.

1. What is the maximum revenue for n=8?
2. Provide the optimal cutting strategy.
3. Write Python code implementing the solution with memoization.

Show your work with the recurrence relation.""",
        difficulty=Difficulty.HARD,
        category=Category.MATHEMATICS,
        expected_patterns=[r"22|23|24", r"def.*cut|def.*rod"],
        tags=["dynamic-programming", "optimization"]
    ),

    # 4. Complex Logic Puzzle
    TestProblem(
        id="hard_logic_01",
        name="Truth Tellers Grid",
        description="Solve a complex grid-based logic puzzle",
        prompt="""A 4x4 grid contains 16 people. Each person is either a truth-teller (always tells truth) or a liar (always lies).

Each person makes a statement about the number of truth-tellers in their row and column (including themselves).

Grid positions and statements:
(1,1): "My row has exactly 2 truth-tellers"
(1,2): "My column has exactly 3 truth-tellers"
(1,3): "My row has exactly 3 truth-tellers"
(1,4): "My column has exactly 1 truth-teller"
(2,1): "My row has exactly 1 truth-teller"
(2,2): "My column has exactly 2 truth-tellers"
(2,3): "My row has exactly 2 truth-tellers"
(2,4): "My column has exactly 2 truth-tellers"
(3,1): "My row has exactly 3 truth-tellers"
(3,2): "My column has exactly 1 truth-teller"
(3,3): "My row has exactly 1 truth-teller"
(3,4): "My column has exactly 3 truth-tellers"
(4,1): "My row has exactly 2 truth-tellers"
(4,2): "My column has exactly 2 truth-tellers"
(4,3): "My row has exactly 2 truth-tellers"
(4,4): "My column has exactly 2 truth-tellers"

How many total truth-tellers are in the grid?
Which positions are truth-tellers?""",
        difficulty=Difficulty.HARD,
        category=Category.LOGIC_PUZZLE,
        expected_patterns=[r"8|9|10", r"\(.*\).*\(.*\)"],
        tags=["logic", "constraint-satisfaction", "grid"]
    ),

    # 5. Complex Debugging
    TestProblem(
        id="hard_debug_01",
        name="Concurrency Bug",
        description="Find race condition in concurrent code",
        prompt="""This Python code has a race condition. Find it and fix it:

```python
import threading

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        current = self.count
        # Simulate some processing
        self.count = current + 1

    def get_count(self):
        return self.count

def worker(counter, iterations):
    for _ in range(iterations):
        counter.increment()

# Main code
counter = Counter()
threads = []
for _ in range(10):
    t = threading.Thread(target=worker, args=(counter, 1000))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"Final count: {counter.get_count()}")  # Expected: 10000, but often less
```

1. Explain the race condition
2. Provide three different solutions (choose one to implement fully)
3. Implement your chosen solution""",
        difficulty=Difficulty.HARD,
        category=Category.CODE_DEBUGGING,
        expected_patterns=[r"Lock|lock|mutex|atomic", r"race condition|concurrent|thread"],
        tags=["python", "concurrency", "debugging"]
    ),

    # 6. Complex Planning
    TestProblem(
        id="hard_planning_01",
        name="Resource Allocation",
        description="Optimal resource allocation with constraints",
        prompt="""A factory has 3 machines (A, B, C) and needs to schedule 6 jobs.
Each job requires processing on specific machines in order:

Job 1: A(3h) -> B(2h) -> C(1h)
Job 2: A(2h) -> C(3h)
Job 3: B(4h) -> C(2h)
Job 4: A(1h) -> B(3h)
Job 5: B(2h) -> A(2h) -> C(2h)
Job 6: C(3h) -> A(1h) -> B(2h)

Rules:
- Each machine can only process one job at a time
- A job must complete each stage before starting the next
- Jobs can wait between stages

Find a schedule that minimizes the total completion time (makespan).

Provide:
1. The optimal (or near-optimal) schedule as a Gantt chart representation
2. The total makespan
3. Brief explanation of your approach""",
        difficulty=Difficulty.HARD,
        category=Category.PLANNING,
        expected_patterns=[r"gantt|schedule|timeline", r"hour|makespan"],
        tags=["scheduling", "optimization", "constraints"]
    ),

    # 7. Recursive Problem
    TestProblem(
        id="hard_multi_01",
        name="Expression Evaluation",
        description="Parse and evaluate nested expressions",
        prompt="""Write a Python function `evaluate(expr)` that evaluates arithmetic expressions with:
- Numbers (integers and floats)
- Operators: +, -, *, /, ^(power)
- Parentheses for grouping
- Standard operator precedence: ^ > * / > + -
- Right-to-left associativity for ^

Examples:
- evaluate("2 + 3 * 4") = 14
- evaluate("(2 + 3) * 4") = 20
- evaluate("2 ^ 3 ^ 2") = 512 (not 64, because right-to-left)
- evaluate("10 / 2 / 5") = 1 (left-to-right for division)

Do not use eval(). Implement a proper parser.
Handle invalid expressions by raising ValueError with descriptive message.""",
        difficulty=Difficulty.HARD,
        category=Category.CODE_GENERATION,
        expected_patterns=[r"def evaluate", r"parse|token|stack"],
        validation_fn=lambda r: "def evaluate" in r and "eval(" not in r,
        tags=["python", "parsing", "recursion"]
    ),

    # 8. System Design
    TestProblem(
        id="hard_design_01",
        name="Rate Limiter Design",
        description="Design a distributed rate limiter",
        prompt="""Design a rate limiter for an API with these requirements:

1. Limit: 100 requests per minute per user
2. Must work across multiple server instances
3. Should handle burst traffic gracefully
4. Provide sliding window accuracy (not fixed windows)
5. Must be thread-safe

Provide:
1. High-level architecture diagram (as ASCII/text)
2. Python implementation of the core rate limiting logic
3. Explanation of how it would scale to multiple servers
4. Analysis of memory usage and trade-offs

Focus on the sliding window log or sliding window counter algorithm.""",
        difficulty=Difficulty.HARD,
        category=Category.CODE_GENERATION,
        expected_patterns=[r"sliding window|token bucket|leaky bucket", r"redis|distributed"],
        tags=["system-design", "rate-limiting", "distributed"]
    ),

    # 9. Graph Algorithm
    TestProblem(
        id="hard_algo_01",
        name="Shortest Path with Constraints",
        description="Find shortest path with complex constraints",
        prompt="""Write a Python function `shortest_path_with_fuel(graph, start, end, max_fuel, refuel_stations)`:

- graph: Dict[node, List[Tuple[neighbor, distance]]]
- start: Starting node
- end: Destination node
- max_fuel: Maximum fuel tank capacity
- refuel_stations: Set of nodes where you can refuel to max_fuel

The car starts with a full tank. Each unit of distance consumes one unit of fuel.
You cannot traverse an edge if you don't have enough fuel.
At refuel stations, you automatically refuel to max_fuel.

Return the shortest total distance, or -1 if impossible.

Example:
```python
graph = {
    'A': [('B', 5), ('C', 10)],
    'B': [('A', 5), ('C', 3), ('D', 8)],
    'C': [('A', 10), ('B', 3), ('D', 2)],
    'D': [('B', 8), ('C', 2)]
}
# With max_fuel=7, no refuel stations: A->B->C->D = 10, but A->B->D impossible (needs 13)
# With max_fuel=7, refuel at B: A->B(refuel)->D = 13 total
```

Implement using modified Dijkstra's algorithm.""",
        difficulty=Difficulty.HARD,
        category=Category.CODE_GENERATION,
        expected_patterns=[r"def shortest_path", r"dijkstra|heap|priority"],
        tags=["python", "graphs", "dijkstra"]
    ),

    # 10. Complex Semantic Analysis
    TestProblem(
        id="hard_semantic_01",
        name="Code Review Analysis",
        description="Comprehensive code review with security focus",
        prompt="""Perform a comprehensive security code review of this Flask application:

```python
from flask import Flask, request, render_template_string, redirect
import sqlite3
import pickle
import os

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    cursor.execute(query)
    user = cursor.fetchone()

    if user:
        return f"Welcome {username}!"
    return "Invalid credentials"

@app.route('/profile/<username>')
def profile(username):
    template = f"<h1>Profile: {username}</h1>"
    return render_template_string(template)

@app.route('/upload', methods=['POST'])
def upload():
    data = request.files['data'].read()
    obj = pickle.loads(data)
    return str(obj)

@app.route('/file')
def get_file():
    filename = request.args.get('name')
    return open(filename).read()

@app.route('/exec')
def execute():
    cmd = request.args.get('cmd')
    return os.popen(cmd).read()
```

Identify:
1. All security vulnerabilities (name and severity: Critical/High/Medium/Low)
2. The specific attack vector for each
3. The fix for each vulnerability
4. Any missing security headers/configurations""",
        difficulty=Difficulty.HARD,
        category=Category.SEMANTIC,
        expected_patterns=[r"sql injection|sqli", r"xss|cross.site", r"pickle|deserialization", r"command injection|os injection"],
        tags=["security", "code-review", "flask"]
    ),
]


# =============================================================================
# PROBLEM COLLECTION AND UTILITIES
# =============================================================================

@dataclass
class ProblemSet:
    """A collection of test problems."""
    name: str
    problems: List[TestProblem]
    description: str = ""

    def get_by_difficulty(self, difficulty: Difficulty) -> List[TestProblem]:
        """Get problems by difficulty level."""
        return [p for p in self.problems if p.difficulty == difficulty]

    def get_by_category(self, category: Category) -> List[TestProblem]:
        """Get problems by category."""
        return [p for p in self.problems if p.category == category]

    def get_by_tag(self, tag: str) -> List[TestProblem]:
        """Get problems with a specific tag."""
        return [p for p in self.problems if tag in p.tags]


# Create the main problem sets
MEDIUM_PROBLEM_SET = ProblemSet(
    name="Medium Problems",
    problems=MEDIUM_PROBLEMS,
    description="Medium difficulty problems testing core LLM capabilities"
)

HARD_PROBLEM_SET = ProblemSet(
    name="Hard Problems",
    problems=HARD_PROBLEMS,
    description="Hard difficulty problems requiring complex reasoning and implementation"
)

ALL_PROBLEMS = ProblemSet(
    name="All Problems",
    problems=MEDIUM_PROBLEMS + HARD_PROBLEMS,
    description="Complete problem set for comprehensive LLM testing"
)


def get_problem_by_id(problem_id: str) -> Optional[TestProblem]:
    """Get a specific problem by ID."""
    for problem in ALL_PROBLEMS.problems:
        if problem.id == problem_id:
            return problem
    return None


def get_problems_summary() -> Dict[str, Any]:
    """Get summary statistics about available problems."""
    return {
        "total": len(ALL_PROBLEMS.problems),
        "by_difficulty": {
            d.value: len([p for p in ALL_PROBLEMS.problems if p.difficulty == d])
            for d in Difficulty
        },
        "by_category": {
            c.value: len([p for p in ALL_PROBLEMS.problems if p.category == c])
            for c in Category
        },
        "problem_ids": [p.id for p in ALL_PROBLEMS.problems]
    }
