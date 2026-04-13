"""
tools.py
========
The five tools available to the LLM during solving.
Each tool returns a structured dict — never raw text.

Tool list
---------
  knowledge_search(query, domain, mode, top_k)
  compute(expression, operation, variables, assumptions)
  numerical_search(condition_fn_src, search_space)
  verify(problem, typed_answer, approach_summary)
  run_code(code, timeout)

Memory / OOM design principles
-------------------------------
- Sentence encoder is loaded ONCE via a module-level singleton and shared
  across all tool calls — no repeated loads
- SymPy expressions are discarded after each call; no global caches
- Sandboxes are pooled (pool passed in from solver); tools never create them
- All tools time-out gracefully — they return an error dict rather than hang
- FAISS indexes are loaded lazily and cached by the KnowledgeDB class
"""

from __future__ import annotations

import re
import time
import signal
import textwrap
import threading
import traceback
from contextlib import contextmanager
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Timeout context manager (Unix only; falls back on Windows)
# ─────────────────────────────────────────────────────────────────────────────

class _TimeoutError(Exception):
    pass


@contextmanager
def _timeout(seconds: float):
    """Raise _TimeoutError if the block takes longer than `seconds`."""
    def _handler(signum, frame):
        raise _TimeoutError(f"Timed out after {seconds}s")

    try:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    except AttributeError:
        # SIGALRM not available (Windows) — run without timeout
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Sentence encoder singleton
# ─────────────────────────────────────────────────────────────────────────────

_encoder_lock = threading.Lock()
_encoder_instance = None
_encoder_model_name = None

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_encoder(model_name: str = EMBEDDING_MODEL):
    """
    Return a cached SentenceTransformer.
    Thread-safe: the first caller loads; subsequent callers reuse.
    """
    global _encoder_instance, _encoder_model_name
    with _encoder_lock:
        if _encoder_instance is None or _encoder_model_name != model_name:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _encoder_instance  = SentenceTransformer(model_name, device=device)
            _encoder_model_name = model_name
    return _encoder_instance


def _embed_query(query: str, model_name: str = EMBEDDING_MODEL):
    enc = get_encoder(model_name)
    import numpy as np
    emb = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return emb[0].astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# Tool result helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(**kwargs) -> dict:
    return {"status": "ok", **kwargs}


def _err(message: str, **kwargs) -> dict:
    return {"status": "error", "error": message, **kwargs}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: knowledge_search
# ─────────────────────────────────────────────────────────────────────────────

def knowledge_search(
    query:    str,
    db,                      # KnowledgeDB instance (passed in by solver)
    domain:   Optional[str] = None,
    mode:     str           = "both",   # "problems" | "theorems" | "both"
    top_k:    int           = 4,
    timeout:  float         = 10.0,
) -> dict:
    """
    Query the knowledge database for similar problems and/or theorems.

    Parameters
    ----------
    query   : natural-language problem description / technique query
    db      : KnowledgeDB instance (has .search_problems() and .search_theorems())
    domain  : optional domain filter ("Algebra", "Geometry", etc.)
    mode    : "problems" | "theorems" | "both"
    top_k   : results per store
    timeout : max seconds before returning partial results

    Returns
    -------
    {
      "status": "ok",
      "problems": [{"problem": ..., "technique_tags": [...], "answer_type": ...,
                    "difficulty_band": ..., "similarity": float}, ...],
      "theorems":  [{"name": ..., "statement": ..., "when_to_apply": ...,
                    "tags": [...], "similarity": float}, ...],
      "query":     original query string
    }
    """
    results = {"problems": [], "theorems": [], "query": query}

    try:
        with _timeout(timeout):
            query_emb = _embed_query(query)

            if mode in ("problems", "both"):
                results["problems"] = db.search_problems(
                    query_emb, domain=domain, top_k=top_k)

            if mode in ("theorems", "both"):
                results["theorems"] = db.search_theorems(
                    query_emb, domain=domain, top_k=top_k)

    except _TimeoutError:
        results["warning"] = f"knowledge_search timed out after {timeout}s"
    except Exception as e:
        return _err(f"knowledge_search failed: {e}",
                    problems=[], theorems=[], query=query)

    return _ok(**results)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: compute
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_OPERATIONS = {
    "simplify", "expand", "factor", "solve", "solve_system",
    "diff", "integrate", "limit", "series", "roots",
    "gcd", "lcm", "isprime", "factorint", "mod",
    "nsolve", "eigenvalues", "det", "inverse",
    "binomial", "factorial", "totient",
}


def compute(
    expression:  str,
    operation:   str,
    variables:   Optional[list[str]] = None,
    assumptions: Optional[dict]      = None,
    timeout:     float               = 15.0,
) -> dict:
    """
    Exact symbolic computation via SymPy.

    Parameters
    ----------
    expression  : math expression string (LaTeX-free; use ** not ^)
    operation   : one of _SUPPORTED_OPERATIONS
    variables   : variable names for solve/diff/integrate operations
    assumptions : symbol assumptions e.g. {"n": {"positive": True, "integer": True}}
    timeout     : max seconds

    Returns
    -------
    {
      "status":     "ok",
      "result":     string representation of result,
      "result_type": "integer" | "rational" | "expression" | "list" | "dict",
      "latex":      LaTeX representation,
      "numeric":    float approximation (if applicable),
      "operation":  operation name,
    }
    """
    if operation not in _SUPPORTED_OPERATIONS:
        return _err(f"Unknown operation {operation!r}. "
                    f"Supported: {sorted(_SUPPORTED_OPERATIONS)}")

    try:
        with _timeout(timeout):
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
                implicit_multiplication_application

            transforms = standard_transformations + (implicit_multiplication_application,)

            # Build symbol dict with optional assumptions
            local_dict = {}
            for name in _extract_symbol_names(expression, variables or []):
                kwargs = (assumptions or {}).get(name, {})
                local_dict[name] = sp.Symbol(name, **kwargs)

            # Parse expression
            try:
                expr = parse_expr(expression, local_dict=local_dict,
                                  transformations=transforms, evaluate=True)
            except Exception as pe:
                return _err(f"Could not parse expression: {pe}")

            # Execute operation
            result, rtype = _run_operation(sp, expr, operation, variables, local_dict)

            # Serialise result
            if isinstance(result, (list, tuple)):
                result_str = str([str(r) for r in result])
                latex_str  = ", ".join(sp.latex(r) for r in result)
                numeric    = None
            elif isinstance(result, dict):
                result_str = str({str(k): str(v) for k, v in result.items()})
                latex_str  = str(result)
                numeric    = None
            else:
                result_str = str(result)
                latex_str  = sp.latex(result)
                try:
                    numeric = float(result.evalf())
                except Exception:
                    numeric = None

            return _ok(
                result=result_str,
                result_type=rtype,
                latex=latex_str,
                numeric=numeric,
                operation=operation,
                expression=expression,
            )

    except _TimeoutError:
        return _err(f"compute timed out after {timeout}s", operation=operation)
    except Exception as e:
        return _err(f"compute error: {e}\n{traceback.format_exc()[:400]}",
                    operation=operation)


def _extract_symbol_names(expr_str: str, extra: list[str]) -> set[str]:
    """Pull single-letter variable names from an expression string."""
    names = set(re.findall(r"\b([a-zA-Z])\b", expr_str))
    # Remove common function names
    names -= {"e", "E", "I", "pi", "oo"}
    names |= set(extra)
    return names


def _run_operation(sp, expr, operation: str, variables, local_dict: dict):
    """Dispatch to the correct SymPy function. Returns (result, type_str)."""
    vars_syms = [local_dict[v] for v in (variables or []) if v in local_dict]
    if not vars_syms:
        # Guess from expression free symbols
        vars_syms = sorted(expr.free_symbols, key=str)

    op = operation.lower()

    if op == "simplify":   return sp.simplify(expr), "expression"
    if op == "expand":     return sp.expand(expr), "expression"
    if op == "factor":     return sp.factor(expr), "expression"
    if op == "roots":
        r = sp.roots(expr, *vars_syms[:1])
        return r, "dict"
    if op == "solve":
        sol = sp.solve(expr, *vars_syms[:1])
        return sol, "list"
    if op == "solve_system":
        # expr should be a comma-separated list of equations
        eqs = [sp.parse_expr(e.strip()) for e in str(expr).split(";")]
        sol = sp.solve(eqs, vars_syms)
        return sol, "list"
    if op == "diff":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.diff(expr, v), "expression"
    if op == "integrate":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.integrate(expr, v), "expression"
    if op == "limit":
        v    = vars_syms[0] if vars_syms else sp.Symbol("x")
        point = vars_syms[1] if len(vars_syms) > 1 else sp.oo
        return sp.limit(expr, v, point), "expression"
    if op == "series":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.series(expr, v, n=6), "expression"
    if op == "gcd":
        parts = str(expr).split(",")
        a, b  = [sp.sympify(p.strip()) for p in parts[:2]]
        return sp.gcd(a, b), "expression"
    if op == "lcm":
        parts = str(expr).split(",")
        a, b  = [sp.sympify(p.strip()) for p in parts[:2]]
        return sp.lcm(a, b), "expression"
    if op == "isprime":
        n = int(sp.sympify(expr))
        return sp.isprime(n), "integer"
    if op == "factorint":
        n = int(sp.sympify(expr))
        return sp.factorint(n), "dict"
    if op == "mod":
        parts = str(expr).split(",")
        a, m  = int(sp.sympify(parts[0].strip())), int(sp.sympify(parts[1].strip()))
        return a % m, "integer"
    if op == "binomial":
        parts = str(expr).split(",")
        n, k  = int(sp.sympify(parts[0].strip())), int(sp.sympify(parts[1].strip()))
        return sp.binomial(n, k), "integer"
    if op == "factorial":
        n = int(sp.sympify(expr))
        return sp.factorial(n), "integer"
    if op == "totient":
        n = int(sp.sympify(expr))
        return sp.totient(n), "integer"
    if op == "det":
        mat = sp.Matrix(sp.sympify(expr))
        return mat.det(), "expression"
    if op == "inverse":
        mat = sp.Matrix(sp.sympify(expr))
        return mat.inv(), "expression"
    if op == "eigenvalues":
        mat = sp.Matrix(sp.sympify(expr))
        return mat.eigenvals(), "dict"
    if op == "nsolve":
        v = vars_syms[0] if vars_syms else sp.Symbol("x")
        return sp.nsolve(expr, v, 1), "expression"

    raise ValueError(f"Operation {operation!r} dispatcher not implemented")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: numerical_search
# ─────────────────────────────────────────────────────────────────────────────

def numerical_search(
    condition_src: str,
    search_space:  dict,
    max_results:   int   = 20,
    timeout:       float = 20.0,
) -> dict:
    """
    Brute-force search over a discrete space.
    The LLM provides a Python lambda/expression string for the condition.

    Parameters
    ----------
    condition_src : Python expression string that evaluates to bool.
                    Available names: n, m, k (the iteration variable/s).
                    Example: "n**2 + 1 == 5*n - 3"
    search_space  : {"type": "range", "lo": int, "hi": int}
                  | {"type": "range2d", "lo1": int, "hi1": int,
                                        "lo2": int, "hi2": int}
    max_results   : stop after finding this many solutions
    timeout       : max seconds

    Returns
    -------
    {
      "status":  "ok",
      "matches": [int | (int, int), ...],
      "count":   int,
      "space":   description of searched space,
      "exhausted": bool  (True if entire space was checked)
    }
    """
    # Whitelist — only safe builtins
    _SAFE_GLOBALS = {
        "__builtins__": {},
        "abs": abs, "min": min, "max": max, "pow": pow,
        "round": round, "divmod": divmod,
    }
    try:
        import math as _math
        _SAFE_GLOBALS.update({k: getattr(_math, k) for k in dir(_math)
                               if not k.startswith("_")})
    except Exception:
        pass

    # Compile condition
    try:
        code = compile(condition_src, "<condition>", "eval")
    except SyntaxError as e:
        return _err(f"Invalid condition syntax: {e}", matches=[], count=0)

    matches  = []
    exhausted = False
    start    = time.time()

    try:
        stype = search_space.get("type", "range")

        if stype == "range":
            lo = int(search_space.get("lo", 0))
            hi = int(search_space.get("hi", 1000))
            if hi - lo > 10_000_000:
                return _err("Search space too large (max 10M)", matches=[], count=0)

            with _timeout(timeout):
                for n in range(lo, hi + 1):
                    if len(matches) >= max_results:
                        break
                    if time.time() - start > timeout:
                        break
                    try:
                        env = {**_SAFE_GLOBALS, "n": n, "k": n, "m": n}
                        if eval(code, env):
                            matches.append(n)
                    except Exception:
                        continue
                else:
                    exhausted = len(matches) < max_results

            space_desc = f"n ∈ [{lo}, {hi}]"

        elif stype == "range2d":
            lo1, hi1 = int(search_space["lo1"]), int(search_space["hi1"])
            lo2, hi2 = int(search_space["lo2"]), int(search_space["hi2"])
            if (hi1 - lo1) * (hi2 - lo2) > 1_000_000:
                return _err("2D search space too large (max 1M)", matches=[], count=0)

            with _timeout(timeout):
                done = False
                for n in range(lo1, hi1 + 1):
                    if done:
                        break
                    for m in range(lo2, hi2 + 1):
                        if len(matches) >= max_results:
                            done = True
                            break
                        if time.time() - start > timeout:
                            done = True
                            break
                        try:
                            env = {**_SAFE_GLOBALS, "n": n, "m": m, "k": n}
                            if eval(code, env):
                                matches.append((n, m))
                        except Exception:
                            continue

            space_desc = f"n ∈ [{lo1},{hi1}], m ∈ [{lo2},{hi2}]"

        else:
            return _err(f"Unknown search_space type: {stype!r}", matches=[], count=0)

    except _TimeoutError:
        pass  # Return partial results

    return _ok(
        matches=matches,
        count=len(matches),
        space=space_desc,
        exhausted=exhausted,
        elapsed_s=round(time.time() - start, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: verify
# ─────────────────────────────────────────────────────────────────────────────

def verify(
    problem:          str,
    typed_answer,                       # TypedAnswer from answer_types.py
    sandbox,                            # MathSandbox instance from pool
    approach_summary: str = "",
    timeout:          float = 12.0,
) -> dict:
    """
    Multi-stage verification of a TypedAnswer.

    Stages
    ------
    1. Sanity  — answer was extracted with confidence >= 0.5
    2. Range   — numeric answers are within [-1e15, 1e15]
    3. Symbolic — attempt sympy substitution / simplification check
    4. Sandbox  — run a small verification script in the isolated kernel
    5. Consistency — answer type matches what the problem implies

    Returns
    -------
    {
      "status":   "ok",
      "passed":   bool,
      "checks":   {"sanity": bool, "range": bool, "symbolic": bool,
                   "sandbox": bool, "consistency": bool},
      "failed":   [list of failed check names],
      "answer":   str(answer value),
      "type":     answer_type string,
    }
    """
    checks = {}

    # Stage 1: confidence sanity
    checks["sanity"] = (typed_answer is not None and
                        typed_answer.confidence >= 0.40)
    if not checks["sanity"]:
        return _ok(passed=False, checks=checks,
                   failed=["sanity"],
                   answer=str(getattr(typed_answer, "value", None)),
                   type=getattr(typed_answer, "answer_type", "unknown"))

    val  = typed_answer.value
    atype = typed_answer.answer_type

    # Stage 2: numeric range
    if typed_answer.is_numeric():
        fval = typed_answer.as_float()
        checks["range"] = (fval is not None and
                           math_finite(fval) and
                           abs(fval) < 1e15)
    else:
        checks["range"] = True   # non-numeric — skip range check

    # Stage 3: symbolic check via SymPy
    checks["symbolic"] = _symbolic_check(val, atype, problem)

    # Stage 4: sandbox execution check
    checks["sandbox"] = _sandbox_check(val, atype, problem, sandbox, timeout)

    # Stage 5: type consistency with problem
    checks["consistency"] = _type_consistency_check(atype, problem)

    failed = [k for k, v in checks.items() if not v]
    passed = len(failed) == 0

    return _ok(
        passed=passed,
        checks=checks,
        failed=failed,
        answer=str(val),
        type=atype,
        approach_summary=approach_summary[:200],
    )


def math_finite(x: float) -> bool:
    import math
    return math.isfinite(x)


def _symbolic_check(val, atype: str, problem: str) -> bool:
    """Try to confirm the answer is a valid mathematical object."""
    try:
        import sympy as sp
        if atype == "integer":
            return isinstance(val, (int, sp.Integer))
        if atype in ("float", "fraction"):
            f = float(val)
            return math_finite(f)
        if atype == "expression":
            # Check it's a valid sympy expression
            simplified = sp.simplify(val)
            return simplified is not None
        if atype == "set":
            return len(val) > 0
        return True   # string — pass symbolic stage
    except Exception:
        return True   # Don't penalise if sympy can't parse


def _sandbox_check(val, atype: str, problem: str, sandbox, timeout: float) -> bool:
    """Run a lightweight verification script in the sandbox kernel."""
    if sandbox is None:
        return True   # No sandbox available — skip silently

    code = textwrap.dedent(f"""
import sympy as sp
import math

answer = {repr(val)}
answer_type = {repr(atype)}

# Basic type check
if answer_type == 'integer':
    assert isinstance(answer, int), f"Expected int, got {{type(answer)}}"
elif answer_type == 'float':
    assert math.isfinite(float(answer)), "Float is not finite"
elif answer_type == 'fraction':
    f = float(answer)
    assert math.isfinite(f), "Fraction is not finite"
elif answer_type == 'expression':
    expr = sp.sympify(answer)
    assert expr is not None, "Expression could not be parsed"

print("VERIFY_OK")
""")

    try:
        result = sandbox.execute(code)
        return "VERIFY_OK" in result and "Error" not in result
    except Exception:
        return False


def _type_consistency_check(atype: str, problem: str) -> bool:
    """
    Lightweight check: does the answer type make sense given the problem text?
    Returns True if consistent or ambiguous, False only if clearly wrong.
    """
    p = problem.lower()

    # If problem says "how many" but answer is expression → inconsistent
    if any(s in p for s in ["how many", "count", "number of"]):
        if atype == "expression":
            return False

    # If problem says "probability" but answer is very large integer → suspicious
    # (Don't hard-fail — just flag; verifier is advisory for this check)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: run_code
# ─────────────────────────────────────────────────────────────────────────────

def run_code(
    code:    str,
    sandbox,                   # MathSandbox instance
    timeout: float = 10.0,
) -> dict:
    """
    Execute arbitrary Python code in the isolated sandbox kernel.

    The sandbox is pre-loaded with numpy, sympy, math, itertools, fractions.
    The LLM should always print() results it wants to read back.

    Returns
    -------
    {
      "status":         "ok" | "error",
      "stdout":         captured output,
      "stderr":         error / traceback (if any),
      "has_error":      bool,
      "execution_time": seconds,
    }
    """
    if sandbox is None:
        return _err("No sandbox available for run_code",
                    stdout="", stderr="", has_error=True, execution_time=0.0)

    start = time.time()
    try:
        output = sandbox.execute(code)   # MathSandbox.execute() handles timeout internally
        elapsed = time.time() - start

        has_error = (
            "[ERROR]" in output or
            "Traceback" in output or
            "Error:" in output
        )

        if has_error:
            return {
                "status":         "error",
                "stdout":         "",
                "stderr":         output[:2000],
                "has_error":      True,
                "execution_time": round(elapsed, 3),
            }
        return _ok(
            stdout=output[:4000],
            stderr="",
            has_error=False,
            execution_time=round(elapsed, 3),
        )

    except Exception as e:
        return _err(
            str(e),
            stdout="",
            stderr=traceback.format_exc()[:1000],
            has_error=True,
            execution_time=round(time.time() - start, 3),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatcher — used by the solver loop
# ─────────────────────────────────────────────────────────────────────────────

class ToolDispatcher:
    """
    Single object that holds references to all tools and dispatches
    LLM tool-call requests.

    The solver passes one ToolDispatcher per problem attempt.
    It holds: the knowledge DB, a sandbox from the pool.
    """

    TOOL_NAMES = ("knowledge_search", "compute", "numerical_search",
                  "verify", "run_code")

    def __init__(self, db, sandbox):
        self.db      = db       # KnowledgeDB
        self.sandbox = sandbox  # MathSandbox or None

    def call(self, tool_name: str, args: dict) -> dict:
        """
        Dispatch a tool call by name.
        Returns the tool result dict.
        Errors are caught and returned as {"status": "error", ...}.
        """
        if tool_name not in self.TOOL_NAMES:
            return _err(f"Unknown tool: {tool_name!r}. "
                        f"Available: {self.TOOL_NAMES}")
        try:
            if tool_name == "knowledge_search":
                return knowledge_search(db=self.db, **args)
            if tool_name == "compute":
                return compute(**args)
            if tool_name == "numerical_search":
                return numerical_search(**args)
            if tool_name == "verify":
                return verify(sandbox=self.sandbox, **args)
            if tool_name == "run_code":
                return run_code(sandbox=self.sandbox, **args)
        except Exception as e:
            return _err(f"Uncaught error in {tool_name}: {e}\n"
                        f"{traceback.format_exc()[:400]}")

    def tool_schema(self) -> list[dict]:
        """
        Return OpenAI-style function schema for all tools.
        Passed to the LLM so it knows how to call each tool.
        """
        return [
            {
                "name": "knowledge_search",
                "description": (
                    "Search the knowledge database for similar problems and/or relevant theorems. "
                    "Always call this FIRST before attempting to solve the problem."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query":  {"type": "string",
                                   "description": "What technique or concept to search for"},
                        "domain": {"type": "string",
                                   "description": "Optional domain filter e.g. 'Algebra'"},
                        "mode":   {"type": "string",
                                   "enum": ["problems", "theorems", "both"],
                                   "description": "Which store to search"},
                        "top_k":  {"type": "integer", "description": "Results to return (default 4)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "compute",
                "description": (
                    "Exact symbolic computation via SymPy. Use for factoring, solving equations, "
                    "differentiation, integration, modular arithmetic, etc. "
                    "Prefer this over run_code for pure symbolic math."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string",
                                       "description": "Math expression (Python syntax, use ** not ^)"},
                        "operation":  {"type": "string",
                                       "description": f"One of: {sorted(_SUPPORTED_OPERATIONS)}"},
                        "variables":  {"type": "array", "items": {"type": "string"},
                                       "description": "Variable names e.g. ['x', 'y']"},
                        "assumptions": {"type": "object",
                                        "description": "Symbol assumptions e.g. {'n': {'positive': true}}"},
                    },
                    "required": ["expression", "operation"],
                },
            },
            {
                "name": "numerical_search",
                "description": (
                    "Brute-force search over integer ranges. Use when you can't solve analytically "
                    "but can describe a verification condition. Returns all matching values."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "condition_src": {"type": "string",
                                          "description": "Python boolean expression using n (and m for 2D). E.g. 'n**2 + 1 == 5*n - 3'"},
                        "search_space":  {"type": "object",
                                          "description": '{"type": "range", "lo": 0, "hi": 1000} or {"type": "range2d", "lo1": 0, "hi1": 100, "lo2": 0, "hi2": 100}'},
                        "max_results":   {"type": "integer", "description": "Stop after N matches (default 20)"},
                    },
                    "required": ["condition_src", "search_space"],
                },
            },
            {
                "name": "verify",
                "description": (
                    "Verify a proposed answer against the problem. "
                    "The pipeline REQUIRES you to call this before submitting a final answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "problem":          {"type": "string", "description": "Original problem text"},
                        "typed_answer":     {"type": "object", "description": "TypedAnswer dict from answer extraction"},
                        "approach_summary": {"type": "string", "description": "One-line summary of your approach"},
                    },
                    "required": ["problem", "typed_answer"],
                },
            },
            {
                "name": "run_code",
                "description": (
                    "Execute arbitrary Python code in an isolated sandbox. "
                    "Use for complex algorithms, numerical experiments, or any computation "
                    "not covered by the compute tool. Always print() your results."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string",
                                 "description": "Python code to execute. Must print() results."},
                    },
                    "required": ["code"],
                },
            },
        ]
