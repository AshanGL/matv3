"""
answer_types.py
===============
Multi-type answer system for the Olympiad Math Solver.

Supports: integer, float, fraction, expression, set, string
Every component downstream (verify, vote, compare) works with
TypedAnswer objects — never raw strings or plain ints.

Memory notes
------------
- No models loaded here; pure rule-based + SymPy
- SymPy is lazy-imported so import cost is paid once
- All functions are stateless — safe to call from multiple threads
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Core dataclass
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_TYPES = ("integer", "float", "fraction", "expression", "set", "string")


@dataclass
class TypedAnswer:
    """
    Canonical container for any answer produced by the solver.

    Attributes
    ----------
    value        : int | float | sp.Expr | sp.Rational | frozenset | str
    answer_type  : one of ANSWER_TYPES
    raw_str      : original string from \\boxed{} — preserved for debugging
    confidence   : 0.0–1.0  extraction reliability
    tolerance    : absolute tolerance used for float/fraction comparison
    attempt_idx  : which solver attempt produced this (for voting)
    """
    value:       Any
    answer_type: str
    raw_str:     str
    confidence:  float        = 1.0
    tolerance:   float        = 1e-9
    attempt_idx: int          = -1
    extra:       dict         = field(default_factory=dict)

    def __post_init__(self):
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(f"Unknown answer_type: {self.answer_type!r}. "
                             f"Must be one of {ANSWER_TYPES}")

    def __repr__(self):
        return (f"TypedAnswer(type={self.answer_type}, value={self.value!r}, "
                f"conf={self.confidence:.2f})")

    # Convenience
    def is_numeric(self) -> bool:
        return self.answer_type in ("integer", "float", "fraction")

    def as_float(self) -> Optional[float]:
        try:
            return float(self.value)
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Answer type detector
# ─────────────────────────────────────────────────────────────────────────────

class AnswerTypeDetector:
    """
    Infers the expected answer type from the problem text + boxed content.

    Priority: problem-text signals  >  boxed-content shape  >  fallback
    The detector is intentionally conservative — when unsure it returns
    "string" so downstream code preserves the raw text rather than misparse.
    """

    # Signals in problem text → type
    _INTEGER_SIGNALS = [
        "how many", "find the number", "count the", "number of",
        "largest integer", "smallest integer", "smallest positive integer",
        "greatest integer", "least positive integer",
        "remainder when", "last two digits", "last digit",
        "number of ways", "in how many",
    ]
    _FLOAT_SIGNALS = [
        "probability", "expected value", "to 2 decimal", "to 3 decimal",
        "to the nearest", "average", "mean value", "approximate",
    ]
    _FRACTION_SIGNALS = [
        "as a fraction", "in the form p/q", "express as a ratio",
        "in lowest terms", "reduced fraction",
    ]
    _EXPRESSION_SIGNALS = [
        "in terms of", "express", "simplify", "factor", "find all real",
        "find f(", "closed form", "general term", "polynomial",
    ]
    _SET_SIGNALS = [
        "find all values", "list all", "all solutions", "all integers",
        "which values", "find all x", "find all n",
    ]

    def detect(self, problem: str, boxed_content: str) -> str:
        p = problem.lower()
        b = boxed_content.strip()

        # 1. Hard signals from problem text (order matters — most specific first)
        if any(s in p for s in self._FRACTION_SIGNALS):    return "fraction"
        if any(s in p for s in self._INTEGER_SIGNALS):     return "integer"
        if any(s in p for s in self._FLOAT_SIGNALS):       return "float"
        if any(s in p for s in self._SET_SIGNALS):         return "set"
        if any(s in p for s in self._EXPRESSION_SIGNALS):  return "expression"

        # 2. Infer from shape of boxed content
        return self._infer_from_content(b)

    def _infer_from_content(self, b: str) -> str:
        b = b.strip()

        # Pure integer (possibly signed, possibly comma-separated thousands)
        if re.fullmatch(r"-?\s*[\d,]+", b):
            return "integer"

        # Decimal float
        if re.fullmatch(r"-?\d+\.\d+", b):
            return "float"

        # Fraction  a/b  or  -a/b
        if re.fullmatch(r"-?\d+\s*/\s*-?\d+", b):
            return "fraction"

        # LaTeX fraction \frac{a}{b}
        if re.search(r"\\frac\s*\{", b):
            return "fraction"

        # Set notation  {1, 2, 3}  or  \{1, 2\}
        if (b.startswith("{") and b.endswith("}")) or \
           (b.startswith("\\{") and b.endswith("\\}")):
            return "set"

        # Comma-separated list of numbers without braces → set
        if re.fullmatch(r"-?[\d\./]+(,\s*-?[\d\./]+)+", b):
            return "set"

        # Contains algebra variable or math operators  → expression
        if re.search(r"[a-wyzA-WYZ\^\\]", b) or \
           any(tok in b for tok in ["sqrt", "\\sqrt", "log", "\\log",
                                     "sin", "cos", "tan", "pi", "\\pi",
                                     "infty", "\\infty"]):
            return "expression"

        # Default
        return "string"


# ─────────────────────────────────────────────────────────────────────────────
# Boxed content extractor  (raw string from LLM output)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns tried in order of reliability
_BOXED_PATTERNS = [
    r"\\boxed\s*\{\s*([^{}]+)\s*\}",          # \boxed{...}
    r"\\boxed\s*\{([^}]*)\}",                  # \boxed{...} no inner braces
    r"the\s+answer\s+is\s*[:\s]*([^\n.]+)",    # "the answer is ..."
    r"final\s+answer\s*[:\s=]*([^\n.]+)",      # "final answer: ..."
    r"=\s*\\boxed\s*\{\s*([^{}]+)\s*\}",       # = \boxed{...}
]


def extract_raw_boxed(text: str) -> Optional[str]:
    """
    Pull the raw string from inside \\boxed{} in the LLM output.
    Returns None if nothing found.
    Only looks in the assistant portion of the output (after 'assistant' token).
    """
    # Restrict to assistant section to avoid contamination from problem text
    for split_tok in ["assistant\n", "assistant\r", "assistant"]:
        if split_tok in text.lower():
            idx  = text.lower().rfind(split_tok)
            text = text[idx:]
            break

    for pat in _BOXED_PATTERNS:
        matches = re.findall(pat, text, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[-1].strip()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Type-specific extractors
# ─────────────────────────────────────────────────────────────────────────────

def _sympy():
    """Lazy import of sympy to avoid paying startup cost until needed."""
    import sympy
    return sympy


def extract_integer(raw: str) -> TypedAnswer:
    clean = raw.replace(",", "").replace(" ", "").strip()
    # Handle LaTeX negation  {-7}
    clean = re.sub(r"^\{(.*)\}$", r"\1", clean)
    try:
        val = int(clean)
        return TypedAnswer(value=val, answer_type="integer",
                           raw_str=raw, confidence=1.0)
    except ValueError:
        # Try via sympy
        sp = _sympy()
        try:
            val = int(sp.Integer(clean))
            return TypedAnswer(value=val, answer_type="integer",
                               raw_str=raw, confidence=0.85)
        except Exception:
            return TypedAnswer(value=raw, answer_type="string",
                               raw_str=raw, confidence=0.1,
                               extra={"parse_error": "integer"})


def extract_float(raw: str) -> TypedAnswer:
    clean = raw.strip().replace(",", "")
    try:
        val = float(clean)
        return TypedAnswer(value=val, answer_type="float",
                           raw_str=raw, confidence=0.95,
                           tolerance=1e-6)
    except ValueError:
        return TypedAnswer(value=raw, answer_type="string",
                           raw_str=raw, confidence=0.1,
                           extra={"parse_error": "float"})


def extract_fraction(raw: str) -> TypedAnswer:
    sp = _sympy()
    # a/b form
    m = re.search(r"(-?\d+)\s*/\s*(-?\d+)", raw)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return TypedAnswer(value=raw, answer_type="string",
                               raw_str=raw, confidence=0.0,
                               extra={"parse_error": "division by zero"})
        val = sp.Rational(num, den)
        return TypedAnswer(value=val, answer_type="fraction",
                           raw_str=raw, confidence=0.98)

    # \frac{a}{b}
    m2 = re.search(r"\\frac\s*\{\s*(-?\d+)\s*\}\s*\{\s*(-?\d+)\s*\}", raw)
    if m2:
        val = sp.Rational(int(m2.group(1)), int(m2.group(2)))
        return TypedAnswer(value=val, answer_type="fraction",
                           raw_str=raw, confidence=0.95)

    # Decimal that is actually rational  e.g. "0.75"
    try:
        f = float(raw.strip())
        val = sp.Rational(f).limit_denominator(10000)
        return TypedAnswer(value=val, answer_type="fraction",
                           raw_str=raw, confidence=0.80)
    except Exception:
        return TypedAnswer(value=raw, answer_type="string",
                           raw_str=raw, confidence=0.1,
                           extra={"parse_error": "fraction"})


def extract_expression(raw: str) -> TypedAnswer:
    sp = _sympy()

    # Normalise LaTeX → sympy-parseable
    clean = raw
    replacements = [
        (r"\\cdot",            "*"),
        (r"\\times",           "*"),
        (r"\\div",             "/"),
        (r"\\sqrt\s*\{([^}]+)\}", r"sqrt(\1)"),
        (r"\\sqrt\s*(\w+)",    r"sqrt(\1)"),
        (r"\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}", r"((\1)/(\2))"),
        (r"\^",                "**"),
        (r"\\left",            ""),
        (r"\\right",           ""),
        (r"\\pm",              ""),   # ambiguous — drop
        (r"\{",                "("),
        (r"\}",                ")"),
        (r"\\pi",              "pi"),
        (r"\\infty",           "oo"),
        (r"\\log",             "log"),
        (r"\\ln",              "log"),
        (r"\\sin",             "sin"),
        (r"\\cos",             "cos"),
        (r"\\tan",             "tan"),
    ]
    for pat, repl in replacements:
        clean = re.sub(pat, repl, clean)
    clean = clean.strip()

    try:
        expr = sp.parse_expr(clean, evaluate=True)
        expr = sp.simplify(expr)
        return TypedAnswer(value=expr, answer_type="expression",
                           raw_str=raw, confidence=0.88)
    except Exception:
        # Return raw as string — don't crash
        return TypedAnswer(value=raw, answer_type="string",
                           raw_str=raw, confidence=0.3,
                           extra={"parse_error": "expression", "clean": clean})


def extract_set(raw: str) -> TypedAnswer:
    sp = _sympy()

    # Strip outer braces
    inner = raw.strip()
    inner = re.sub(r"^\\?\{", "", inner)
    inner = re.sub(r"\\?\}$", "", inner)
    inner = inner.replace(" and ", ",").replace(" or ", ",")

    parts = [p.strip() for p in inner.split(",") if p.strip()]
    if not parts:
        return TypedAnswer(value=frozenset(), answer_type="set",
                           raw_str=raw, confidence=0.5)

    parsed = []
    all_ok = True
    for p in parts:
        try:
            parsed.append(sp.parse_expr(p, evaluate=True))
        except Exception:
            parsed.append(p.strip().lower())
            all_ok = False

    confidence = 0.90 if all_ok else 0.65
    return TypedAnswer(value=frozenset(str(x) for x in parsed),
                       answer_type="set",
                       raw_str=raw,
                       confidence=confidence)


def extract_string(raw: str) -> TypedAnswer:
    return TypedAnswer(value=raw.strip().lower(),
                       answer_type="string",
                       raw_str=raw,
                       confidence=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Master extraction entry point
# ─────────────────────────────────────────────────────────────────────────────

_DETECTOR = AnswerTypeDetector()

_EXTRACTOR_MAP = {
    "integer":    extract_integer,
    "float":      extract_float,
    "fraction":   extract_fraction,
    "expression": extract_expression,
    "set":        extract_set,
    "string":     extract_string,
}


def extract_answer(
    llm_output:  str,
    problem:     str,
    attempt_idx: int = -1,
    forced_type: Optional[str] = None,
) -> Optional[TypedAnswer]:
    """
    Full extraction pipeline:
      1. Pull raw boxed string from LLM output
      2. Detect answer type (or use forced_type)
      3. Run type-specific extractor
      4. Return TypedAnswer (or None if nothing found)

    Parameters
    ----------
    llm_output  : full text generated by the LLM for this attempt
    problem     : original problem text (used by type detector)
    attempt_idx : solver attempt index (stored in TypedAnswer for voting)
    forced_type : override type detection — useful when you already know the type
    """
    raw = extract_raw_boxed(llm_output)
    if raw is None:
        return None

    ans_type = forced_type or _DETECTOR.detect(problem, raw)
    extractor = _EXTRACTOR_MAP.get(ans_type, extract_string)
    typed = extractor(raw)
    typed.attempt_idx = attempt_idx
    return typed


# ─────────────────────────────────────────────────────────────────────────────
# Type-aware comparator
# ─────────────────────────────────────────────────────────────────────────────

def answers_match(a: TypedAnswer, b: TypedAnswer) -> bool:
    """
    Return True if two TypedAnswers represent the same mathematical value.

    Rules
    -----
    - Numeric types (integer / float / fraction) are compared as floats
      within tolerance; exact types (integer vs integer) use ==
    - Expressions are compared via sympy.simplify(a - b) == 0
    - Sets are compared element-wise after sympy simplification
    - Strings are compared case-insensitively after stripping whitespace
    - Incompatible type pairs → False (not a match)
    """
    numeric = {"integer", "float", "fraction"}

    # ── Both numeric ──────────────────────────────────────────────────────────
    if a.answer_type in numeric and b.answer_type in numeric:
        # integer vs integer → exact
        if a.answer_type == "integer" and b.answer_type == "integer":
            return a.value == b.value
        # otherwise → float comparison within tolerance
        fa, fb = a.as_float(), b.as_float()
        if fa is None or fb is None:
            return False
        tol = max(a.tolerance, b.tolerance)
        return abs(fa - fb) <= tol

    # ── Both expressions ──────────────────────────────────────────────────────
    if a.answer_type == "expression" and b.answer_type == "expression":
        sp = _sympy()
        try:
            diff = sp.simplify(a.value - b.value)
            return diff == 0
        except Exception:
            # Fallback: compare canonical string representations
            try:
                return str(sp.simplify(a.value)) == str(sp.simplify(b.value))
            except Exception:
                return str(a.value) == str(b.value)

    # ── Both sets ─────────────────────────────────────────────────────────────
    if a.answer_type == "set" and b.answer_type == "set":
        if len(a.value) != len(b.value):
            return False
        return a.value == b.value   # frozenset of canonical strings

    # ── Both strings ──────────────────────────────────────────────────────────
    if a.answer_type == "string" and b.answer_type == "string":
        return str(a.value).strip().lower() == str(b.value).strip().lower()

    # ── fraction vs expression ────────────────────────────────────────────────
    if {a.answer_type, b.answer_type} <= {"fraction", "expression"}:
        sp = _sympy()
        try:
            diff = sp.simplify(sp.sympify(a.value) - sp.sympify(b.value))
            return diff == 0
        except Exception:
            return False

    return False   # incompatible types


# ─────────────────────────────────────────────────────────────────────────────
# Answer vote selector  (replaces entropy-weighted voting)
# ─────────────────────────────────────────────────────────────────────────────

def select_best_answer(
    candidates: list[TypedAnswer],
    min_votes:  int = 2,
) -> Optional[TypedAnswer]:
    """
    Select the best answer from a list of TypedAnswer objects.

    Strategy
    --------
    1. Group candidates by mathematical equivalence using answers_match()
    2. The group with the most members wins (plurality vote)
    3. Within a tied group, prefer the one with highest confidence
    4. If no group has >= min_votes members, return the highest-confidence answer

    Parameters
    ----------
    candidates : list of TypedAnswer objects from all solver attempts
    min_votes  : minimum votes required for an answer to be selected by majority;
                 if no group reaches this threshold the highest-confidence single
                 answer is returned as a best-effort fallback
    """
    if not candidates:
        return None

    # Remove None and low-confidence failures
    valid = [c for c in candidates if c is not None and c.confidence >= 0.3]
    if not valid:
        return None

    # Group by equivalence
    groups: list[list[TypedAnswer]] = []
    for ans in valid:
        placed = False
        for group in groups:
            if answers_match(ans, group[0]):
                group.append(ans)
                placed = True
                break
        if not placed:
            groups.append([ans])

    # Sort groups by size desc, then by max confidence desc
    groups.sort(key=lambda g: (len(g), max(x.confidence for x in g)), reverse=True)

    best_group = groups[0]
    if len(best_group) >= min_votes:
        # Return the member with highest confidence in the winning group
        return max(best_group, key=lambda x: x.confidence)

    # No group has enough votes — return single highest-confidence answer
    return max(valid, key=lambda x: x.confidence)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = [
        ("Find the number of integers",          "\\boxed{42}",           "integer"),
        ("Find the probability",                  "\\boxed{0.375}",        "float"),
        ("Express as a fraction",                 "\\boxed{3/4}",          "fraction"),
        ("Simplify the expression",               "\\boxed{x^2 + 1}",      "expression"),
        ("Find all integer solutions",            "\\boxed{-1, 2, 5}",     "set"),
        ("State the theorem",                     "\\boxed{Fermat's last}", "string"),
    ]
    print("\n=== Answer type extraction smoke test ===\n")
    for prob, boxed, expected in cases:
        fake_output = f"assistant\n{boxed}"
        ta = extract_answer(fake_output, prob)
        ok = "OK" if (ta and ta.answer_type == expected) else "FAIL"
        print(f"  [{ok}]  expected={expected:12s}  got={ta}")
    print()
