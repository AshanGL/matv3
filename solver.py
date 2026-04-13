"""
solver.py
=========
OlympiadSolver — semi-agentic tool-based solver.

Architecture
------------
  Pipeline controller  (this file)
      │
      ├─ Phase 1: forced knowledge_search
      ├─ Phase 2: LLM plans approach
      ├─ Phase 3: LLM ↔ tools loop (up to max_turns)
      ├─ Phase 4: forced verify
      └─ Phase 5: vote and select answer

Memory / OOM design
-------------------
- vLLM server runs as a subprocess; its GPU memory is isolated
- Sandbox pool uses a fixed number of kernel processes (not threads)
- Trained classifiers (AnswerTypeClassifier, VerifyScorer) are loaded
  in CPU inference mode with half precision — negligible memory
- Each attempt's full_response string is discarded after answer extraction
- torch.cuda.empty_cache() called between problems
- max_tokens set to 8192 (was 4096 — the bug that caused most failures)
"""

from __future__ import annotations

import gc
import os
import re
import json
import time
import math
import queue
import threading
import traceback
import contextlib
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
warnings.filterwarnings("ignore")

from answer_types import (
    TypedAnswer, extract_answer, select_best_answer, answers_match, ANSWER_TYPES
)
from tools import ToolDispatcher, run_code, verify
from knowledge_db import KnowledgeDB

# ─────────────────────────────────────────────────────────────────────────────
# Solver configuration
# ─────────────────────────────────────────────────────────────────────────────

class SolverConfig:
    # Model
    model_path           = "/kaggle/input/gpt-oss-120b/transformers/default/1"
    served_model_name    = "gpt-oss"

    # Memory
    kv_cache_dtype       = "fp8_e4m3"
    dtype                = "auto"
    gpu_memory_utilization = 0.92    # leave 8% headroom

    # Generation  — FIXED: was 4096, bumped to 16384
    max_tokens           = 16384
    temperature          = 0.8
    min_p                = 0.02
    top_logprobs         = 5
    context_tokens       = 131072
    stream_interval      = 128

    # Timing
    notebook_limit       = 17400
    server_timeout       = 180
    high_problem_timeout = 900
    base_problem_timeout = 300
    sandbox_timeout      = 5
    jupyter_timeout      = 10

    # Concurrency
    attempts             = 6       # parallel attempts per problem
    workers              = 6       # sandbox pool size (= attempts)
    max_tool_turns       = 12      # max tool calls per attempt
    early_stop_votes     = 4       # stop if N attempts agree
    seed                 = 42
    batch_size           = 64

    # Paths
    ckpt_dir             = "/kaggle/working/checkpoints"
    db_dir               = "/kaggle/working/knowledge_db"


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight inference wrappers for trained classifiers (CPU, half-precision)
# ─────────────────────────────────────────────────────────────────────────────

class AnswerTypeInference:
    """Load trained AnswerTypeClassifier for CPU inference."""

    def __init__(self, ckpt_dir: str):
        import json
        from transformers import AutoTokenizer
        from train_new import AnswerTypeModel, ANSWER_TYPES

        with open(os.path.join(ckpt_dir, "config.json")) as f:
            cfg = json.load(f)

        self.tokenizer    = AutoTokenizer.from_pretrained(ckpt_dir)
        self.model        = AnswerTypeModel(cfg["encoder_name"],
                                            n_classes=len(ANSWER_TYPES))
        self.model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location="cpu"))
        self.model.eval()
        self.answer_types = cfg.get("answer_types", ANSWER_TYPES)

    def predict(self, problem: str) -> str:
        enc = self.tokenizer(problem, max_length=192, padding="max_length",
                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        return self.answer_types[logits.argmax(-1).item()]


class VerifyScorerInference:
    """Load trained VerifyScorer for CPU inference."""

    def __init__(self, ckpt_dir: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir, num_labels=2)
        self.model.eval()

    def score(self, problem: str, trace: str) -> tuple[bool, float]:
        enc = self.tokenizer(problem, trace, max_length=384, padding="max_length",
                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"]).logits
        probs    = torch.softmax(logits, -1).squeeze()
        is_valid = logits.argmax(-1).item() == 1
        return is_valid, float(probs[1])


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert olympiad mathematics solver.
You have access to five tools:
  • knowledge_search  — search for similar problems and theorems (call FIRST)
  • compute           — exact symbolic math via SymPy
  • numerical_search  — brute-force integer range search
  • verify            — check your final answer (REQUIRED before finishing)
  • run_code          — execute arbitrary Python in a sandbox

PROCESS:
1. Call knowledge_search immediately to find relevant theorems and techniques.
2. Analyse the search results and choose your approach.
3. Use compute / numerical_search / run_code to work through the solution.
4. When you have a candidate answer, call verify.
5. If verify fails, revise your approach and try again.
6. Express your final answer as \\boxed{answer} where answer is the exact value.

RULES:
- Answer types: integer, float (e.g. 0.75), fraction (e.g. 3/4),
  expression (e.g. x^2+1), set (e.g. {1, 2, 3}), or string.
- Use SymPy for exact arithmetic — never float approximations for integer problems.
- Always print() intermediate results when using run_code.
- You MUST call verify before giving your final \\boxed{} answer.
"""


def build_user_prompt(problem: str, knowledge_results: dict) -> str:
    parts = [f"Problem:\n{problem}\n"]

    if knowledge_results:
        theorems  = knowledge_results.get("theorems", [])
        problems  = knowledge_results.get("problems", [])

        if theorems:
            parts.append("Potentially relevant theorems:")
            for t in theorems[:3]:
                parts.append(f"  • {t['name']}: {t['when_to_apply']}")

        if problems:
            parts.append("\nSimilar problems used these techniques:")
            for p in problems[:2]:
                tags = ", ".join(p.get("technique_tags", [])[:4])
                if tags:
                    parts.append(f"  • [{p.get('domain','')} / {p.get('difficulty_band','')}] "
                                 f"Techniques: {tags}")

    parts.append("\nSolve step by step using the available tools. "
                 "Give your final answer as \\boxed{answer}.")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# LLM call  (fix: max_tokens=16384, proper error propagation)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(client, system_prompt: str, messages: list[dict], cfg: SolverConfig,
             seed: int = 42) -> str:
    """
    Call the vLLM server and return the full response text.
    Raises on failure instead of silently returning "".
    """
    try:
        from openai_harmony import (
            load_harmony_encoding, HarmonyEncodingName,
            SystemContent, ReasoningEffort,
            Message, Role, Conversation,
        )

        sys_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(ReasoningEffort.HIGH)
        )
        sys_msg  = Message.from_role_and_content(Role.SYSTEM, sys_content)
        conv_msgs = [sys_msg] + [
            Message.from_role_and_content(
                Role.USER if m["role"] == "user" else Role.ASSISTANT,
                m["content"]
            ) for m in messages
        ]

        conversation     = Conversation(messages=conv_msgs)
        encoding         = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids   = encoding.stop_tokens_for_assistant_actions()
        prompt_token_ids = encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT)

        chunks = []
        stream = client.completions.create(
            model       = cfg.served_model_name,
            prompt      = prompt_token_ids,
            temperature = cfg.temperature,
            max_tokens  = cfg.max_tokens,    # 16384 — not 4096
            seed        = seed,
            stream      = True,
            extra_body  = {
                "min_p":          cfg.min_p,
                "stop_token_ids": stop_token_ids,
            },
        )
        for chunk in stream:
            delta = chunk.choices[0].text
            if delta:
                chunks.append(delta)
        stream.close()
        return "".join(chunks)

    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parser  (reads LLM output for tool call JSON blocks)
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
    re.DOTALL,
)
_JSON_BLOCK_PATTERN = re.compile(
    r'```json\s*(\{.*?\})\s*```',
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict]:
    """
    Extract tool calls from LLM output.
    Supports two formats:
      <tool_call>{"name": "compute", "arguments": {...}}</tool_call>
      ```json {"name": "compute", "arguments": {...}} ```
    """
    calls = []
    for pat in (_TOOL_CALL_PATTERN, _JSON_BLOCK_PATTERN):
        for match in pat.finditer(text):
            try:
                obj = json.loads(match.group(1))
                if "name" in obj and obj["name"] in ToolDispatcher.TOOL_NAMES:
                    calls.append({"name": obj["name"],
                                  "arguments": obj.get("arguments", obj.get("args", {}))})
            except json.JSONDecodeError:
                pass
    return calls


def format_tool_result(tool_name: str, result: dict) -> str:
    """Format a tool result for injection into the conversation."""
    status = result.get("status", "?")
    if tool_name == "knowledge_search":
        theorems = result.get("theorems", [])
        problems = result.get("problems", [])
        lines = [f"[knowledge_search result — status: {status}]"]
        if theorems:
            lines.append("Theorems:")
            for t in theorems[:3]:
                lines.append(f"  • {t['name']} ({t.get('similarity',0):.2f}): "
                             f"{t['when_to_apply']}")
        if problems:
            lines.append("Similar problems:")
            for p in problems[:2]:
                tags = ", ".join(p.get("technique_tags", [])[:4])
                lines.append(f"  • [{p.get('domain','')}] techniques: {tags}")
        return "\n".join(lines)

    if tool_name == "compute":
        if status == "ok":
            return (f"[compute result]\n"
                    f"  operation: {result.get('operation')}\n"
                    f"  result:    {result.get('result')}\n"
                    f"  latex:     {result.get('latex')}\n"
                    f"  numeric:   {result.get('numeric')}")
        return f"[compute error] {result.get('error')}"

    if tool_name == "numerical_search":
        if status == "ok":
            return (f"[numerical_search result]\n"
                    f"  space:    {result.get('space')}\n"
                    f"  matches:  {result.get('matches')}\n"
                    f"  count:    {result.get('count')}")
        return f"[numerical_search error] {result.get('error')}"

    if tool_name == "verify":
        if status == "ok":
            passed = result.get("passed")
            return (f"[verify result]\n"
                    f"  passed:  {passed}\n"
                    f"  checks:  {result.get('checks')}\n"
                    f"  failed:  {result.get('failed')}\n"
                    f"  answer:  {result.get('answer')}  ({result.get('type')})")
        return f"[verify error] {result.get('error')}"

    if tool_name == "run_code":
        if status == "ok":
            return (f"[run_code output]\n"
                    f"{result.get('stdout', '')[:1500]}")
        return (f"[run_code error]\n"
                f"{result.get('stderr', '')[:500]}")

    return f"[{tool_name}] {json.dumps(result)[:500]}"


# ─────────────────────────────────────────────────────────────────────────────
# Single attempt executor
# ─────────────────────────────────────────────────────────────────────────────

def run_attempt(
    problem:       str,
    client,
    dispatcher:    ToolDispatcher,
    cfg:           SolverConfig,
    attempt_idx:   int,
    stop_event:    threading.Event,
    deadline:      float,
    forced_type:   Optional[str] = None,
) -> Optional[TypedAnswer]:
    """
    Run one attempt of the solver loop.
    Returns a TypedAnswer or None if no answer was found.

    Phase 1 (forced): knowledge_search
    Phase 2 (LLM):    plan
    Phase 3 (LLM↔tools): solve loop
    Phase 4 (forced): verify
    """
    if stop_event.is_set() or time.time() > deadline:
        return None

    seed = (cfg.seed + attempt_idx * 7) % (2**31)
    messages: list[dict] = []
    verified_answer: Optional[TypedAnswer] = None

    try:
        # ── Phase 1: forced knowledge_search ─────────────────────────────────
        from tools import get_encoder
        emb = get_encoder().encode(
            [problem], normalize_embeddings=True, convert_to_numpy=True)[0]

        ks_result = dispatcher.call("knowledge_search", {
            "query":  problem,
            "mode":   "both",
            "top_k":  4,
        })

        # Build initial user message with search results baked in
        user_msg = build_user_prompt(problem, ks_result)
        messages.append({"role": "user", "content": user_msg})

        # ── Phase 3: solve loop ───────────────────────────────────────────────
        turn = 0
        verify_called = False

        while turn < cfg.max_tool_turns:
            if stop_event.is_set() or time.time() > deadline:
                break

            # LLM generates next step
            llm_text = call_llm(client, SYSTEM_PROMPT, messages, cfg, seed=seed + turn)
            messages.append({"role": "assistant", "content": llm_text})

            # Check if LLM provided a final answer
            typed = extract_answer(llm_text, problem, attempt_idx=attempt_idx,
                                   forced_type=forced_type)

            # Parse tool calls from LLM output
            tool_calls = parse_tool_calls(llm_text)

            if not tool_calls:
                # No tool calls — LLM is done reasoning
                if typed is not None:
                    break
                # Prompt LLM to use a tool
                messages.append({
                    "role": "user",
                    "content": ("Please use a tool to make progress. "
                                "Start with knowledge_search if you haven't yet, "
                                "then compute or run_code to solve.")
                })
                turn += 1
                continue

            # Execute each tool call
            tool_result_texts = []
            for tc in tool_calls[:3]:   # max 3 tool calls per LLM turn
                if stop_event.is_set():
                    break
                tname  = tc["name"]
                targs  = tc.get("arguments", {})
                result = dispatcher.call(tname, targs)
                tool_result_texts.append(format_tool_result(tname, result))

                # Track if verify was called
                if tname == "verify" and result.get("passed"):
                    verify_called = True
                    # Extract the answer from the verify call
                    if typed is not None:
                        verified_answer = typed

            # Inject tool results back into conversation
            combined = "\n\n".join(tool_result_texts)
            messages.append({"role": "user", "content": combined})
            turn += 1

        # ── Phase 4: force verify if not called ──────────────────────────────
        if typed is not None and not verify_called:
            verify_result = dispatcher.call("verify", {
                "problem":      problem,
                "typed_answer": {"value": typed.value,
                                 "answer_type": typed.answer_type,
                                 "raw_str": typed.raw_str,
                                 "confidence": typed.confidence},
                "approach_summary": "Pipeline-forced final verify",
            })
            if verify_result.get("passed"):
                verified_answer = typed

        # Return best answer found (verified or raw)
        return verified_answer if verified_answer is not None else typed

    except Exception as e:
        print(f"  [attempt {attempt_idx}] error: {e}")
        traceback.print_exc()
        return None
    finally:
        # Release message list from memory
        del messages


# ─────────────────────────────────────────────────────────────────────────────
# OlympiadSolver
# ─────────────────────────────────────────────────────────────────────────────

class OlympiadSolver:
    """
    Main solver class.  Initialise once; call solve_problem() per question.

    Memory budget
    -------------
    vLLM model:            ~60 GB GPU (120B at fp8)
    Sandboxes (6 kernels): ~300 MB CPU
    FAISS indexes:         < 20 MB CPU
    Classifiers (CPU):     ~400 MB CPU
    Total CPU RAM:         ~2 GB
    GPU RAM for vLLM:      controlled by gpu_memory_utilization=0.92
    """

    def __init__(
        self,
        cfg:         SolverConfig = None,
        load_models: bool         = True,
        port:        int          = 8000,
    ):
        self.cfg  = cfg or SolverConfig()
        self.port = port
        self.notebook_start_time = time.time()
        self.problems_remaining  = 50

        # ── Knowledge DB ──────────────────────────────────────────────────────
        print("Loading knowledge database...")
        self.db = KnowledgeDB(db_dir=self.cfg.db_dir)
        if not self.db.is_built():
            print("  WARNING: Knowledge DB not built. Run notebook cell 2 first.")

        # ── Trained classifiers (CPU, lightweight) ────────────────────────────
        self.type_classifier = None
        self.verify_scorer   = None

        if load_models:
            try:
                ckpt = os.path.join(self.cfg.ckpt_dir, "answer_type_classifier")
                self.type_classifier = AnswerTypeInference(ckpt)
                print("  AnswerTypeClassifier: loaded")
            except Exception as e:
                print(f"  AnswerTypeClassifier: skipped ({e})")

            try:
                ckpt = os.path.join(self.cfg.ckpt_dir, "verify_scorer")
                self.verify_scorer = VerifyScorerInference(ckpt)
                print("  VerifyScorer: loaded")
            except Exception as e:
                print(f"  VerifyScorer: skipped ({e})")

        # ── vLLM server ───────────────────────────────────────────────────────
        print("Starting vLLM server...")
        self._start_vllm_server()

        # ── Sandbox pool ──────────────────────────────────────────────────────
        print(f"Starting {self.cfg.workers} sandboxes...")
        self.sandbox_pool = queue.Queue()
        self._init_sandboxes()
        print("Ready.\n")

    def _start_vllm_server(self):
        import sys, subprocess
        cfg = self.cfg

        # Pre-load weights into page cache
        model_dir = cfg.model_path
        if os.path.isdir(model_dir):
            files = [str(p) for p in Path(model_dir).rglob("*.safetensors")]
            if not files:
                files = [str(p) for p in Path(model_dir).rglob("*.bin")]
            total_gb = sum(os.path.getsize(f) for f in files) / 1e9
            print(f"  Pre-loading {len(files)} weight files ({total_gb:.1f} GB)...")
            from concurrent.futures import ThreadPoolExecutor
            def _read(p):
                with open(p, "rb") as f:
                    while f.read(1 << 20):
                        pass
            with ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(_read, files))

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model",                  cfg.model_path,
            "--served-model-name",      cfg.served_model_name,
            "--tensor-parallel-size",   "1",
            "--max-num-seqs",           str(cfg.batch_size),
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
            "--host",                   "0.0.0.0",
            "--port",                   str(self.port),
            "--dtype",                  cfg.dtype,
            "--kv-cache-dtype",         cfg.kv_cache_dtype,
            "--max-model-len",          str(cfg.context_tokens),
            "--stream-interval",        str(cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
            "--disable-spec-decode",   # prevent vLLM from loading draft model from local path
        ]

        self._log_file    = open("vllm_server.log", "w")
        self._server_proc = subprocess.Popen(
            cmd, stdout=self._log_file, stderr=subprocess.STDOUT,
            start_new_session=True)

        from openai import OpenAI
        self.client = OpenAI(base_url=f"http://localhost:{self.port}/v1",
                             api_key="placeholder")

        print("  Waiting for vLLM server...", end="", flush=True)
        for i in range(cfg.server_timeout):
            if self._server_proc.poll() is not None:
                with open("vllm_server.log") as lf:
                    raise RuntimeError(f"vLLM server died:\n{lf.read()}")
            try:
                self.client.models.list()
                print(f" ready in {i}s")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("vLLM server did not start in time.")

    def _init_sandboxes(self):
        from llm import MathSandbox   # reuse existing sandbox class
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
            futs = [ex.submit(MathSandbox, self.cfg.jupyter_timeout)
                    for _ in range(self.cfg.workers)]
            for fut in as_completed(futs):
                try:
                    self.sandbox_pool.put(fut.result())
                except Exception as e:
                    print(f"  Sandbox init warning: {e}")

    def _get_forced_type(self, problem: str) -> Optional[str]:
        if self.type_classifier:
            try:
                return self.type_classifier.predict(problem)
            except Exception:
                pass
        return None

    def solve_problem(self, problem: str):
        """
        Solve one problem.
        Returns the best answer (int | float | str | etc.) or 0 as fallback.
        """
        print(f"\n{'─'*60}")
        print(f"Problem: {problem[:120]}{'...' if len(problem)>120 else ''}")

        # Time budget
        elapsed   = time.time() - self.notebook_start_time
        time_left = self.cfg.notebook_limit - elapsed
        reserved  = max(0, self.problems_remaining - 1) * self.cfg.base_problem_timeout
        budget    = min(time_left - reserved, self.cfg.high_problem_timeout)
        budget    = max(budget, self.cfg.base_problem_timeout)
        deadline  = time.time() + budget
        print(f"Budget: {budget:.0f}s")

        # Predict answer type
        forced_type = self._get_forced_type(problem)
        if forced_type:
            print(f"Predicted answer type: {forced_type}")

        stop_event = threading.Event()
        executor   = ThreadPoolExecutor(max_workers=self.cfg.attempts)
        results: list[TypedAnswer] = []

        def _attempt(idx):
            sandbox = None
            try:
                sandbox = self.sandbox_pool.get(timeout=self.cfg.sandbox_timeout)
                dispatcher = ToolDispatcher(db=self.db, sandbox=sandbox)
                return run_attempt(
                    problem, self.client, dispatcher, self.cfg,
                    attempt_idx=idx,
                    stop_event=stop_event,
                    deadline=deadline,
                    forced_type=forced_type,
                )
            except queue.Empty:
                return None
            except Exception as e:
                print(f"  [attempt {idx}] uncaught: {e}")
                return None
            finally:
                if sandbox is not None:
                    try:
                        sandbox.reset()
                    except Exception:
                        pass
                    self.sandbox_pool.put(sandbox)

        futures = {executor.submit(_attempt, i): i
                   for i in range(self.cfg.attempts)}

        try:
            for fut in as_completed(futures, timeout=budget):
                if stop_event.is_set():
                    break
                try:
                    ans = fut.result()
                    if ans is not None:
                        results.append(ans)
                        print(f"  attempt done: {ans}")
                        # Early stop: enough agreement
                        # Count equivalence groups
                        groups = []
                        for a in results:
                            placed = False
                            for g in groups:
                                if answers_match(a, g[0]):
                                    g.append(a)
                                    placed = True
                                    break
                            if not placed:
                                groups.append([a])
                        if groups and max(len(g) for g in groups) >= self.cfg.early_stop_votes:
                            stop_event.set()
                            break
                except Exception as e:
                    print(f"  attempt exception: {e}")
        finally:
            stop_event.set()
            executor.shutdown(wait=False, cancel_futures=True)
            self.problems_remaining = max(0, self.problems_remaining - 1)

        if not results:
            print("No answers found. Returning fallback 0.")
            return 0

        # Phase 5: select best answer
        best = select_best_answer(results, min_votes=2)
        if best is None:
            best = max(results, key=lambda x: x.confidence)

        print(f"Final answer: {best.value}  (type={best.answer_type}, "
              f"conf={best.confidence:.2f})")

        # Free memory between problems
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best.value

    def __del__(self):
        with contextlib.suppress(Exception):
            self._server_proc.terminate()
            self._server_proc.wait()
        with contextlib.suppress(Exception):
            self._log_file.close()
        while not self.sandbox_pool.empty():
            with contextlib.suppress(Exception):
                self.sandbox_pool.get_nowait().close()


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle competition entry point
# ─────────────────────────────────────────────────────────────────────────────

_solver: Optional[OlympiadSolver] = None


def get_solver() -> OlympiadSolver:
    global _solver
    if _solver is None:
        _solver = OlympiadSolver()
    return _solver


def predict(id_, question, answer=None):
    """Kaggle evaluation gateway compatible predict function."""
    import polars as pl

    id_value      = id_.item(0)
    question_text = question.item(0)

    solver = get_solver()
    result = solver.solve_problem(question_text)

    return pl.DataFrame({"id": [id_value], "answer": [result]})
