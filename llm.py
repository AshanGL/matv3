"""
llm.py
======
vLLM server lifecycle + OpenAI-compatible client wrappers.
Mirrors the pattern from the reference AIMO-3 Kaggle notebook:
  - Starts vLLM as a subprocess (OpenAI-compatible API server)
  - Uses openai client for completions
  - Manages a pool of Jupyter sandboxes for Python tool calls
  - Provides domain-aware system prompts per difficulty band
"""

import os
import re
import sys
import math
import time
import queue
import random
import threading
import subprocess
import contextlib
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import torch
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class SolverConfig:
    # Model
    served_model_name    = "gpt-oss"
    model_path           = "/kaggle/input/gpt-oss-120b/transformers/default/1"

    # Precision
    kv_cache_dtype       = "fp8_e4m3"
    dtype                = "auto"

    # Timeouts
    high_problem_timeout = 900
    base_problem_timeout = 300
    notebook_limit       = 17400
    server_timeout       = 180
    session_timeout      = 960
    jupyter_timeout      = 6
    sandbox_timeout      = 3

    # Generation
    stream_interval      = 200
    context_tokens       = 131072
    buffer_tokens        = 512
    search_tokens        = 32
    top_logprobs         = 5
    batch_size           = 128
    early_stop           = 5
    attempts             = 8
    workers              = 16
    turns                = 128
    seed                 = 42

    # Sampling
    gpu_memory_utilization = 0.96
    temperature           = 1.0
    min_p                 = 0.02

    # -----------------------------------------------------------------------
    # Domain + difficulty-aware system prompts
    # -----------------------------------------------------------------------
    # In SolverConfig, replace SYSTEM_PROMPTS with:
    SYSTEM_PROMPTS = {
        "easy": (
            "You are an expert mathematics solver.\n\n"
            "PROCESS:\n"
            "1. Analyze the problem.\n"
            "2. Write Python/SymPy code to solve it.\n"
            "3. Print all intermediate results.\n"
            "4. Give final answer as \\boxed{N}.\n\n"
            "Always use ```python\\n...\\n``` blocks.\n"
            "Answer must be a non-negative integer."
        ),
        "medium": (
            "You are a competition mathematics solver.\n\n"
            "PROCESS:\n"
            "1. Identify the key technique.\n"
            "2. Write Python/SymPy code to solve it.\n"
            "3. Verify your answer with a second check.\n"
            "4. Give final answer as \\boxed{N}.\n\n"
            "Always use ```python\\n...\\n``` blocks.\n"
            "Answer must be a non-negative integer."
        ),
        "hard": (
            "You are an advanced Olympiad mathematician.\n\n"
            "PROCESS:\n"
            "1. Identify key insight or technique.\n"
            "2. Write Python/SymPy code to solve and verify.\n"
            "3. Cross-check with a different approach.\n"
            "4. Give final answer as \\boxed{N}.\n\n"
            "Always use ```python\\n...\\n``` blocks.\n"
            "Use sympy.solve(), sympy.factor(), sympy.isprime() etc.\n"
            "Answer must be a non-negative integer."
        ),
        "olympiad": (
            "You are an IMO-level problem solver.\n\n"
            "PROCESS:\n"
            "1. ANALYZE: Identify domain, constraints, goal.\n"
            "2. PLAN: Choose technique (AM-GM, induction, modular, etc).\n"
            "3. CODE: Write Python/SymPy to solve and verify.\n"
            "4. VERIFY: Run a second independent check.\n"
            "5. ANSWER: Give final answer as \\boxed{N}.\n\n"
            "RULES:\n"
            "- Always write code in ```python\\n...\\n``` blocks.\n"
            "- Use SymPy for exact computation (never float approximations).\n"
            "- Always print() intermediate results.\n"
            "- Verify answer satisfies all constraints.\n"
            "- Answer must be a non-negative integer less than 1000000."
        ),
    }

    DOMAIN_HINTS = {
        "Algebra": (
            "This is an Algebra problem. Consider: polynomial factoring, "
            "Vieta's formulas, AM-GM / Cauchy-Schwarz inequalities, "
            "functional equations, or substitution."
        ),
        "Geometry": (
            "This is a Geometry problem. Consider: angle chasing, "
            "power of a point, radical axes, trigonometric identities, "
            "coordinate geometry, or inversive geometry."
        ),
        "Number Theory": (
            "This is a Number Theory problem. Consider: modular arithmetic, "
            "CRT, lifting the exponent, Zsygmondy, Diophantine equations, "
            "or prime factorization."
        ),
        "Discrete Mathematics": (
            "This is a Discrete Mathematics / Combinatorics problem. Consider: "
            "pigeonhole, inclusion-exclusion, generating functions, "
            "graph theory, or bijective proofs."
        ),
        "Applied Mathematics": (
            "This is an Applied Mathematics problem. Consider: probability, "
            "statistics, optimization, linear algebra, or differential equations."
        ),
        "Calculus": (
            "This is a Calculus problem. Use SymPy for symbolic integration, "
            "differentiation, limits, or series. Verify numerically."
        ),
        "Precalculus": (
            "This is a Precalculus problem. Use algebraic manipulation, "
            "trigonometric identities, or coordinate geometry."
        ),
        "Other": (
            "Analyze the problem structure carefully before choosing a technique."
        ),
    }

    ANSWER_ONLY_PROMPT = (
        "Solve the problem and output only \\boxed{answer}. No explanation."
    )

    tool_prompt = (
        "You may use Python. Write code in ```python\\n...\\n``` blocks. "
        "Prioritize SymPy for exact symbolic computation. "
        "Use print() to see results."
    )

    preference_prompt = (
        "Think step by step. Give your final answer as \\boxed{N} "
        "where N is a non-negative integer less than 1000."
    )


# ---------------------------------------------------------------------------
# Jupyter sandbox (isolated Python kernel for tool calls)
# ---------------------------------------------------------------------------

class MathSandbox:
    """Isolated Jupyter kernel for LLM-driven Python tool execution."""

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_ports(cls, n: int = 5) -> list[int]:
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + n))
            cls._next_port += n
            return ports

    def __init__(self, timeout: float = 6.0):
        from jupyter_client import KernelManager
        self._timeout    = timeout
        self._km         = None
        self._client     = None
        self._owns_kernel = False

        ports = self._get_ports()
        env   = os.environ.copy()
        env.update({
            "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "PYDEVD_WARN_EVALUATION_TIMEOUT": "0",
            "JUPYTER_PLATFORM_DIRS": "1",
            "PYTHONWARNINGS": "ignore",
            "MPLBACKEND": "Agg",
        })

        self._km = KernelManager()
        self._km.shell_port, self._km.iopub_port, self._km.stdin_port, \
            self._km.hb_port, self._km.control_port = ports

        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=30)
        self._owns_kernel = True

        self.execute(
            "import math, numpy, sympy, itertools, collections, mpmath\n"
            "mpmath.mp.dps = 64\n"
            "import sympy as sp\n"
        )

    def execute(self, code: str, timeout: float = None) -> str:
        client  = self._client
        timeout = timeout or self._timeout
        msg_id  = client.execute(code, store_history=True, allow_stdin=False, stop_on_error=False)

        stdout_parts, stderr_parts = [], []
        start = time.time()

        while True:
            if time.time() - start > timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Timed out after {timeout}s"
            try:
                msg = client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            mt      = msg.get("msg_type")
            content = msg.get("content", {})

            if mt == "stream":
                (stdout_parts if content.get("name") == "stdout" else stderr_parts).append(
                    content.get("text", ""))
            elif mt == "error":
                tb = re.sub(r"\x1b\[[0-9;]*m", "", "".join(content.get("traceback", [])))
                stderr_parts.append(tb)
            elif mt in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else text + "\n")
            elif mt == "status" and content.get("execution_state") == "idle":
                break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)
        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[WARN] No output. Use print() to display results."

    def reset(self):
        self.execute("%reset -f\nimport math, numpy, sympy, itertools, collections, mpmath\n"
                     "mpmath.mp.dps = 64\nimport sympy as sp\n")

    def close(self):
        with contextlib.suppress(Exception):
            if self._client:
                self._client.stop_channels()
        if self._owns_kernel and self._km:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()


# ---------------------------------------------------------------------------
# vLLM server manager
# ---------------------------------------------------------------------------

class VLLMServer:
    """Start and manage the vLLM OpenAI-compatible API server."""

    def __init__(self, cfg: SolverConfig = None, port: int = 8000):
        from openai import OpenAI
        self.cfg    = cfg or SolverConfig()
        self.port   = port
        self.client = OpenAI(base_url=f"http://0.0.0.0:{port}/v1", api_key="dummy")
        self.server_process = None
        self.log_file       = None

    def _preload_weights(self):
        """Page-cache the model weights before starting the server (faster cold start)."""
        import glob
        from concurrent.futures import ThreadPoolExecutor

        model_dir = self.cfg.model_path
        files = glob.glob(os.path.join(model_dir, "*.safetensors")) or \
                glob.glob(os.path.join(model_dir, "*.bin"))

        if not files:
            return

        total_size = sum(os.path.getsize(f) for f in files)
        print(f"Preloading {len(files)} weight files ({total_size/1e9:.2f} GB)...")
        start = time.time()

        def _read(path):
            with open(path, "rb") as f:
                while f.read(1 << 20):
                    pass

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_read, files))

        print(f"Preloaded in {time.time()-start:.1f}s\n")

    def start(self):
        self._preload_weights()

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--seed",                    str(self.cfg.seed),
            "--model",                   self.cfg.model_path,
            "--served-model-name",       self.cfg.served_model_name,
            "--tensor-parallel-size",    "1",
            "--max-num-seqs",            str(self.cfg.batch_size),
            "--gpu-memory-utilization",  str(self.cfg.gpu_memory_utilization),
            "--host",                    "0.0.0.0",
            "--port",                    str(self.port),
            "--dtype",                   self.cfg.dtype,
            "--kv-cache-dtype",          self.cfg.kv_cache_dtype,
            "--max-model-len",           str(self.cfg.context_tokens),
            "--stream-interval",         str(self.cfg.stream_interval),
            "--async-scheduling",
            "--disable-log-stats",
            "--enable-prefix-caching",
        ]

        self.log_file       = open("vllm_server.log", "w")
        self.server_process = subprocess.Popen(
            cmd, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)

        print("Waiting for vLLM server...")
        start = time.time()
        for _ in range(self.cfg.server_timeout):
            if self.server_process.poll() is not None:
                self.log_file.flush()
                with open("vllm_server.log") as lf:
                    raise RuntimeError(f"Server died:\n{lf.read()}")
            try:
                self.client.models.list()
                print(f"Server ready in {time.time()-start:.1f}s\n")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("vLLM server failed to start (timeout)")

    def stop(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
        if self.log_file:
            self.log_file.close()


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str):
    """Extract answer - only look after 'assistant' token to avoid reasoning contamination."""
    
    # Split at assistant token - only look at final answer part
    for split_token in ['assistant\n', 'assistant\r', 'assistantfinal', 'assistant']:
        if split_token in text.lower():
            idx = text.lower().rfind(split_token)
            text = text[idx:]
            break
    
    # Try patterns in order of reliability
    patterns = [
        r'\\boxed\s*\{\s*([0-9,]+)\s*\}',           # \boxed{123}
        r'\\boxed\s*\{\s*(-?[0-9]+)\s*\}',           # \boxed{-123}
        r'the\s+answer\s+is\s*[:\s]*([0-9,]+)',       # the answer is 123
        r'final\s+answer\s*[:\s=]*([0-9,]+)',         # final answer: 123
        r'=\s*\\boxed\s*\{\s*([0-9,]+)\s*\}',        # = \boxed{123}
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                val = int(str(matches[-1]).replace(",", ""))
                if 0 <= val <= 999999:
                    return val
            except ValueError:
                pass
    
    # Debug: show what boxed content was found
    all_boxed = re.findall(r'\\boxed\s*\{([^}]*)\}', text)
    if all_boxed:
        print(f"  [debug] boxed found but not integer: {all_boxed[:3]}")
    
    return None


def extract_python_blocks(text: str) -> list[str]:
    """Extract ```python ... ``` code blocks from model output."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def build_prompt(
    problem:    str,
    difficulty_band: str = "olympiad",
    domain:     str      = "Algebra",
    context:    str      = "",
    cfg:        SolverConfig = None,
) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt) tailored to difficulty + domain.
    Injects retrieved context if provided.
    """
    if cfg is None:
        cfg = SolverConfig()

    sys_prompt  = cfg.SYSTEM_PROMPTS.get(difficulty_band, cfg.SYSTEM_PROMPTS["olympiad"])
    domain_hint = cfg.DOMAIN_HINTS.get(domain, "")

    user_parts = []
    if domain_hint:
        user_parts.append(f"[Domain hint]: {domain_hint}")
    if context:
        user_parts.append(f"[Relevant context from similar problems]:\n{context}")
    user_parts.append(f"Problem:\n{problem}")
    user_parts.append(cfg.preference_prompt)

    user_prompt = "\n\n".join(user_parts)
    return sys_prompt, user_prompt


def call_llm_stream(client, system_prompt, user_prompt, cfg=None, seed=42):
    if cfg is None:
        cfg = SolverConfig()

    try:
        from openai_harmony import (
            load_harmony_encoding, HarmonyEncodingName,
            SystemContent, ReasoningEffort,
            Message, Role, Conversation
        )

        system_content = (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(ReasoningEffort.HIGH)
        )
        system_msg = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_msg   = Message.from_role_and_content(Role.USER, user_prompt)

        conversation   = Conversation(messages=[system_msg, user_msg])
        encoding       = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        stop_token_ids = encoding.stop_tokens_for_assistant_actions()

        # ← correct method
        prompt_token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        logprobs_buffer = []
        text_chunks     = []

        stream = client.completions.create(
            model      = cfg.served_model_name,
            prompt     = prompt_token_ids,
            temperature= cfg.temperature,
            max_tokens = 4096,
            seed       = seed,
            stream     = True,
            logprobs   = cfg.top_logprobs,
            extra_body = {
                "min_p":          cfg.min_p,
                "stop_token_ids": stop_token_ids,
            },
        )

        for chunk in stream:
            delta = chunk.choices[0].text
            if delta:
                text_chunks.append(delta)
            lp = chunk.choices[0].logprobs
            if lp and lp.top_logprobs:
                for tok_dict in lp.top_logprobs:
                    if tok_dict:
                        logprobs_buffer.append(tok_dict)

        stream.close()
        full_text = "".join(text_chunks)
        print(f"  [LLM raw]: {full_text[:300]}")
        return full_text, logprobs_buffer, len(text_chunks)

    except Exception as e:
        import traceback
        print(f"  [LLM ERROR]: {e}")
        traceback.print_exc()
        return "", [], 0


def compute_entropy(logprobs_buffer: list) -> float:
    if not logprobs_buffer:
        return float("inf")
    total, count = 0.0, 0
    for lp_dict in logprobs_buffer:
        if not isinstance(lp_dict, dict):
            continue
        for _, lp in lp_dict.items():
            p = math.exp(lp)
            if p > 0:
                total -= p * math.log2(p)
        count += 1
    return total / count if count else float("inf")


def run_python_tool(sandbox: MathSandbox, code: str) -> str:
    """Execute Python code in the sandbox and return output."""
    return sandbox.execute(code)
