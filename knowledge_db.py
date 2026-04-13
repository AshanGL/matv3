"""
knowledge_db.py
===============
Unified knowledge database: problem store + theorem store.
Both stores use FAISS inner-product indexes over sentence-transformer embeddings.

Memory design
-------------
- FAISS indexes are loaded lazily on first query, never reloaded
- Embedding model is shared via tools.get_encoder() singleton
- Theorem store is tiny (< 1k entries) — loaded fully into RAM once
- Problem store records are loaded on demand from JSONL (not all into RAM)
- No GPU memory used here — CPU FAISS only

Directory layout
----------------
knowledge_db/
  problems/
    <domain_slug>/
      records.jsonl       ← one JSON per problem
      embeddings.npy      ← float32 embeddings, shape (N, dim)
      faiss.index         ← FAISS IndexFlatIP
      meta.json           ← domain stats
  theorems/
    records.jsonl         ← ALL theorems in one file
    embeddings.npy
    faiss.index
  manifest.json           ← build metadata
"""

from __future__ import annotations

import os
import json
import hashlib
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = [
    "Algebra", "Geometry", "Number Theory", "Discrete Mathematics",
    "Calculus", "Precalculus", "Applied Mathematics", "Other",
]

DOMAIN_SLUGS = {d: d.lower().replace(" ", "_") for d in DOMAINS}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Built-in theorem library covering all 8 domains
# Each entry: name, statement, domain, tags, when_to_apply
BUILTIN_THEOREMS = [
    # ── Algebra ──────────────────────────────────────────────────────────────
    {"name": "AM-GM Inequality",
     "statement": "For non-negative reals a₁,...,aₙ: (a₁+...+aₙ)/n ≥ (a₁·...·aₙ)^(1/n). Equality iff all aᵢ equal.",
     "domain": "Algebra", "tags": ["inequality", "optimization", "am-gm"],
     "when_to_apply": "Proving minimum/maximum of symmetric expressions involving products and sums."},
    {"name": "Vieta's Formulas",
     "statement": "For xⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₀ = 0 with roots r₁,...,rₙ: Σrᵢ = -aₙ₋₁, Πrᵢ = (-1)ⁿa₀.",
     "domain": "Algebra", "tags": ["polynomial", "roots", "vieta"],
     "when_to_apply": "Relating sums/products of roots to polynomial coefficients without finding roots explicitly."},
    {"name": "Cauchy-Schwarz Inequality",
     "statement": "(Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²). Equality iff aᵢ/bᵢ is constant.",
     "domain": "Algebra", "tags": ["inequality", "cauchy-schwarz"],
     "when_to_apply": "Bounding dot products; proving sum inequalities with two sequences."},
    {"name": "Polynomial Remainder Theorem",
     "statement": "The remainder of p(x) divided by (x-a) equals p(a).",
     "domain": "Algebra", "tags": ["polynomial", "remainder", "factor"],
     "when_to_apply": "Finding remainders or checking if (x-a) is a factor."},
    {"name": "Schur's Inequality",
     "statement": "For non-negative a,b,c and t>0: aᵗ(a-b)(a-c) + bᵗ(b-a)(b-c) + cᵗ(c-a)(c-b) ≥ 0.",
     "domain": "Algebra", "tags": ["inequality", "schur", "symmetric"],
     "when_to_apply": "Symmetric inequalities in three non-negative variables."},

    # ── Number Theory ─────────────────────────────────────────────────────────
    {"name": "Fermat's Little Theorem",
     "statement": "If p is prime and gcd(a,p)=1, then aᵖ⁻¹ ≡ 1 (mod p).",
     "domain": "Number Theory", "tags": ["modular", "prime", "fermat"],
     "when_to_apply": "Computing large powers modulo a prime; proving divisibility."},
    {"name": "Chinese Remainder Theorem",
     "statement": "If n₁,...,nₖ are pairwise coprime, the system x ≡ aᵢ (mod nᵢ) has a unique solution mod n₁·...·nₖ.",
     "domain": "Number Theory", "tags": ["crt", "modular", "congruence"],
     "when_to_apply": "Solving systems of modular equations with coprime moduli."},
    {"name": "Lifting the Exponent (LTE)",
     "statement": "For odd prime p, p|a-b, p∤a, p∤b: vₚ(aⁿ-bⁿ) = vₚ(a-b) + vₚ(n).",
     "domain": "Number Theory", "tags": ["lte", "p-adic", "divisibility"],
     "when_to_apply": "Finding the exact prime-power in aⁿ±bⁿ."},
    {"name": "Euler's Totient Theorem",
     "statement": "If gcd(a,n)=1, then a^φ(n) ≡ 1 (mod n), where φ(n) = n·Π(1-1/p).",
     "domain": "Number Theory", "tags": ["euler", "totient", "modular"],
     "when_to_apply": "Reducing large exponents modulo n when gcd(a,n)=1."},
    {"name": "Wilson's Theorem",
     "statement": "p is prime iff (p-1)! ≡ -1 (mod p).",
     "domain": "Number Theory", "tags": ["prime", "wilson", "factorial"],
     "when_to_apply": "Primality tests involving factorials mod p."},
    {"name": "Legendre's Formula",
     "statement": "The exponent of prime p in n! is Σ_{k≥1} ⌊n/pᵏ⌋.",
     "domain": "Number Theory", "tags": ["factorial", "prime", "legendre"],
     "when_to_apply": "Finding the power of a prime in n!."},

    # ── Geometry ─────────────────────────────────────────────────────────────
    {"name": "Power of a Point",
     "statement": "For point P and circle: PA·PB = PC·PD for any two chords/secants through P. Equals d²-r² for external point.",
     "domain": "Geometry", "tags": ["circle", "power", "chord", "tangent"],
     "when_to_apply": "Relating lengths of chords, secants, tangents from a common point."},
    {"name": "Ptolemy's Theorem",
     "statement": "For cyclic quadrilateral ABCD: AC·BD = AB·CD + AD·BC.",
     "domain": "Geometry", "tags": ["cyclic", "ptolemy", "quadrilateral"],
     "when_to_apply": "Length problems in cyclic quadrilaterals."},
    {"name": "Stewart's Theorem",
     "statement": "For triangle ABC with cevian AD: b²m + c²n = a(d² + mn), where a=m+n.",
     "domain": "Geometry", "tags": ["cevian", "stewart", "triangle", "length"],
     "when_to_apply": "Finding length of a cevian in a triangle."},
    {"name": "Ceva's Theorem",
     "statement": "Cevians AD, BE, CF of triangle ABC are concurrent iff (AF/FB)·(BD/DC)·(CE/EA) = 1.",
     "domain": "Geometry", "tags": ["cevian", "concurrent", "ceva"],
     "when_to_apply": "Proving three cevians meet at a point; finding ratios."},
    {"name": "Extended Law of Sines",
     "statement": "a/sin A = b/sin B = c/sin C = 2R, where R is the circumradius.",
     "domain": "Geometry", "tags": ["sine_rule", "circumradius", "triangle"],
     "when_to_apply": "Relating side lengths to angles and circumradius."},

    # ── Discrete Mathematics ──────────────────────────────────────────────────
    {"name": "Pigeonhole Principle",
     "statement": "If n+1 objects are placed in n boxes, at least one box contains ≥2 objects.",
     "domain": "Discrete Mathematics", "tags": ["pigeonhole", "existence"],
     "when_to_apply": "Proving existence of a collision, repetition, or concentration."},
    {"name": "Inclusion-Exclusion Principle",
     "statement": "|A₁∪...∪Aₙ| = Σ|Aᵢ| - Σ|Aᵢ∩Aⱼ| + ... + (-1)^(n+1)|A₁∩...∩Aₙ|.",
     "domain": "Discrete Mathematics", "tags": ["inclusion-exclusion", "counting"],
     "when_to_apply": "Counting elements satisfying at least one of several properties."},
    {"name": "Burnside's Lemma",
     "statement": "Number of distinct colorings = (1/|G|) Σ_{g∈G} |Fix(g)|.",
     "domain": "Discrete Mathematics", "tags": ["burnside", "symmetry", "coloring"],
     "when_to_apply": "Counting distinct objects under group symmetry."},
    {"name": "Stars and Bars",
     "statement": "Number of non-negative integer solutions to x₁+...+xₖ=n is C(n+k-1, k-1).",
     "domain": "Discrete Mathematics", "tags": ["counting", "partition", "stars-bars"],
     "when_to_apply": "Distributing identical objects into distinct bins."},
    {"name": "Lindström-Gessel-Viennot Lemma",
     "statement": "Number of non-intersecting lattice path systems equals a determinant of path counts.",
     "domain": "Discrete Mathematics", "tags": ["lgv", "paths", "determinant"],
     "when_to_apply": "Counting non-crossing paths; proving combinatorial identities."},

    # ── Calculus ──────────────────────────────────────────────────────────────
    {"name": "Fundamental Theorem of Calculus",
     "statement": "If F'=f, then ∫_a^b f(x)dx = F(b)-F(a). Also: d/dx ∫_a^x f(t)dt = f(x).",
     "domain": "Calculus", "tags": ["integral", "derivative", "ftc"],
     "when_to_apply": "Evaluating definite integrals; differentiating under the integral sign."},
    {"name": "L'Hôpital's Rule",
     "statement": "If lim f/g = 0/0 or ∞/∞, then lim f/g = lim f'/g' (if the latter exists).",
     "domain": "Calculus", "tags": ["limit", "hopital", "indeterminate"],
     "when_to_apply": "Evaluating indeterminate limits of the form 0/0 or ∞/∞."},
    {"name": "Taylor's Theorem",
     "statement": "f(x) = Σ_{k=0}^n f^(k)(a)/k! · (x-a)^k + R_n(x).",
     "domain": "Calculus", "tags": ["taylor", "series", "approximation"],
     "when_to_apply": "Approximating functions near a point; proving inequalities via series."},

    # ── Applied Mathematics / Probability ─────────────────────────────────────
    {"name": "Bayes' Theorem",
     "statement": "P(A|B) = P(B|A)·P(A) / P(B).",
     "domain": "Applied Mathematics", "tags": ["probability", "bayes", "conditional"],
     "when_to_apply": "Updating probabilities given new evidence."},
    {"name": "Linearity of Expectation",
     "statement": "E[X+Y] = E[X] + E[Y] for any random variables X, Y (no independence needed).",
     "domain": "Applied Mathematics", "tags": ["expectation", "probability", "linearity"],
     "when_to_apply": "Computing expected values of sums; counting problems with indicator variables."},
    {"name": "Markov's Inequality",
     "statement": "For non-negative X and a>0: P(X≥a) ≤ E[X]/a.",
     "domain": "Applied Mathematics", "tags": ["probability", "markov", "bound"],
     "when_to_apply": "Upper-bounding tail probabilities."},
]


# ─────────────────────────────────────────────────────────────────────────────
# FAISS helpers  (CPU only — no GPU memory used)
# ─────────────────────────────────────────────────────────────────────────────

def _build_faiss_cpu(embeddings: np.ndarray):
    import faiss
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # inner product = cosine for L2-normed vecs
    index.add(embeddings.astype(np.float32))
    return index


def _save_faiss(index, path: str):
    import faiss
    faiss.write_index(index, path)


def _load_faiss(path: str):
    import faiss
    return faiss.read_index(path)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper  (reuses singleton from tools.py)
# ─────────────────────────────────────────────────────────────────────────────

def _embed(texts: list[str], model_name: str = EMBEDDING_MODEL,
           batch_size: int = 128) -> np.ndarray:
    from tools import get_encoder
    enc = get_encoder(model_name)
    return enc.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 200,
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeDB
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeDB:
    """
    Unified knowledge database with two stores:
      - Problem store : per-domain FAISS + JSONL records
      - Theorem store : single FAISS + JSONL

    Memory profile
    --------------
    - Each domain FAISS index: ~N × dim × 4 bytes  (MiniLM dim=384)
      For N=500 problems: 500×384×4 = ~750 KB per domain — negligible
    - Theorem index: ~30 theorems × 384 × 4 = ~45 KB
    - Records are read from JSONL on demand — not held in RAM
    - Total memory: < 20 MB for the full DB
    """

    MANIFEST = "manifest.json"

    def __init__(self, db_dir: str = "/kaggle/working/knowledge_db",
                 model_name: str = EMBEDDING_MODEL):
        self.db_dir     = db_dir
        self.model_name = model_name

        # Lazy-loaded caches
        self._prob_indexes:  dict[str, object]     = {}   # slug → faiss index
        self._prob_records:  dict[str, list[dict]] = {}   # slug → records list
        self._theo_index  = None
        self._theo_records: list[dict]             = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_dataframe(
        self,
        train_df,
        force: bool = False,
    ) -> None:
        """
        Build the problem store from a pandas DataFrame.
        Expected columns: problem, solution, answer, answer_type,
                          main_domain, difficulty_band, technique_tags, source
        """
        import pandas as pd

        prob_root = os.path.join(self.db_dir, "problems")
        if os.path.exists(prob_root) and not force:
            print("  Problem store already exists. Pass force=True to rebuild.")
        else:
            self._build_problem_store(train_df, prob_root)

        theo_root = os.path.join(self.db_dir, "theorems")
        if os.path.exists(theo_root) and not force:
            print("  Theorem store already exists.")
        else:
            self._build_theorem_store(theo_root)

        self._write_manifest()

    def _build_problem_store(self, df, prob_root: str) -> None:
        import pandas as pd
        os.makedirs(prob_root, exist_ok=True)
        print("\n=== Building problem store ===")

        for domain in DOMAINS:
            slug     = DOMAIN_SLUGS[domain]
            dom_df   = df[df["main_domain"] == domain].reset_index(drop=True)
            if len(dom_df) == 0:
                continue

            domain_dir = os.path.join(prob_root, slug)
            os.makedirs(domain_dir, exist_ok=True)

            # Build records
            records = []
            for _, row in dom_df.iterrows():
                tags = row.get("technique_tags", [])
                if isinstance(tags, str):
                    try:    tags = json.loads(tags)
                    except: tags = [tags]
                records.append({
                    "problem":        str(row.get("problem", "")),
                    "solution_sketch": str(row.get("solution", ""))[:500],
                    "answer":         str(row.get("answer", "")),
                    "answer_type":    str(row.get("answer_type", "string")),
                    "domain":         domain,
                    "difficulty_band": str(row.get("difficulty_band", "olympiad")),
                    "technique_tags": tags if isinstance(tags, list) else [],
                    "source":         str(row.get("source", "")),
                })

            # Save JSONL
            jsonl_path = os.path.join(domain_dir, "records.jsonl")
            with open(jsonl_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # Embed problems
            print(f"  [{domain}] {len(records)} problems — embedding...", end="", flush=True)
            texts = [r["problem"] for r in records]
            embs  = _embed(texts, self.model_name)

            # Save embeddings
            np.save(os.path.join(domain_dir, "embeddings.npy"), embs)

            # Build and save FAISS
            index = _build_faiss_cpu(embs)
            _save_faiss(index, os.path.join(domain_dir, "faiss.index"))

            # Save meta
            meta = {"domain": domain, "slug": slug, "n_problems": len(records),
                    "embedding_dim": int(embs.shape[1]),
                    "embedding_model": self.model_name}
            with open(os.path.join(domain_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  done ({embs.shape[1]}d)")

        print("Problem store built.\n")

    def _build_theorem_store(self, theo_root: str) -> None:
        os.makedirs(theo_root, exist_ok=True)
        print("=== Building theorem store ===")

        records = BUILTIN_THEOREMS  # extend with custom theorems here
        # Embed on the "name + when_to_apply" text — optimised for retrieval
        texts = [f"{r['name']}: {r['when_to_apply']}" for r in records]

        print(f"  {len(records)} theorems — embedding...", end="", flush=True)
        embs = _embed(texts, self.model_name)

        with open(os.path.join(theo_root, "records.jsonl"), "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        np.save(os.path.join(theo_root, "embeddings.npy"), embs)

        index = _build_faiss_cpu(embs)
        _save_faiss(index, os.path.join(theo_root, "faiss.index"))

        print(f"  done")
        print("Theorem store built.\n")

    def _write_manifest(self) -> None:
        manifest = {
            "built_at":    datetime.now(timezone.utc).isoformat(),
            "db_dir":      self.db_dir,
            "model":       self.model_name,
            "n_theorems":  len(BUILTIN_THEOREMS),
        }
        with open(os.path.join(self.db_dir, self.MANIFEST), "w") as f:
            json.dump(manifest, f, indent=2)

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _load_prob_domain(self, slug: str) -> bool:
        if slug in self._prob_indexes:
            return True
        domain_dir = os.path.join(self.db_dir, "problems", slug)
        faiss_path = os.path.join(domain_dir, "faiss.index")
        jsonl_path = os.path.join(domain_dir, "records.jsonl")
        if not os.path.exists(faiss_path):
            return False
        self._prob_indexes[slug] = _load_faiss(faiss_path)
        records = []
        with open(jsonl_path) as f:
            for line in f:
                records.append(json.loads(line))
        self._prob_records[slug] = records
        return True

    def _load_theorems(self) -> bool:
        if self._theo_index is not None:
            return True
        theo_dir   = os.path.join(self.db_dir, "theorems")
        faiss_path = os.path.join(theo_dir, "faiss.index")
        jsonl_path = os.path.join(theo_dir, "records.jsonl")
        if not os.path.exists(faiss_path):
            return False
        self._theo_index = _load_faiss(faiss_path)
        self._theo_records = []
        with open(jsonl_path) as f:
            for line in f:
                self._theo_records.append(json.loads(line))
        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search_problems(
        self,
        query_emb: np.ndarray,
        domain:    Optional[str] = None,
        top_k:     int           = 4,
        min_sim:   float         = 0.25,   # lowered from old 0.5 threshold
    ) -> list[dict]:
        """
        Return top_k most similar problems.
        If domain is given, only search that domain's index.
        Otherwise search all domains and merge.
        """
        query = query_emb.astype(np.float32).reshape(1, -1)

        if domain is not None:
            slug = DOMAIN_SLUGS.get(domain, domain.lower().replace(" ", "_"))
            return self._search_one_domain(slug, query, top_k, min_sim)

        # Search all domains, merge, re-rank
        all_results = []
        for slug in DOMAIN_SLUGS.values():
            all_results.extend(self._search_one_domain(slug, query, top_k, min_sim))

        all_results.sort(key=lambda r: r["similarity"], reverse=True)
        # Deduplicate by problem text prefix
        seen, deduped = set(), []
        for r in all_results:
            key = r["problem"][:80]
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped[:top_k]

    def _search_one_domain(
        self, slug: str, query: np.ndarray, top_k: int, min_sim: float
    ) -> list[dict]:
        if not self._load_prob_domain(slug):
            return []
        index   = self._prob_indexes[slug]
        records = self._prob_records[slug]
        n       = min(top_k * 3, index.ntotal)
        if n == 0:
            return []
        scores, idxs = index.search(query, n)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or float(score) < min_sim:
                continue
            rec = dict(records[int(idx)])
            rec["similarity"] = round(float(score), 4)
            # Only return technique hints, not full solution — avoids copying
            rec.pop("solution_sketch", None)
            results.append(rec)
            if len(results) >= top_k:
                break
        return results

    def search_theorems(
        self,
        query_emb: np.ndarray,
        domain:    Optional[str] = None,
        top_k:     int           = 4,
        min_sim:   float         = 0.15,
    ) -> list[dict]:
        """Return top_k most relevant theorems."""
        if not self._load_theorems():
            return []

        query    = query_emb.astype(np.float32).reshape(1, -1)
        n        = min(top_k * 3, self._theo_index.ntotal)
        scores, idxs = self._theo_index.search(query, n)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or float(score) < min_sim:
                continue
            rec = dict(self._theo_records[int(idx)])
            rec["similarity"] = round(float(score), 4)
            if domain and rec.get("domain") != domain:
                # Soft filter — don't hard-exclude, just deprioritise
                rec["similarity"] *= 0.8
            results.append(rec)

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:top_k]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def is_built(self) -> bool:
        return os.path.exists(os.path.join(self.db_dir, self.MANIFEST))

    def add_theorem(self, theorem: dict, rebuild_index: bool = True) -> None:
        """Add a single custom theorem and optionally rebuild the theorem index."""
        theo_dir   = os.path.join(self.db_dir, "theorems")
        jsonl_path = os.path.join(theo_dir, "records.jsonl")
        os.makedirs(theo_dir, exist_ok=True)

        with open(jsonl_path, "a") as f:
            f.write(json.dumps(theorem, ensure_ascii=False) + "\n")

        if rebuild_index:
            records = []
            with open(jsonl_path) as f:
                for line in f:
                    records.append(json.loads(line))
            texts = [f"{r['name']}: {r['when_to_apply']}" for r in records]
            embs  = _embed(texts, self.model_name)
            np.save(os.path.join(theo_dir, "embeddings.npy"), embs)
            index = _build_faiss_cpu(embs)
            _save_faiss(index, os.path.join(theo_dir, "faiss.index"))
            # Invalidate cache
            self._theo_index   = None
            self._theo_records = []

    def status(self) -> dict:
        info = {"db_dir": self.db_dir, "is_built": self.is_built(), "domains": {}}
        prob_root = os.path.join(self.db_dir, "problems")
        for domain, slug in DOMAIN_SLUGS.items():
            domain_dir = os.path.join(prob_root, slug)
            meta_path  = os.path.join(domain_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    info["domains"][domain] = json.load(f)
        theo_dir = os.path.join(self.db_dir, "theorems")
        info["theorems"] = len(self._theo_records) or (
            sum(1 for _ in open(os.path.join(theo_dir, "records.jsonl")))
            if os.path.exists(os.path.join(theo_dir, "records.jsonl")) else 0
        )
        return info

    def print_status(self) -> None:
        s = self.status()
        print(f"\n{'='*52}")
        print(f"  KnowledgeDB  —  {self.db_dir}")
        print(f"{'='*52}")
        for domain, meta in s["domains"].items():
            print(f"  {domain:<25} {meta.get('n_problems',0):>5} problems")
        print(f"  {'Theorems':<25} {s['theorems']:>5}")
        print(f"{'='*52}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Auto-label answer_type from existing data
# ─────────────────────────────────────────────────────────────────────────────

def auto_label_answer_type(answer_str: str) -> str:
    """
    Infer answer_type from a stored answer string.
    Used to enrich training data that doesn't have explicit type labels.
    """
    import re
    s = str(answer_str).strip()
    if re.fullmatch(r"-?\d+", s):                           return "integer"
    if re.fullmatch(r"-?\d+\.\d+", s):                     return "float"
    if re.search(r"(-?\d+)\s*/\s*(-?\d+)", s):             return "fraction"
    if s.startswith("{") or (", " in s and s[0].isdigit()): return "set"
    if re.search(r"[a-zA-Z\^\\]", s):                      return "expression"
    return "string"


def enrich_dataframe(df):
    """Add answer_type column to a DataFrame if missing."""
    if "answer_type" not in df.columns:
        df = df.copy()
        df["answer_type"] = df["answer"].apply(
            lambda x: auto_label_answer_type(str(x)))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    db = KnowledgeDB(db_dir="./test_knowledge_db")
    if db.is_built():
        db.print_status()
    else:
        print("DB not built. Run build_from_dataframe() in the notebook first.")
