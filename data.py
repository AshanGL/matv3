"""
data.py
=======
Dataset loading, cleaning, splitting, and per-domain database preparation
for the Olympiad Math Solver architecture.

Dataset: AshanGimhana/omni-math-clean  (4,428 samples)
Domains: Algebra, Geometry, Applied Mathematics, Discrete Mathematics,
         Number Theory, Calculus, Precalculus, Other

Storage strategy:
  - Per-domain FAISS indexes  (fast ANN retrieval)
  - Per-domain HuggingFace datasets saved to disk (structured records)
  - SQLite metadata DB        (fast filtering by difficulty / sub_path)
  - Sentence-transformer embeddings cached as .npy files
"""

import os
import json
import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID   = "AshanGimhana/omni-math-clean"
RANDOM_SEED  = 42

# All 8 main domains in the dataset
DOMAINS = [
    "Algebra",
    "Geometry",
    "Applied Mathematics",
    "Discrete Mathematics",
    "Number Theory",
    "Calculus",
    "Precalculus",
    "Other",
]

# Folder-safe names (used as directory names)
DOMAIN_SLUGS = {
    "Algebra":                "algebra",
    "Geometry":               "geometry",
    "Applied Mathematics":    "applied_mathematics",
    "Discrete Mathematics":   "discrete_mathematics",
    "Number Theory":          "number_theory",
    "Calculus":               "calculus",
    "Precalculus":            "precalculus",
    "Other":                  "other",
}

# Difficulty bands (maps float difficulty → string tier)
DIFFICULTY_BANDS = {
    (0.0, 3.0):  "easy",
    (3.0, 6.0):  "medium",
    (6.0, 8.0):  "hard",
    (8.0, 10.1): "olympiad",
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # fast, good quality


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def difficulty_to_band(score: float) -> str:
    for (lo, hi), label in DIFFICULTY_BANDS.items():
        if lo <= score < hi:
            return label
    return "olympiad"


def safe_slug(domain: str) -> str:
    return DOMAIN_SLUGS.get(domain, domain.lower().replace(" ", "_"))


# ---------------------------------------------------------------------------
# 1.  Load raw dataset from HuggingFace
# ---------------------------------------------------------------------------

def load_raw_dataset():
    """Download and return the full dataset as a HuggingFace Dataset."""
    from datasets import load_dataset
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train")
    print(f"  Total samples: {len(ds)}")
    return ds


# ---------------------------------------------------------------------------
# 2.  Clean + enrich each record
# ---------------------------------------------------------------------------

def clean_record(record: dict) -> dict:
    """Normalise fields and add derived columns."""
    difficulty = float(record.get("difficulty") or 5.0)
    sub_path   = record.get("sub_path") or []
    if isinstance(sub_path, str):
        try:
            sub_path = json.loads(sub_path)
        except Exception:
            sub_path = [sub_path]

    return {
        "problem":     (record.get("problem") or "").strip(),
        "solution":    (record.get("solution") or "").strip(),
        "answer":      str(record.get("answer") or "").strip(),
        "difficulty":  difficulty,
        "difficulty_band": difficulty_to_band(difficulty),
        "source":      record.get("source") or "",
        "main_domain": record.get("main_domain") or "Other",
        "sub_path":    sub_path,
        "full_path":   record.get("full_path") or "",
    }


def clean_dataset(ds):
    """Apply cleaning to every record; return a pandas DataFrame."""
    print("Cleaning and enriching records...")
    records = [clean_record(r) for r in ds]
    df = pd.DataFrame(records)
    # Drop rows without a problem or solution
    df = df[df["problem"].str.len() > 10].reset_index(drop=True)
    print(f"  Clean samples: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# 3.  Split  (train / val / test)
# ---------------------------------------------------------------------------

def split_dataset(df, train_frac=0.70, val_frac=0.15, seed=RANDOM_SEED):
    from sklearn.model_selection import train_test_split

    # Drop domains with fewer than 2 samples (can't stratify)
    counts = df['main_domain'].value_counts()
    df = df[df['main_domain'].isin(counts[counts >= 2].index)].reset_index(drop=True)

    assert train_frac + val_frac < 1.0
    test_frac = round(1.0 - train_frac - val_frac, 4)
    print(f"Splitting  train={train_frac:.0%}  val={val_frac:.0%}  test={test_frac:.0%}")

    train_df, tmp_df = train_test_split(
        df, test_size=(val_frac + test_frac),
        stratify=df['main_domain'], random_state=seed,
    )
    relative_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        tmp_df, test_size=(1.0 - relative_val),
        stratify=tmp_df['main_domain'], random_state=seed,
    )

    for name, part in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"  {name:5s}: {len(part):5d} samples")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
 




# ---------------------------------------------------------------------------
# 4.  Compute embeddings
# ---------------------------------------------------------------------------

def compute_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 256,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Encode a list of texts with a SentenceTransformer.
    Loads/saves from cache_path (.npy) to avoid recomputation.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    print(f"  Encoding {len(texts)} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot-product
        convert_to_numpy=True,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"  Saved embeddings → {cache_path}")

    return embeddings


# ---------------------------------------------------------------------------
# 5.  Build per-domain FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    """Inner-product index (for normalised vectors = cosine similarity)."""
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_faiss_index(index, path: str):
    import faiss
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str):
    import faiss
    return faiss.read_index(path)


# ---------------------------------------------------------------------------
# 6.  Build SQLite metadata database
# ---------------------------------------------------------------------------

def build_sqlite_db(df: pd.DataFrame, db_path: str):
    """
    Store all records in a SQLite database with indexed columns
    for fast filtered lookups (domain, difficulty_band, difficulty).
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS problems")
    cur.execute("""
        CREATE TABLE problems (
            id             INTEGER PRIMARY KEY,
            problem        TEXT,
            solution       TEXT,
            answer         TEXT,
            difficulty     REAL,
            difficulty_band TEXT,
            source         TEXT,
            main_domain    TEXT,
            sub_path       TEXT,
            full_path      TEXT,
            split          TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_domain     ON problems(main_domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_diff_band  ON problems(difficulty_band)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_difficulty ON problems(difficulty)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_split      ON problems(split)")

    rows = []
    for _, row in df.iterrows():
        rows.append((
            int(row.name),
            row["problem"],
            row["solution"],
            row["answer"],
            float(row["difficulty"]),
            row["difficulty_band"],
            row["source"],
            row["main_domain"],
            json.dumps(row["sub_path"]),
            row["full_path"],
            row.get("split", "train"),
        ))

    cur.executemany(
        "INSERT INTO problems VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"  SQLite DB saved → {db_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# 7.  Build per-domain databases
# ---------------------------------------------------------------------------

def build_domain_databases(
    train_df: pd.DataFrame,
    base_dir: str = "/kaggle/working/databases",
    embedding_model: str = EMBEDDING_MODEL,
):
    """
    For each domain:
      databases/<slug>/
        records.jsonl          — all training records for this domain
        embeddings.npy         — problem embeddings
        faiss.index            — FAISS inner-product index
        metadata.json          — domain stats
    """
    print("\n=== Building per-domain databases ===")

    for domain in DOMAINS:
        slug      = safe_slug(domain)
        domain_df = train_df[train_df["main_domain"] == domain].reset_index(drop=True)

        if len(domain_df) == 0:
            print(f"  Skipping {domain} — no samples")
            continue

        domain_dir = os.path.join(base_dir, slug)
        os.makedirs(domain_dir, exist_ok=True)

        print(f"\n  [{domain}]  {len(domain_df)} training samples")

        # 1. Save JSONL records
        records_path = os.path.join(domain_dir, "records.jsonl")
        with open(records_path, "w") as f:
            for _, row in domain_df.iterrows():
                f.write(json.dumps(row.to_dict()) + "\n")
        print(f"    records.jsonl saved")

        # 2. Compute + save embeddings
        emb_path = os.path.join(domain_dir, "embeddings.npy")
        embeddings = compute_embeddings(
            domain_df["problem"].tolist(),
            model_name=embedding_model,
            cache_path=emb_path,
        )

        # 3. Build + save FAISS index
        faiss_path = os.path.join(domain_dir, "faiss.index")
        index = build_faiss_index(embeddings)
        save_faiss_index(index, faiss_path)
        print(f"    FAISS index saved  ({index.ntotal} vectors, dim={embeddings.shape[1]})")

        # 4. Save domain metadata
        meta = {
            "domain":       domain,
            "slug":         slug,
            "n_samples":    len(domain_df),
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_model": embedding_model,
            "difficulty_distribution": {
                band: int((domain_df["difficulty_band"] == band).sum())
                for band in ["easy", "medium", "hard", "olympiad"]
            },
            "sub_paths": (
                domain_df["sub_path"]
                .explode()
                .dropna()
                .value_counts()
                .head(20)
                .to_dict()
            ),
        }
        with open(os.path.join(domain_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    print("\nDomain databases complete.")


# ---------------------------------------------------------------------------
# 8.  Retrieval helper used at inference time
# ---------------------------------------------------------------------------

class DomainRetriever:
    """
    Load a pre-built domain database and retrieve the top-k most
    similar problems to a query embedding.
    """

    def __init__(self, domain: str, base_dir: str = "/kaggle/working/databases"):
        slug       = safe_slug(domain)
        domain_dir = os.path.join(base_dir, slug)

        self.domain    = domain
        self.index     = load_faiss_index(os.path.join(domain_dir, "faiss.index"))
        self.records   = self._load_records(os.path.join(domain_dir, "records.jsonl"))

        with open(os.path.join(domain_dir, "metadata.json")) as f:
            self.meta = json.load(f)

    @staticmethod
    def _load_records(path: str) -> list[dict]:
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))
        return records

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        difficulty_band: Optional[str] = None,
    ) -> list[dict]:
        """
        Return top_k records most similar to query_embedding.
        Optionally filter by difficulty_band (post-filter).
        """
        query = query_embedding.astype(np.float32).reshape(1, -1)

        # Over-fetch if filtering, so we still return top_k after filter
        fetch_k = top_k * 5 if difficulty_band else top_k
        scores, indices = self.index.search(query, min(fetch_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rec = dict(self.records[idx])
            rec["similarity_score"] = float(score)
            if difficulty_band and rec.get("difficulty_band") != difficulty_band:
                continue
            results.append(rec)
            if len(results) >= top_k:
                break

        return results


# ---------------------------------------------------------------------------
# 9.  Global SQLite query helpers
# ---------------------------------------------------------------------------

class MetadataDB:
    """Thin wrapper for filtered SQL lookups across all domains."""

    def __init__(self, db_path: str = "/kaggle/working/databases/metadata.sqlite"):
        self.db_path = db_path

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_by_domain(self, domain: str, split: str = "train", limit: int = 100) -> list[dict]:
        return self.query(
            "SELECT * FROM problems WHERE main_domain=? AND split=? LIMIT ?",
            (domain, split, limit),
        )

    def get_by_difficulty(self, band: str, domain: Optional[str] = None, limit: int = 50) -> list[dict]:
        if domain:
            return self.query(
                "SELECT * FROM problems WHERE difficulty_band=? AND main_domain=? LIMIT ?",
                (band, domain, limit),
            )
        return self.query(
            "SELECT * FROM problems WHERE difficulty_band=? LIMIT ?",
            (band, limit),
        )


# ---------------------------------------------------------------------------
# 10. Master build function  (called from notebook)
# ---------------------------------------------------------------------------

def build_all(
    base_dir:    str   = "/kaggle/working/databases",
    train_frac:  float = 0.70,
    val_frac:    float = 0.15,
    save_splits: bool  = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline:
      1. Load raw dataset
      2. Clean
      3. Split (configurable fractions)
      4. Build per-domain FAISS databases (train set only)
      5. Build global SQLite metadata DB (all splits)
      6. Save split CSVs

    Returns (train_df, val_df, test_df).
    """
    ds       = load_raw_dataset()
    df       = clean_dataset(ds)
    train_df, val_df, test_df = split_dataset(df, train_frac, val_frac)

    # Tag split column for SQLite
    for part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        part["split"] = name

    # Build per-domain databases on training data only
    build_domain_databases(train_df, base_dir=base_dir)

    # Build global SQLite (all splits combined)
    all_df   = pd.concat([train_df, val_df, test_df], ignore_index=True)
    db_path  = os.path.join(base_dir, "metadata.sqlite")
    build_sqlite_db(all_df, db_path)

    # Save split CSVs
    if save_splits:
        splits_dir = os.path.join(base_dir, "splits")
        os.makedirs(splits_dir, exist_ok=True)
        train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(splits_dir,   "val.csv"),   index=False)
        test_df.to_csv(os.path.join(splits_dir,  "test.csv"),  index=False)
        print(f"\nSplit CSVs saved to {splits_dir}/")

    return train_df, val_df, test_df


if __name__ == "__main__":
    build_all()
