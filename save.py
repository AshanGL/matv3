"""
save.py
=======
Save and load utilities for every artefact the Olympiad Math Solver produces.

Artefact map
────────────
  checkpoints/
    type_difficulty_predictor/   ← PyTorch weights + HF tokenizer + config.json
    suggestion_classifier/
    logic_verifier/
    retrieval_encoder/

  databases/                     ← built from HuggingFace dataset (data.py)
    <domain_slug>/
      records.jsonl
      embeddings.npy
      faiss.index
      metadata.json
    metadata.sqlite
    splits/  train.csv  val.csv  test.csv

  ret_db/                        ← built from your PDFs (dom_db.py)
    manifest.json
    <domain_slug>/
      chunks.jsonl
      embeddings.npy
      faiss.index
      domain_stats.json

  pipeline_state.json            ← top-level manifest

All save functions are idempotent.
"""

from __future__ import annotations

import os
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DB_DIR      = "/kaggle/working/databases"
CKPT_DIR    = "/kaggle/working/checkpoints"
RET_DB_DIR  = "/kaggle/working/ret_db"
STATE_FILE  = "/kaggle/working/pipeline_state.json"

CHECKPOINT_NAMES = [
    "type_difficulty_predictor",
    "suggestion_classifier",
    "logic_verifier",
    "retrieval_encoder",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024:        return f"{n_bytes} B"
    if n_bytes < 1024**2:     return f"{n_bytes/1024:.1f} KB"
    if n_bytes < 1024**3:     return f"{n_bytes/1024**2:.1f} MB"
    return f"{n_bytes/1024**3:.2f} GB"


def _dir_size(path: str) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) \
        if os.path.isdir(path) else 0


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Checkpoint  save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    tokenizer,
    name:     str,
    extra:    dict = None,
    ckpt_dir: str  = CKPT_DIR,
) -> str:
    """
    Save a PyTorch model + HuggingFace tokenizer.

    Parameters
    ----------
    model     : nn.Module or HF PreTrainedModel
    tokenizer : HF tokenizer
    name      : checkpoint sub-folder name
    extra     : extra metadata written into config.json
    ckpt_dir  : root checkpoint directory

    Returns
    -------
    Absolute path to the saved checkpoint directory.
    """
    save_dir = os.path.join(ckpt_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    tokenizer.save_pretrained(save_dir)

    meta = {"name": name, "saved_at": _ts(), **(extra or {})}
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size = _fmt_size(_dir_size(save_dir))
    print(f"  checkpoint saved  ->  {save_dir}  ({size})")
    return save_dir


def load_checkpoint(
    name:     str,
    ckpt_dir: str = CKPT_DIR,
) -> tuple[dict, dict]:
    """
    Load raw checkpoint artefacts.
    Returns (state_dict, config_dict).
    Use the component classes in inference.py to get a live model.
    """
    ckpt_path = os.path.join(ckpt_dir, name)
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    weights_path = os.path.join(ckpt_path, "best_model.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(weights_path, map_location=device)

    config = {}
    cfg_path = os.path.join(ckpt_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            config = json.load(f)

    print(f"  checkpoint loaded  <-  {ckpt_path}")
    return state_dict, config


def checkpoint_exists(name: str, ckpt_dir: str = CKPT_DIR) -> bool:
    return os.path.isfile(os.path.join(ckpt_dir, name, "best_model.pt"))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training database  (HuggingFace dataset -> FAISS + SQLite)
# ─────────────────────────────────────────────────────────────────────────────

def save_training_db(
    train_df,
    val_df,
    test_df,
    base_dir: str  = DB_DIR,
    rebuild:  bool = False,
) -> None:
    """
    Persist the training database built from the HuggingFace dataset.

    This is the retrieval store used at training + inference time and is
    COMPLETELY SEPARATE from the PDF knowledge base (ret_db).
    """
    from data import build_domain_databases, build_sqlite_db
    import pandas as pd

    if rebuild and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"  Deleted existing training DB at {base_dir}")

    os.makedirs(base_dir, exist_ok=True)

    for df_, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        df_["split"] = name

    build_domain_databases(train_df, base_dir=base_dir)

    all_df  = pd.concat([train_df, val_df, test_df], ignore_index=True)
    db_path = os.path.join(base_dir, "metadata.sqlite")
    build_sqlite_db(all_df, db_path)

    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir,   "val.csv"),   index=False)
    test_df.to_csv(os.path.join(splits_dir,  "test.csv"),  index=False)

    size = _fmt_size(_dir_size(base_dir))
    print(f"  training DB saved  ->  {base_dir}  ({size})")


def training_db_exists(base_dir: str = DB_DIR) -> bool:
    return os.path.isfile(os.path.join(base_dir, "metadata.sqlite"))


def load_domain_metadata(domain: str, base_dir: str = DB_DIR) -> dict:
    slug = domain.lower().replace(" ", "_")
    path = os.path.join(base_dir, slug, "metadata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Domain metadata not found: {path}")
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PDF knowledge base  (ret_db)
# ─────────────────────────────────────────────────────────────────────────────

def save_doc_db(
    doc_folder: str,
    ret_db_dir: str  = RET_DB_DIR,
    force:      bool = False,
) -> "DomainDocDB":
    """
    Build and save the PDF knowledge base from doc_folder.

    Parameters
    ----------
    doc_folder : path to your doc/ folder (subject sub-folders + PDFs)
    ret_db_dir : where to save the database
    force      : if True, delete existing DB and rebuild

    Returns
    -------
    DomainDocDB instance ready for retrieval.
    """
    from dom_db import DomainDocDB
    db = DomainDocDB(ret_db_dir=ret_db_dir)
    if force:
        db.rebuild(doc_folder=doc_folder)
    else:
        db.build(doc_folder=doc_folder)
    return db


def load_doc_db(ret_db_dir: str = RET_DB_DIR) -> "DomainDocDB":
    """Load an already-built PDF knowledge base from disk."""
    from dom_db import load_domain_doc_db
    return load_domain_doc_db(ret_db_dir)


def doc_db_exists(ret_db_dir: str = RET_DB_DIR) -> bool:
    return os.path.isfile(os.path.join(ret_db_dir, "manifest.json"))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Pipeline state  (top-level manifest)
# ─────────────────────────────────────────────────────────────────────────────

def save_pipeline_state(
    extra:      dict = None,
    ckpt_dir:   str  = CKPT_DIR,
    db_dir:     str  = DB_DIR,
    ret_db_dir: str  = RET_DB_DIR,
    state_file: str  = STATE_FILE,
) -> dict:
    state = {
        "saved_at":    _ts(),
        "ckpt_dir":    ckpt_dir,
        "db_dir":      db_dir,
        "ret_db_dir":  ret_db_dir,
        "checkpoints": {n: checkpoint_exists(n, ckpt_dir) for n in CHECKPOINT_NAMES},
        "training_db": training_db_exists(db_dir),
        "doc_db":      doc_db_exists(ret_db_dir),
        **(extra or {}),
    }
    os.makedirs(os.path.dirname(state_file) or ".", exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  pipeline state saved  ->  {state_file}")
    return state


def load_pipeline_state(state_file: str = STATE_FILE) -> dict:
    if not os.path.isfile(state_file):
        raise FileNotFoundError(f"Pipeline state not found: {state_file}")
    with open(state_file) as f:
        state = json.load(f)
    print(f"  pipeline state loaded  <-  {state_file}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Status report
# ─────────────────────────────────────────────────────────────────────────────

def print_pipeline_status(
    ckpt_dir:   str = CKPT_DIR,
    db_dir:     str = DB_DIR,
    ret_db_dir: str = RET_DB_DIR,
) -> None:
    """Rich status table of everything built so far."""
    W = 62
    print()
    print("+" + "=" * W + "+")
    print("|" + "  OLYMPIAD SOLVER  --  PIPELINE STATUS".center(W) + "|")
    print("+" + "=" * W + "+")

    def row(label, ok, detail=""):
        icon  = "[OK]" if ok else "[--]"
        line  = f"  {icon}  {label:<30} {detail}"
        print("|" + f"{line:<{W}}" + "|")

    print("|" + "  Checkpoints".center(W) + "|")
    print("+" + "-" * W + "+")
    for name in CHECKPOINT_NAMES:
        ok     = checkpoint_exists(name, ckpt_dir)
        detail = _fmt_size(_dir_size(os.path.join(ckpt_dir, name))) if ok else "not trained yet"
        row(name, ok, detail)

    print("+" + "-" * W + "+")
    print("|" + "  Databases".center(W) + "|")
    print("+" + "-" * W + "+")

    ok_train = training_db_exists(db_dir)
    row("Training DB  (HuggingFace)", ok_train,
        _fmt_size(_dir_size(db_dir)) if ok_train else "not built yet")

    ok_doc = doc_db_exists(ret_db_dir)
    if ok_doc:
        try:
            m      = json.load(open(os.path.join(ret_db_dir, "manifest.json")))
            detail = f"{m.get('total_chunks',0):,} chunks  {_fmt_size(_dir_size(ret_db_dir))}"
        except Exception:
            detail = ""
    else:
        detail = "not built yet"
    row("Doc DB  (PDF knowledge base)", ok_doc, detail)

    print("+" + "=" * W + "+")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Master save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_all(
    train_df    = None,
    val_df      = None,
    test_df     = None,
    doc_folder: str  = None,
    ckpt_dir:   str  = CKPT_DIR,
    db_dir:     str  = DB_DIR,
    ret_db_dir: str  = RET_DB_DIR,
) -> dict:
    """
    Convenience wrapper: save whichever artefacts are provided.
    Checkpoint weights are saved automatically by train_*() in train.py.
    """
    print("\n=== save_all ===================================================")
    if train_df is not None and val_df is not None and test_df is not None:
        save_training_db(train_df, val_df, test_df, base_dir=db_dir)
    if doc_folder:
        save_doc_db(doc_folder=doc_folder, ret_db_dir=ret_db_dir)
    state = save_pipeline_state(ckpt_dir=ckpt_dir, db_dir=db_dir, ret_db_dir=ret_db_dir)
    print_pipeline_status(ckpt_dir, db_dir, ret_db_dir)
    return state


def load_all(
    ckpt_dir:    str  = CKPT_DIR,
    db_dir:      str  = DB_DIR,
    ret_db_dir:  str  = RET_DB_DIR,
    load_doc:    bool = True,
) -> dict:
    """
    Load all available artefacts.  Returns {"state": dict, "doc_db": DomainDocDB|None}.
    """
    result: dict = {}
    result["state"] = {}
    if os.path.exists(STATE_FILE):
        try:
            result["state"] = load_pipeline_state()
        except Exception:
            pass

    result["doc_db"] = None
    if load_doc and doc_db_exists(ret_db_dir):
        try:
            result["doc_db"] = load_doc_db(ret_db_dir)
        except Exception as e:
            print(f"  Warning: could not load doc_db: {e}")

    print_pipeline_status(ckpt_dir, db_dir, ret_db_dir)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Archive  (tar.gz snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def archive_run(
    archive_path:    str,
    ckpt_dir:        str  = CKPT_DIR,
    db_dir:          str  = DB_DIR,
    ret_db_dir:      str  = RET_DB_DIR,
    skip_embeddings: bool = False,
) -> str:
    """
    Create a compressed .tar.gz snapshot of the full pipeline.

    Parameters
    ----------
    archive_path    : destination, e.g. "/kaggle/working/run_v1.tar.gz"
    skip_embeddings : if True, skip .npy / .index files for a smaller archive
    """
    print(f"  Creating archive: {archive_path}")
    SKIP_EXTS = {".npy", ".index"} if skip_embeddings else set()

    with tarfile.open(archive_path, "w:gz") as tar:
        for path, arc_name in [
            (ckpt_dir,   "checkpoints"),
            (db_dir,     "databases"),
            (ret_db_dir, "ret_db"),
        ]:
            if not os.path.exists(path):
                continue
            for item in Path(path).rglob("*"):
                if not item.is_file():
                    continue
                if item.suffix in SKIP_EXTS:
                    continue
                arc = arc_name + str(item).replace(path, "")
                tar.add(item, arcname=arc)
        if os.path.isfile(STATE_FILE):
            tar.add(STATE_FILE, arcname="pipeline_state.json")

    size_mb = os.path.getsize(archive_path) / 1e6
    print(f"  Archive created  ->  {archive_path}  ({size_mb:.1f} MB)")
    return archive_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_pipeline_status()
