"""
train_new.py
============
Training pipeline for all trainable components in the tool-based solver.

Components
----------
1. AnswerTypeClassifier     — problem text → answer type (5-class)
2. RetrievalEncoder         — fine-tune sentence encoder on technique-aware pairs
3. VerifyScorer             — (problem, tool_trace) → is_correct (binary)
4. VoteRanker               — (problem, candidate_answer) → quality score

Memory / OOM prevention
-----------------------
- gradient_checkpointing enabled on all transformer encoders
- fp16 / bf16 mixed precision (auto-detected)
- DataLoader num_workers=2 (not 4) to limit CPU RAM
- Explicit torch.cuda.empty_cache() after each epoch
- Batch sizes are conservative — tunable per GPU
- Model weights deleted from RAM immediately after save
- No global model references kept — each train_* function is self-contained
"""

from __future__ import annotations

import os
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "/kaggle/working/checkpoints"
SEED     = 42

# Use bf16 on Ampere+ (A100, H100); fp16 elsewhere; fp32 on CPU
def _get_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

AMP_DTYPE = _get_dtype()

ANSWER_TYPES = ["integer", "float", "fraction", "expression", "set", "string"]
TYPE2ID      = {t: i for i, t in enumerate(ANSWER_TYPES)}

ENCODER_NAME = "microsoft/deberta-v3-base"   # default; overridable


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _free_memory():
    """Release all GPU + CPU caches between training runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ─────────────────────────────────────────────────────────────────────────────
# Component 1: AnswerTypeClassifier
# ─────────────────────────────────────────────────────────────────────────────

class AnswerTypeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len: int = 192):
        self.texts   = df["problem"].tolist()
        self.labels  = [TYPE2ID.get(str(t), 5) for t in df["answer_type"]]
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


class AnswerTypeModel(nn.Module):
    def __init__(self, encoder_name: str = ENCODER_NAME, n_classes: int = 6):
        super().__init__()
        from transformers import AutoModel
        self.encoder   = AutoModel.from_pretrained(encoder_name)
        # Enable gradient checkpointing to save ~40% GPU memory
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        h              = self.encoder.config.hidden_size
        self.head      = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(h, 256),
            nn.GELU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)


def train_answer_type_classifier(
    train_df,
    val_df,
    encoder_name: str  = ENCODER_NAME,
    epochs:       int  = 15,
    lr:           float = 2e-5,
    batch_size:   int  = 16,    # conservative for memory
    save_dir:     str  = None,
) -> str:
    """
    Train the AnswerTypeClassifier.
    Returns path to saved checkpoint.
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from knowledge_db import enrich_dataframe

    save_dir = save_dir or os.path.join(CKPT_DIR, "answer_type_classifier")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Training AnswerTypeClassifier ({encoder_name}) ===")
    set_seed()
    _free_memory()

    # Ensure answer_type column exists
    train_df = enrich_dataframe(train_df)
    val_df   = enrich_dataframe(val_df)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model     = AnswerTypeModel(encoder_name, n_classes=len(ANSWER_TYPES)).to(DEVICE)

    train_dl = DataLoader(
        AnswerTypeDataset(train_df, tokenizer),
        batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_dl = DataLoader(
        AnswerTypeDataset(val_df, tokenizer),
        batch_size=batch_size * 2, shuffle=False, num_workers=2,
    )

    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                logits = model(batch["input_ids"].to(DEVICE),
                               batch["attention_mask"].to(DEVICE))
                loss   = criterion(logits, batch["label"].to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = total = 0
        per_class = {t: {"tp": 0, "total": 0} for t in ANSWER_TYPES}
        with torch.no_grad():
            for batch in val_dl:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                    logits = model(batch["input_ids"].to(DEVICE),
                                   batch["attention_mask"].to(DEVICE))
                preds  = logits.argmax(-1).cpu()
                labels = batch["label"]
                correct += (preds == labels).sum().item()
                total   += len(labels)
                for p, l in zip(preds.tolist(), labels.tolist()):
                    t = ANSWER_TYPES[l]
                    per_class[t]["total"] += 1
                    if p == l:
                        per_class[t]["tp"] += 1

        acc = correct / max(total, 1)
        print(f"  Epoch {epoch}/{epochs}  "
              f"loss={total_loss/len(train_dl):.4f}  val_acc={acc:.3f}")
        for t, c in per_class.items():
            if c["total"] > 0:
                print(f"    {t:<12}  {c['tp']}/{c['total']}  "
                      f"({100*c['tp']/c['total']:.0f}%)")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump({"encoder_name": encoder_name,
                           "answer_types": ANSWER_TYPES,
                           "best_val_acc": best_acc}, f, indent=2)
            print(f"    Saved (val_acc={best_acc:.3f})")

        _free_memory()

    # Delete model from RAM after training
    del model
    _free_memory()
    print(f"AnswerTypeClassifier saved → {save_dir}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# Component 2: RetrievalEncoder (fine-tune sentence encoder)
# ─────────────────────────────────────────────────────────────────────────────

def _build_retrieval_pairs(df):
    """
    Build positive/negative pairs for contrastive training.
    Positive: problems sharing technique_tags
    Hard negative: same domain, different technique_tags
    """
    import ast

    def parse_tags(t):
        if isinstance(t, list):  return set(t)
        try:    return set(ast.literal_eval(str(t)))
        except: return {str(t)} if t else set()

    records = []
    for _, row in df.iterrows():
        records.append({
            "problem": str(row["problem"]),
            "domain":  str(row.get("main_domain", "Other")),
            "tags":    parse_tags(row.get("technique_tags", [])),
        })

    pairs = []
    domain_groups: dict[str, list] = {}
    for r in records:
        domain_groups.setdefault(r["domain"], []).append(r)

    for domain, group in domain_groups.items():
        if len(group) < 2:
            continue
        for i, anchor in enumerate(group):
            if not anchor["tags"]:
                continue
            # Find a positive: same tags overlap
            positives = [r for j, r in enumerate(group)
                         if j != i and anchor["tags"] & r["tags"]]
            # Hard negative: same domain, no tag overlap
            hard_negs = [r for j, r in enumerate(group)
                         if j != i and not (anchor["tags"] & r["tags"])]
            if positives:
                pos = random.choice(positives)
                neg = random.choice(hard_negs) if hard_negs else random.choice(
                    [r for r in records if r["domain"] != domain])
                pairs.append((anchor["problem"], pos["problem"], neg["problem"]))

    random.shuffle(pairs)
    return pairs


class TripletDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str, str]], tokenizer, max_len: int = 192):
        self.pairs   = pairs
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.pairs)

    def _enc(self, text: str):
        return self.tok(text, max_length=self.max_len, padding="max_length",
                        truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        a, p, n = self.pairs[idx]
        ea, ep, en_ = self._enc(a), self._enc(p), self._enc(n)
        return {
            "a_ids":  ea["input_ids"].squeeze(0),
            "a_mask": ea["attention_mask"].squeeze(0),
            "p_ids":  ep["input_ids"].squeeze(0),
            "p_mask": ep["attention_mask"].squeeze(0),
            "n_ids":  en_["input_ids"].squeeze(0),
            "n_mask": en_["attention_mask"].squeeze(0),
        }


def train_retrieval_encoder(
    train_df,
    encoder_name: str   = "sentence-transformers/all-MiniLM-L6-v2",
    epochs:       int   = 10,
    lr:           float = 1e-5,
    batch_size:   int   = 32,
    margin:       float = 0.3,
    save_dir:     str   = None,
) -> str:
    """
    Fine-tune the retrieval encoder with triplet loss on technique-aware pairs.
    Returns path to saved checkpoint.
    """
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

    save_dir = save_dir or os.path.join(CKPT_DIR, "retrieval_encoder")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Training RetrievalEncoder ({encoder_name}) ===")
    set_seed()
    _free_memory()

    pairs = _build_retrieval_pairs(train_df)
    if not pairs:
        print("  WARNING: No training pairs found — technique_tags missing.")
        print("  Skipping retrieval encoder training.")
        return save_dir

    print(f"  {len(pairs)} triplet pairs built")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    class TripletModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        def _pool(self, ids, mask):
            out = self.encoder(input_ids=ids, attention_mask=mask)
            # Mean pooling (standard for sentence transformers)
            token_embeddings = out.last_hidden_state
            input_mask_expanded = mask.unsqueeze(-1).float()
            return (token_embeddings * input_mask_expanded).sum(1) / \
                   input_mask_expanded.sum(1).clamp(min=1e-9)
        def forward(self, a_ids, a_mask, p_ids, p_mask, n_ids, n_mask):
            ea = self._pool(a_ids, a_mask)
            ep = self._pool(p_ids, p_mask)
            en = self._pool(n_ids, n_mask)
            return ea, ep, en

    model = TripletModel().to(DEVICE)
    ds    = TripletDataset(pairs, tokenizer)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dl) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
    scaler       = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in dl:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                ea, ep, en = model(
                    batch["a_ids"].to(DEVICE), batch["a_mask"].to(DEVICE),
                    batch["p_ids"].to(DEVICE), batch["p_mask"].to(DEVICE),
                    batch["n_ids"].to(DEVICE), batch["n_mask"].to(DEVICE),
                )
                loss = triplet_loss(ea, ep, en)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg = total_loss / len(dl)
        print(f"  Epoch {epoch}/{epochs}  triplet_loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            # Save just the encoder weights (not the triplet wrapper)
            model.encoder.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump({"encoder_name": encoder_name,
                           "best_triplet_loss": best_loss}, f, indent=2)
            print(f"    Saved (triplet_loss={best_loss:.4f})")

        _free_memory()

    del model
    _free_memory()
    print(f"RetrievalEncoder saved → {save_dir}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# Component 3: VerifyScorer
# ─────────────────────────────────────────────────────────────────────────────

class VerifyDataset(Dataset):
    """
    Items: (problem_text, solve_trace, label)
    label=1 if the final answer in the trace is correct.

    solve_trace is a string summarising what tool calls were made and their results.
    """
    def __init__(self, items: list[tuple[str, str, int]], tokenizer, max_len: int = 384):
        self.items   = items
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        problem, trace, label = self.items[idx]
        enc = self.tok(
            problem, trace,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def _make_verify_items(df) -> list[tuple[str, str, int]]:
    """
    Create training items from a DataFrame with columns:
      problem, answer (ground truth), solution (or solution_sketch)

    Positive: problem + correct solution summary → label 1
    Negative: problem + perturbed solution       → label 0
    """
    items = []
    for _, row in df.iterrows():
        prob = str(row.get("problem", ""))
        sol  = str(row.get("solution", row.get("solution_sketch", "")))
        ans  = str(row.get("answer", ""))
        if not prob or not sol:
            continue
        # Positive: correct trace
        trace_pos = f"Approach: standard method. Result: {ans}. Solution: {sol[:300]}"
        items.append((prob, trace_pos, 1))
        # Negative: wrong answer (offset by a small random amount)
        try:
            wrong = str(int(float(ans)) + random.choice([-1, 1, 2, -2, 10]))
        except Exception:
            wrong = ans + "_wrong"
        trace_neg = f"Approach: failed computation. Result: {wrong}. Solution: {_perturb(sol)[:300]}"
        items.append((prob, trace_neg, 0))

    random.shuffle(items)
    return items


def _perturb(text: str) -> str:
    """Simple perturbation for negative training examples."""
    lines = text.split("\n")
    if len(lines) > 2:
        i = random.randint(0, len(lines) - 2)
        lines[i], lines[i + 1] = lines[i + 1], lines[i]
    for op, opp in [("+", "-"), ("×", "÷"), ("=", "≠")]:
        if op in text:
            return "\n".join(lines).replace(op, opp, 1)
    return "\n".join(lines)


def train_verify_scorer(
    train_df,
    val_df,
    encoder_name: str   = ENCODER_NAME,
    epochs:       int   = 10,
    lr:           float = 1e-5,
    batch_size:   int   = 8,     # small — 384-token pairs are heavy
    save_dir:     str   = None,
) -> str:
    """Train the VerifyScorer. Returns checkpoint path."""
    from transformers import (AutoTokenizer,
                               AutoModelForSequenceClassification,
                               get_linear_schedule_with_warmup)

    save_dir = save_dir or os.path.join(CKPT_DIR, "verify_scorer")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Training VerifyScorer ({encoder_name}) ===")
    set_seed()
    _free_memory()

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        encoder_name, num_labels=2).to(DEVICE)
    model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_items = _make_verify_items(train_df)
    val_items   = _make_verify_items(val_df)
    print(f"  {len(train_items)} train / {len(val_items)} val items")

    train_dl = DataLoader(VerifyDataset(train_items, tokenizer),
                          batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(VerifyDataset(val_items,   tokenizer),
                          batch_size=batch_size * 2, shuffle=False, num_workers=2)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    scaler  = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                out  = model(input_ids=batch["input_ids"].to(DEVICE),
                             attention_mask=batch["attention_mask"].to(DEVICE),
                             labels=batch["label"].to(DEVICE))
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for batch in val_dl:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                    out   = model(input_ids=batch["input_ids"].to(DEVICE),
                                  attention_mask=batch["attention_mask"].to(DEVICE))
                preds = out.logits.argmax(-1).cpu()
                lbls  = batch["label"]
                tp += ((preds == 1) & (lbls == 1)).sum().item()
                fp += ((preds == 1) & (lbls == 0)).sum().item()
                fn += ((preds == 0) & (lbls == 1)).sum().item()

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        print(f"  Epoch {epoch}/{epochs}  loss={total_loss/len(train_dl):.4f}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    Saved (F1={best_f1:.3f})")

        _free_memory()

    del model
    _free_memory()
    print(f"VerifyScorer saved → {save_dir}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# Component 4: VoteRanker (shares backbone with VerifyScorer)
# ─────────────────────────────────────────────────────────────────────────────

def train_vote_ranker(
    train_df,
    val_df,
    encoder_name: str   = ENCODER_NAME,
    epochs:       int   = 10,
    lr:           float = 1e-5,
    batch_size:   int   = 8,
    save_dir:     str   = None,
) -> str:
    """
    Train VoteRanker: given (problem, candidate_answer_str) → quality score.
    Architecture identical to VerifyScorer (binary: correct / wrong answer).
    Can reuse VerifyScorer weights if the dataset is the same.
    """
    # Build items: (problem, answer_str, label)
    class RankDataset(Dataset):
        def __init__(self, items, tokenizer, max_len=256):
            self.items   = items
            self.tok     = tokenizer
            self.max_len = max_len
        def __len__(self): return len(self.items)
        def __getitem__(self, idx):
            prob, ans_str, label = self.items[idx]
            enc = self.tok(prob, f"Proposed answer: {ans_str}",
                           max_length=self.max_len, padding="max_length",
                           truncation=True, return_tensors="pt")
            return {
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label":          torch.tensor(label, dtype=torch.long),
            }

    def _make_rank_items(df):
        items = []
        for _, row in df.iterrows():
            prob = str(row.get("problem", ""))
            ans  = str(row.get("answer", ""))
            if not prob:
                continue
            items.append((prob, ans, 1))   # correct
            try:
                wrong = str(int(float(ans)) + random.choice([-1, 1, 3, -3]))
            except Exception:
                wrong = ans + "_wrong"
            items.append((prob, wrong, 0))  # wrong
        random.shuffle(items)
        return items

    from transformers import (AutoTokenizer,
                               AutoModelForSequenceClassification,
                               get_linear_schedule_with_warmup)

    save_dir = save_dir or os.path.join(CKPT_DIR, "vote_ranker")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Training VoteRanker ({encoder_name}) ===")
    set_seed()
    _free_memory()

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        encoder_name, num_labels=2).to(DEVICE)
    model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_items = _make_rank_items(train_df)
    val_items   = _make_rank_items(val_df)

    train_dl = DataLoader(RankDataset(train_items, tokenizer),
                          batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(RankDataset(val_items,   tokenizer),
                          batch_size=batch_size * 2, shuffle=False, num_workers=2)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dl) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )
    scaler  = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                out  = model(input_ids=batch["input_ids"].to(DEVICE),
                             attention_mask=batch["attention_mask"].to(DEVICE),
                             labels=batch["label"].to(DEVICE))
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += out.loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                with torch.cuda.amp.autocast(dtype=AMP_DTYPE, enabled=(DEVICE == "cuda")):
                    out   = model(input_ids=batch["input_ids"].to(DEVICE),
                                  attention_mask=batch["attention_mask"].to(DEVICE))
                preds = out.logits.argmax(-1).cpu()
                correct += (preds == batch["label"]).sum().item()
                total   += len(batch["label"])

        acc = correct / max(total, 1)
        print(f"  Epoch {epoch}/{epochs}  loss={total_loss/len(train_dl):.4f}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"    Saved (val_acc={best_acc:.3f})")

        _free_memory()

    del model
    _free_memory()
    print(f"VoteRanker saved → {save_dir}")
    return save_dir


# ─────────────────────────────────────────────────────────────────────────────
# Master trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    train_df,
    val_df,
    components:   Optional[list[str]] = None,
    encoder_name: str = ENCODER_NAME,
    ckpt_dir:     str = CKPT_DIR,
) -> dict[str, str]:
    """
    Train all (or a subset of) components sequentially.
    Frees GPU memory between each training run.

    Parameters
    ----------
    components : subset of ["answer_type", "retrieval", "verify", "vote"]
                 If None, trains all four.

    Returns dict mapping component name → checkpoint path.
    """
    all_components = ["answer_type", "retrieval", "verify", "vote"]
    if components is None:
        components = all_components

    results = {}

    if "answer_type" in components:
        results["answer_type"] = train_answer_type_classifier(
            train_df, val_df,
            encoder_name=encoder_name,
            save_dir=os.path.join(ckpt_dir, "answer_type_classifier"),
        )
        _free_memory()

    if "retrieval" in components:
        results["retrieval"] = train_retrieval_encoder(
            train_df,
            save_dir=os.path.join(ckpt_dir, "retrieval_encoder"),
        )
        _free_memory()

    if "verify" in components:
        results["verify"] = train_verify_scorer(
            train_df, val_df,
            encoder_name=encoder_name,
            save_dir=os.path.join(ckpt_dir, "verify_scorer"),
        )
        _free_memory()

    if "vote" in components:
        results["vote"] = train_vote_ranker(
            train_df, val_df,
            encoder_name=encoder_name,
            save_dir=os.path.join(ckpt_dir, "vote_ranker"),
        )
        _free_memory()

    print("\n=== All training complete ===")
    for name, path in results.items():
        print(f"  {name:<20} → {path}")
    return results
