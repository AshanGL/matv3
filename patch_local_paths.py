"""
patch_local_paths.py
────────────────────
Run this as Cell 2.5 (after checkpoints are copied, before solver is loaded).

What it does:
  1. Scans input datasets to find locally available model folders
  2. Patches config.json files inside each checkpoint so encoder_name
     points to the local path instead of a HuggingFace hub string
  3. Sets HF env vars so transformers never tries the internet

Models it looks for:
  - microsoft/deberta-v3-base        (answer_type_classifier, verify_scorer)
  - sentence-transformers/all-MiniLM-L6-v2  (retrieval_encoder, knowledge_db)
"""

import os
import json

# ── 1. Where to search for local model folders ────────────────────────────────
SEARCH_ROOTS = [
    '/kaggle/input',
    '/kaggle/input/datasets',
    '/kaggle/working',
]

# Models needed: HF hub name → local folder name variations to look for
MODELS_NEEDED = {
    'microsoft/deberta-v3-base': [
        'deberta-v3-base',
        'deberta_v3_base',
        'deberta-v3',
        'deberta',
    ],
    'sentence-transformers/all-MiniLM-L6-v2': [
        'all-MiniLM-L6-v2',
        'all_MiniLM_L6_v2',
        'minilm',
        'MiniLM-L6-v2',
        'all-minilm-l6-v2',
    ],
    'mistralai/Mistral-Small-3.1-24B-Instruct-2503': [
        'Mistral-Small-3.1-24B-Instruct-2503',
        'mistral-small',
    ],
}

# ── 2. Checkpoint config files to patch ───────────────────────────────────────
WORKING = '/kaggle/working'
CHECKPOINT_CONFIGS = [
    os.path.join(WORKING, 'checkpoints', 'answer_type_classifier', 'config.json'),
    os.path.join(WORKING, 'checkpoints', 'verify_scorer',          'config.json'),
    os.path.join(WORKING, 'checkpoints', 'retrieval_encoder',      'config.json'),
    os.path.join(WORKING, 'checkpoints', 'vote_ranker',            'config.json'),
]

# ── 3. Source files to patch (knowledge_db.py constant) ───────────────────────
SOURCE_FILES = [
    os.path.join(WORKING, 'matv3', 'knowledge_db.py'),
    os.path.join(WORKING, 'matv3', 'train_new.py'),
]

# ─────────────────────────────────────────────────────────────────────────────
# Step A: Find local model paths
# ─────────────────────────────────────────────────────────────────────────────

def find_model_locally(hub_name: str, name_variants: list) -> str | None:
    """Walk SEARCH_ROOTS and return path of first folder that looks like the model."""
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Check if this dir looks like a HF model (has config.json or pytorch_model.bin)
            has_model_files = any(
                f in filenames for f in [
                    'config.json', 'pytorch_model.bin',
                    'model.safetensors', 'tokenizer_config.json'
                ]
            )
            if not has_model_files:
                continue
            folder_lower = os.path.basename(dirpath).lower()
            for variant in name_variants:
                if variant.lower() in folder_lower or folder_lower in variant.lower():
                    return dirpath
    return None


print("=" * 60)
print("Scanning for local model folders...")
print("=" * 60)

resolved = {}   # hub_name → local_path

for hub_name, variants in MODELS_NEEDED.items():
    local_path = find_model_locally(hub_name, variants)
    if local_path:
        resolved[hub_name] = local_path
        print(f"  ✓  {hub_name}")
        print(f"       → {local_path}")
    else:
        print(f"  ✗  {hub_name}  — NOT FOUND locally")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Step B: Patch checkpoint config.json files
# ─────────────────────────────────────────────────────────────────────────────

print("Patching checkpoint config.json files...")

for cfg_path in CHECKPOINT_CONFIGS:
    if not os.path.exists(cfg_path):
        print(f"  skip (not found): {cfg_path}")
        continue

    with open(cfg_path) as f:
        cfg_data = json.load(f)

    changed = False
    for key in ('encoder_name', 'model_name', 'base_model'):
        if key not in cfg_data:
            continue
        hub_name = cfg_data[key]
        if hub_name in resolved:
            cfg_data[key] = resolved[hub_name]
            print(f"  patched  {os.path.basename(os.path.dirname(cfg_path))}/config.json")
            print(f"           {key}: {hub_name}")
            print(f"                → {resolved[hub_name]}")
            changed = True
        elif '/' in hub_name and not hub_name.startswith('/'):
            print(f"  WARNING: {hub_name} still points to HuggingFace (no local copy found)")

    if changed:
        with open(cfg_path, 'w') as f:
            json.dump(cfg_data, f, indent=2)

print()

# ─────────────────────────────────────────────────────────────────────────────
# Step C: Patch EMBEDDING_MODEL constant in source files
# ─────────────────────────────────────────────────────────────────────────────

print("Patching source file constants...")

MINILM_HUB = 'sentence-transformers/all-MiniLM-L6-v2'
DEBERTA_HUB = 'microsoft/deberta-v3-base'

for src_path in SOURCE_FILES:
    if not os.path.exists(src_path):
        print(f"  skip (not found): {src_path}")
        continue

    with open(src_path) as f:
        content = f.read()

    original = content

    if MINILM_HUB in resolved:
        content = content.replace(
            f'"{MINILM_HUB}"',
            f'"{resolved[MINILM_HUB]}"'
        )

    if DEBERTA_HUB in resolved:
        content = content.replace(
            f'"{DEBERTA_HUB}"',
            f'"{resolved[DEBERTA_HUB]}"'
        )

    if content != original:
        with open(src_path, 'w') as f:
            f.write(content)
        print(f"  patched: {src_path}")
    else:
        print(f"  no changes needed: {src_path}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Step D: Set environment variables to block all HuggingFace network calls
# ─────────────────────────────────────────────────────────────────────────────

import os

os.environ['TRANSFORMERS_OFFLINE']    = '1'
os.environ['HF_DATASETS_OFFLINE']     = '1'
os.environ['HF_HUB_OFFLINE']          = '1'
os.environ['TOKENIZERS_PARALLELISM']  = 'false'
os.environ['CUDA_VISIBLE_DEVICES']    = '0'

print("Environment variables set:")
print("  TRANSFORMERS_OFFLINE=1")
print("  HF_DATASETS_OFFLINE=1")
print("  HF_HUB_OFFLINE=1")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Step E: Final summary
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Summary")
print("=" * 60)
missing = [k for k in MODELS_NEEDED if k not in resolved]
if missing:
    print("MISSING local models — these will still fail:")
    for m in missing:
        print(f"  - {m}")
    print()
    print("Fix: download these models and add them as Kaggle datasets.")
else:
    print("All models resolved to local paths. Safe to run offline.")
print("=" * 60)
