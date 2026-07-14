"""Central configuration for the LLM4BEAR dataset pipeline.

This module resolves where the LLM4BEAR data lives and holds the shared constants
(domains, model, generation params) used across every stage.

Data layout
-----------
This folder is self-contained: the LLM4BEAR data it needs (``BundleRec Data``, ``Bundle
Annotations``, ``2_Bundle Refinement``, ``1_EGPO``) is bundled inside it under ``./_repo``
so a reviewer can pull just this folder and run everything — nothing is cloned at runtime.
``REPO_ROOT`` is resolved in this order:

1. ``$LLM4BEAR_REPO_ROOT`` if set (override).
2. ``./_repo``, the bundled data (the normal case).
3. The nearest ancestor directory that contains ``BundleRec Data`` (fallback: if this
   folder is dropped into a full LLM4BEAR checkout without its own ``_repo``).
4. Fall back to the parent directory.
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Load a local .env (OPENAI_API_KEY, WANDB_KEY) if python-dotenv is available.
try:
    from dotenv import load_dotenv

    load_dotenv(HERE / ".env")
except ImportError:
    pass


def _resolve_repo_root() -> Path:
    env = os.environ.get("LLM4BEAR_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if (HERE / "_repo" / "BundleRec Data").is_dir():
        return HERE / "_repo"  # standalone bundled data (dev)
    # Deployed as a repo subfolder: walk up to the repo root (holds "BundleRec Data").
    for parent in (HERE, *HERE.parents):
        if (parent / "BundleRec Data").is_dir():
            return parent
    return HERE.parent


REPO_ROOT = _resolve_repo_root()

# --- Data directories (all under REPO_ROOT) ---
BUNDLEREC_DATA = REPO_ROOT / "BundleRec Data"
BUNDLE_ANNOTATIONS = REPO_ROOT / "Bundle Annotations"
EGPO_DIR = REPO_ROOT / "1_EGPO"
REFINEMENT_DIR = REPO_ROOT / "2_Bundle Refinement"
EGPO_FINAL_PROMPTS = EGPO_DIR / "final_prompts"
REFINEMENT_FINAL_PROMPTS = REFINEMENT_DIR / "final_prompts"
HISTORIC_REFINEMENT = REFINEMENT_DIR / "historic_bundle_refinement"

# --- Pipeline output (created at runtime) ---
OUTPUT_DIR = HERE / "output"

# --- Task constants ---
DOMAINS = ["electronic", "clothing", "food"]
ROLES = ["bad", "middle", "good"]  # == Harsh / Balanced / Lenient

# --- Model / generation config (matches the original notebooks) ---
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
MAX_TOKENS = 8000
SEED = 42
BATCH_SIZE = 128
EMBED_MODEL = "intfloat/e5-large-v2"

# --- Canonical run parameters ---
# These were hardcoded inside the notebooks; surfaced here so a full run is reproducible
# and a cheap run just swaps in mock_configs.py. See mock_configs.py for the reduced set.
REFINEMENT_NUM_ITERATIONS = 10   # stage 3: evaluate->diagnose->edit loops (author used up to 20)
REFINEMENT_SUBSET = None          # stage 3: None = all seed bundles per domain
EGPO_SEARCH_DEPTH = 6             # stage 1: beam-search iterations (must be >= 3; EGPO pools the last ~4)
BEST_PROMPTS_SUBSET = None        # stage 2: None = score all annotated test bundles

# --- Secrets (were google.colab.userdata in the notebooks) ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WANDB_KEY = os.environ.get("WANDB_KEY", "")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
