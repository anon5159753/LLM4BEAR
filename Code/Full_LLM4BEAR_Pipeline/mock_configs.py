"""Reduced ("mock") run configuration for smoke-testing the pipeline end-to-end.

The full pipeline is expensive (thousands of bundles x many LLM calls x up to 20
iterations). These knobs let both the authors and reviewers verify the code actually
runs — cheaply — before committing to a full run.

Every stage accepts ``--mock`` on the command line; when passed, it reads its run
parameters from here instead of the full defaults in ``config.py``.
"""

# Only exercise one domain for a smoke test (any subset of config.DOMAINS).
DOMAINS = ["electronic"]

# Process just the first N starting bundles per domain (full run uses all ~1750+).
SUBSET = 5

# BundleRefinement (stage 3): iterations of evaluate -> diagnose -> add/remove.
# Full run uses 10-20; one iteration is enough to prove the loop works.
NUM_ITERATIONS = 1

# Smaller OpenAI batch so a mock run doesn't fan out 128 requests at once.
BATCH_SIZE = 128

# EGPO (stage 1): beam-search depth. MUST be >= 3 — EGPO pools the last ~4 iterations
# to pick the top-3 prompts, so a shallower run can't produce a valid selection.
EGPO_SEARCH_DEPTH = 3

# Determining Best Prompts (stage 2): cap the test bundles scored per prompt.
BEST_PROMPTS_SUBSET = 5
