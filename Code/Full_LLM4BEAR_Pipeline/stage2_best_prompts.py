"""Stage 2 — Determining Best Prompts (evaluator prompt selection by ICC).

Scores the EGPO-refined evaluator prompts on the held-out *annotated* test bundles
and picks, per (domain x role), the refined prompt whose LLM scores agree best with
the human-expert scores. Agreement is measured with the absolute-agreement,
single-rater intraclass correlation ICC(A,1) computed via a two-way ANOVA.

In the paper the three "personalities" ["Harsh", "Balanced", "Lenient"] correspond to
["bad", "middle", "good"] here. "no_experts" refers to the score-only optimisation
signal, which is the set the downstream pipeline consumes.

Two entrypoints (argparse):

* default (optionally ``--mock``): run the full ICC evaluation over the annotated
  test bundles (makes OpenAI calls), then select best-by-ICC and write
  ``output/selected_best_prompts.pkl``. ``--mock`` caps the scored test bundles to
  ``mock_configs.BEST_PROMPTS_SUBSET`` and restricts to ``mock_configs.DOMAINS``.

* ``--export-selection``: NO API calls. Build ``selected_best_prompts.pkl`` directly
  from the repo-supplied refined prompts using the known best top-3 indices per role
  (the ICC winners hardcoded in the notebook).

The deliverable, ``output/selected_best_prompts.pkl``, is a dict
``{domain: [bad_prompt, middle_prompt, good_prompt]}`` of three prompt STRINGS per
domain, consumed by stage 3.

De-Colab'd: no google.colab / drive mounts, no runtime git clone or pip installs, no
hardcoded /content or Drive paths. Data comes from ``config`` paths and OpenAI calls
go through ``common``. Import is cheap — all data loading and API calls live inside
functions.
"""

import argparse
import asyncio
import pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import config
import mock_configs
from common import (
    extract_bundle_score,
    extract_bundle_verdict,
    input_strings,
    openai_request,
)
from prompts import adding_metrics

# --- JSON output schemas (ported locally) ---
# NOTE: Determining_Best_Prompts orders the fields score-FIRST, then the two yes/no
# verdicts — this differs from prompts.py (which puts score last), so we port the
# notebook's own versions here to match its behaviour exactly.
json_bad = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n```json\n{{\n"
    "score: float, bundle quality out of 5.\n"
    "is_poor_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_acceptable_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n}}"
)

json_middle = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n```json\n{{\n"
    "score: float, bundle quality out of 5.\n"
    "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_good_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n}}"
)

json_good = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n```json\n{{\n"
    "score: float, bundle quality out of 5.\n"
    "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_high_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n}}"
)

# Per-role wiring: which consideration string and JSON schema each evaluator uses.
ROLE_CONSIDERATION = {"bad": "1-2", "middle": "3", "good": "4-5"}
ROLE_JSON = {"bad": json_bad, "middle": json_middle, "good": json_good}

# How each domain's annotated *test* split is sliced out of its files.
#   score: (start, stop) rows of column 2 in <domain>_annotations.xlsx
#   bundle: (start, stop) of the built bundle list (stop=None means "to the end")
ANNOTATION_SLICES = {
    "electronic": {"score": (72, 137), "bundle": (72, None)},
    "clothing": {"score": (80, 151), "bundle": (80, None)},
    "food": {"score": (80, 150), "bundle": (80, 150)},
}

# The known ICC winners hardcoded in the notebook (cell ~31, best_refined_no_exp).
# selected[domain] = [top_3_bad[i_bad], top_3_middle[i_mid], top_3_good[i_good]].
KNOWN_BEST_INDICES = {
    "electronic": {"bad": 1, "middle": 1, "good": 2},
    "clothing": {"bad": 2, "middle": 2, "good": 0},
    "food": {"bad": 0, "middle": 2, "good": 0},
}


# --- Reliability statistics (ported faithfully) ---
def purge_none_pairs(list_a, list_b):
    """Return the sub-lists of (a, b) pairs where neither entry is None."""
    if len(list_a) != len(list_b):
        raise ValueError("Input lists must have the same length.")

    purged_a = []
    purged_b = []
    for score_a, score_b in zip(list_a, list_b):
        if score_a is not None and score_b is not None:
            purged_a.append(score_a)
            purged_b.append(score_b)
    return purged_a, purged_b


def calculate_reliability_metrics_complete(scores_rater_a_raw, scores_rater_b_raw):
    """Purge None pairs, then compute ICC(A,1), Pearson r and Spearman rho.

    ICC is the absolute-agreement, single-rater form derived from a two-way ANOVA
    (raters A/B x items). Returns a dict, or {'Error': ...} when there is not enough
    usable data / the ANOVA is degenerate.
    """
    scores_rater_a, scores_rater_b = purge_none_pairs(scores_rater_a_raw, scores_rater_b_raw)

    if len(scores_rater_a) < 3:  # need at least 3 items for a stable ANOVA
        return {"Error": "Not enough valid paired data points (min 3 required for stability)."}

    n_items = len(scores_rater_a)

    # Long-format frame: each item appears once per rater.
    data = pd.DataFrame({
        "Item": [f"Item_{i}" for i in range(n_items)] * 2,
        "Rater": ["A"] * n_items + ["B"] * n_items,
        "Score": scores_rater_a + scores_rater_b,
    })

    # Two-way ANOVA -> mean squares.
    model = ols("Score ~ C(Item) + C(Rater)", data=data).fit()
    anova_table = anova_lm(model, typ=1)

    try:
        MS_I = anova_table.loc["C(Item)", "mean_sq"]   # items
        MS_R = anova_table.loc["C(Rater)", "mean_sq"]  # raters
        MS_E = anova_table.loc["Residual", "mean_sq"]  # error
    except KeyError:
        return {"Error": "ANOVA Mean Squares extraction failed. Data structure might be perfectly uniform."}

    # ICC(A,1): absolute agreement, single rater.
    k = 2  # number of raters (A and B)
    n = n_items
    denominator = MS_I + (k - 1) * MS_E + (k / n) * (MS_R - MS_E)
    if denominator != 0:
        icc_value = (MS_I - MS_E) / denominator
    else:
        icc_value = float("nan")

    pearson_corr, _ = pearsonr(scores_rater_a, scores_rater_b)
    spearman_rho, _ = spearmanr(scores_rater_a, scores_rater_b)

    return {
        "N_Valid_Pairs": n_items,
        "ICC_Absolute_Agreement": icc_value,
        "Pearson_r": pearson_corr,
        "Spearman_rho": spearman_rho,
    }


# --- Reward helpers (used by separate_considerations, ported for fidelity) ---
def true_rmse(score_list, true_list):
    """RMSE over the pairs where the predicted score is not None."""
    score_array = np.array(score_list, dtype=object)
    true_array = np.array(true_list, dtype=object)

    valid_indices = score_array != None  # noqa: E711 (need element-wise, not `is not`)
    valid_scores = score_array[valid_indices].astype(float)
    valid_trues = true_array[valid_indices].astype(float)

    if len(valid_scores) == 0:
        return 0.0
    return np.sqrt(np.mean((valid_scores - valid_trues) ** 2))


def inverse_reward(collective_rmse, base_reward=1.0, epsilon=0.1):
    """Reward from a collective error via an inverse function (epsilon avoids /0)."""
    return base_reward / (collective_rmse + epsilon)


# --- Data loading ---
def load_refined_prompts(folder, char):
    """Return the ``top_3_prompts`` list from ``<folder>/Refined_<char>.pkl``."""
    with open(folder / f"Refined_{char}.pkl", "rb") as f:
        top_3_prompts, _ = pickle.load(f)
    return top_3_prompts


def load_domain_annotations(domain):
    """Load a domain's annotated *test* split.

    Returns (testing_strings, testing_scores): the formatted intent+items strings for
    each test bundle and the aligned human-expert scores.
    """
    slices = ANNOTATION_SLICES[domain]

    dataframe = pd.read_excel(config.BUNDLE_ANNOTATIONS / f"{domain}_annotations.xlsx")
    s_start, s_stop = slices["score"]
    testing_scores = dataframe.iloc[s_start:s_stop, 2].tolist()

    info = pd.read_pickle(config.BUNDLE_ANNOTATIONS / f"{domain}_bundle_info.pkl")
    # info[0] = per-bundle item titles, info[1] = descriptions, info[2] = image ids,
    # info[3] = intents. Scoring only needs the intents and item titles.
    intents = info[3]
    item_titles = info[0]

    b_start, b_stop = slices["bundle"]
    if b_stop is None:
        b_stop = len(item_titles)
    test_intents = [intents[i] for i in range(b_start, b_stop)]
    test_items = [item_titles[i] for i in range(b_start, b_stop)]

    testing_strings = input_strings(test_intents, test_items)
    return testing_strings, testing_scores


# --- LLM scoring of a prompt over the test bundles ---
async def separate_considerations(char, initial_prompt, sample_data, true_scores,
                                  json_addition, constant_metrics="", consideration=""):
    """Score every bundle in ``sample_data`` with ``initial_prompt`` and persist results.

    Sends one request per bundle (prompt = data + JSON schema, system = evaluator
    prompt + metrics), extracts the numeric bundle score (and yes/no verdicts), and
    computes a reward against ``true_scores``. Saves
    ``output/filter_results/<char>.pkl`` as ``[reward, responses, verdicts, scores]``
    and returns the same four values.
    """
    prompt_list = [{"prompts": data + "\n" + json_addition} for data in sample_data]
    responses = await openai_request(prompt_list, initial_prompt + constant_metrics)

    scores = [extract_bundle_score(r) for r in responses]
    if consideration in ("1-2", "3", "4-5"):
        verdicts = [extract_bundle_verdict(r, consideration) for r in responses]
    else:
        verdicts = None

    reward = 0
    target_scores = true_scores

    if consideration == "1-2":
        for i in range(len(responses)):
            try:
                if target_scores[i] > 2 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":
                    reward += 1
                elif target_scores[i] < 3 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":
                    reward += 1
            except Exception:
                print("response error")
    elif consideration == "4-5":
        for i in range(len(responses)):
            try:
                if target_scores[i] > 3 and verdicts[i][1] == "yes" and verdicts[i][0] == "no":
                    reward += 1
                elif target_scores[i] < 4 and verdicts[i][1] == "no" and verdicts[i][0] == "yes":
                    reward += 1
            except Exception:
                print("response error")
    elif consideration == "3":
        for i in range(len(responses)):
            try:
                if target_scores[i] < 4 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":
                    reward += 1
                elif target_scores[i] > 3 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":
                    reward += 1
            except Exception:
                print("response error")
    elif consideration == "":
        collective_rmse = true_rmse(scores, target_scores)
        reward = inverse_reward(collective_rmse, base_reward=3, epsilon=0.5)

    out = config.ensure_output_dir() / "filter_results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{char}.pkl", "wb") as f:
        pickle.dump([reward, responses, verdicts, scores], f)

    return reward, responses, verdicts, scores


# --- Best-by-ICC selection (default entrypoint) ---
async def evaluate_and_select(domains, subset=None):
    """Score each domain's refined (no-experts) prompts on its annotated test bundles,
    and pick, per (domain x role), the prompt with the highest ICC vs the experts.

    Writes ``output/selected_best_prompts.pkl`` and returns the selection dict.
    """
    selected = {}

    for domain in domains:
        print(f"\n########## Evaluating domain: {domain} ##########")
        testing_strings, testing_scores = load_domain_annotations(domain)

        if subset is not None:
            testing_strings = testing_strings[:subset]
            testing_scores = testing_scores[:subset]

        role_best_prompts = []
        for role in config.ROLES:
            consideration = ROLE_CONSIDERATION[role]
            json_addition = ROLE_JSON[role]
            char_base = f"{role}_evaluator_{domain}_importance_no_experts"

            # Candidates: the three top refined prompts for this (domain, role).
            top_3_prompts = load_refined_prompts(config.EGPO_FINAL_PROMPTS, char_base)

            best_icc = None
            best_prompt = None
            for j, prompt in enumerate(top_3_prompts):
                char = f"{domain}_dataset_{char_base}_{j}"
                _, _, _, scores = await separate_considerations(
                    char=char,
                    initial_prompt=prompt,
                    sample_data=testing_strings,
                    true_scores=testing_scores,
                    json_addition=json_addition,
                    constant_metrics=adding_metrics,
                    consideration=consideration,
                )

                metrics = calculate_reliability_metrics_complete(scores, testing_scores)
                if "Error" in metrics:
                    icc = float("nan")
                else:
                    icc = metrics["ICC_Absolute_Agreement"]
                print(f"  {role} prompt {j}: ICC={icc}")

                # np.nan comparisons are always False, so nan never wins.
                if best_icc is None or (icc == icc and (best_icc != best_icc or icc > best_icc)):
                    best_icc = icc
                    best_prompt = prompt

            # Fall back to the first candidate if every ICC was unusable.
            if best_prompt is None:
                best_prompt = top_3_prompts[0]
            print(f"  -> best {role}: ICC={best_icc}")
            role_best_prompts.append(best_prompt)

        selected[domain] = role_best_prompts

    out = config.ensure_output_dir()
    with open(out / "selected_best_prompts.pkl", "wb") as f:
        pickle.dump(selected, f)
    print(f"\nWrote {out / 'selected_best_prompts.pkl'}")
    return selected


# --- Known-winners export (no API) ---
def export_selection():
    """Build ``selected_best_prompts.pkl`` from the repo-supplied refined prompts using
    the notebook's known ICC-winning indices. Makes no API calls.
    """
    selected = {}
    for domain in config.DOMAINS:
        role_prompts = []
        for role in config.ROLES:
            idx = KNOWN_BEST_INDICES[domain][role]
            char = f"{role}_evaluator_{domain}_importance_no_experts"
            top_3_prompts = load_refined_prompts(config.REFINEMENT_FINAL_PROMPTS, char)
            role_prompts.append(top_3_prompts[idx])
        selected[domain] = role_prompts

    out = config.ensure_output_dir()
    with open(out / "selected_best_prompts.pkl", "wb") as f:
        pickle.dump(selected, f)
    print(f"Wrote {out / 'selected_best_prompts.pkl'} "
          f"({len(selected)} domains, 3 prompts each)")
    return selected


def main():
    parser = argparse.ArgumentParser(description="LLM4BEAR Determining Best Prompts (stage 2)")
    parser.add_argument("--mock", action="store_true", help="cheap smoke test via mock_configs")
    parser.add_argument("--export-selection", action="store_true",
                        help="build selected_best_prompts.pkl from known ICC winners (no API calls)")
    args = parser.parse_args()

    if args.export_selection:
        export_selection()
        return

    if args.mock:
        domains = mock_configs.DOMAINS
        subset = mock_configs.BEST_PROMPTS_SUBSET
    else:
        domains = config.DOMAINS
        subset = None

    asyncio.run(evaluate_and_select(domains, subset))


if __name__ == "__main__":
    main()
