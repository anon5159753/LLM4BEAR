"""Stage 1 — EGPO (Expert-Guided Prompt Optimization).

A beam-search / APO (Automatic Prompt Optimization) loop that refines an evaluator
("judge") prompt against a small annotated training set. For each evaluator config it:

1. Expands the current beam of prompts by diagnosing the cases the prompt got wrong
   (``Improve.run`` -> inferring reasons -> refining -> augmenting).
2. Selects the top-b prompts on the training set via a UCB bandit (``Select.run``).
3. After ``search_depth`` iterations, scores the surviving prompts on the validation
   set and keeps the top-3 (``get_top_3_prompts``).

The heavy lifting lives in the ``opt.*`` engine (config / request / reward / improve /
select) bundled under ``config.EGPO_DIR/opt``. We add that directory to ``sys.path`` at
import time (guarded), but the actual ``from opt... import`` statements happen inside the
functions that use them so that a bare ``import stage1_egpo`` works even when the engine's
optional deps (wandb, yaml, ...) are absent.

De-Colab notes:
- Keys come from ``config.OPENAI_API_KEY`` / ``config.WANDB_KEY`` (were
  ``google.colab.userdata``), and are written into the engine config in ``build_config``.
- Google Drive output paths become ``config.OUTPUT_DIR/final_prompts/Refined_{char}.pkl``.
- The notebook's ``!git clone`` / ``%cd`` / ``!pip install`` shell cells are gone; the
  engine is read straight from the bundled ``_repo`` via the ``sys.path`` insert below.
- The engine's ``opt.config.init_config`` reads ``assets/*.yaml`` that the repo does NOT
  ship, so it crashes. ``build_config`` reconstructs the same dict directly instead — no
  assets needed. See ``build_config`` for the full key inventory and how defaults were set.

The importance-graph reasoning inside the evaluator prompts (``initial_importance_prompt``
and its bad/middle/good variants) is part of the *judge* prompt and is ported verbatim.

A full run is EXPENSIVE (many LLM calls per config x several configs) and needs an
OpenAI key (and optionally a wandb key). Use ``python stage1_egpo.py --mock`` for a
cheap, shallow smoke test.
"""

import argparse
import asyncio
import pickle
import random
import sys

import config
import mock_configs

# Make the engine package importable (``from opt.config import init_config`` etc.).
# Guarded so importing this module never fails just because the repo layout differs.
if config.EGPO_DIR.is_dir():
    _egpo_path = str(config.EGPO_DIR)
    if _egpo_path not in sys.path:
        sys.path.insert(0, _egpo_path)


# Role -> "consideration" gate used by the reward/verdict logic in the engine.
ROLE_CONSIDERATION = {"bad": "1-2", "middle": "3", "good": "4-5"}


def build_config():
    """Return the engine config dict that ``opt.config.init_config`` would have built.

    The upstream ``init_config`` loads ``assets/overall.yaml`` + ``assets/{model}.yaml``
    relative to ``os.getcwd()``, but the repo ships NO ``assets/`` folder (checked the
    full tree), so ``init_config`` crashes. We reconstruct the dict directly instead.

    Keys below are every ``config[...]`` consumed by the engine (parser.py / request.py /
    reward.py / improve.py / select.py). Values marked "reconstructed" are sensible APO
    defaults (the original yaml values are unrecoverable); the rest come from parser.py's
    ``Args`` or from our own ``config.py``. ``structure_refinement`` overwrites the
    per-run keys (initial_prompt, json_addition, case, ...) on top of this."""
    return {
        # --- from parser.py Args / our config.py ---
        "model": config.MODEL,          # gpt-4o-mini
        "seed": config.SEED,            # 42
        "reward_func": "rmse",          # legacy; Reward stores but the bundle path ignores it
        "dataset": "bundle",
        "batch_size": 16,               # train examples sampled per beam expansion (train has 50)
        # --- reconstructed APO hyper-params (were in the missing yaml) ---
        "num_feedbacks": 4,             # reasons inferred per wrong example
        "error_batch_size": 4,          # wrong examples used to infer reasons
        "num_candidates": 5,            # successor candidates kept per parent prompt
        "addition_sample": 1,           # augmentation copies per refined prompt
        "beam_width": 4,                # top-b prompts kept by the UCB select step
        "time_steps": 5,                # UCB iterations inside select
        "sample_num": 3,                # train examples scored per UCB iteration
        "explore_param": 1.0,           # UCB exploration constant
        "use_wandb": False,             # no experiment tracking by default
        "search_depth": config.EGPO_SEARCH_DEPTH,  # beam iterations (overridable per run)
        # Only used by opt.utils (not on our code path); kept for completeness.
        "data_path": str(config.EGPO_DIR / "Dataset") + "/",
    }


# --------------------------------------------------------------------------- #
# Improve-prompt templates (ported verbatim from the notebook).                #
# --------------------------------------------------------------------------- #

adding_metrics = "\nFunctionality Integration: Describe how a user would utilize this collection of items to achieve their primary goal. Considering the entire workflow, is this a complete and logical set of items for the task, or is there an irrelevant or missing item? \n"\
              "Similarity: What is the common theme or category that connects these items?\n"\
              "Complementarity: Are these items more valuable together than they would be if sold separately? Does the presence of one item create a clear reason to buy the other(s)?\n"\
              "Diversity: Does the variety of items in this bundle cater to a broad set of related needs for a single user, or does the mix of items seem unfocused and random?\n"


# Importance graph reasoning generally allows better prompts to be refined. This graph
# lives inside the judge prompt and is legitimate; port it faithfully.
initial_importance_prompt = "You are an expert bundle strategist tasked distinguishing a $quality$/5 quality bundle. Your response should be a single judgement, is this bundle a $quality$/5 bundle?: yes/no. \n"\
                  "Your task is to diligently complete subtasks that can help guide your analysis step by step:\n" \
                  "1. Based on the stated intent, come up with how combinations of items can interact to fulfill the intent.\n" \
                  "2. If from your understanding, the combinations do not meet the stated intent, develop a new intent that your combinations can fulfill.\n" \
                  "3. Design a importance graph analysis for each bundle in the format:\n"\
                  "a. [Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]\n"\
                  "Reason: [Your explanation for this item's rank in this specific scenario]\n"\
                  "b. [2nd Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]\n"\
                  "Reason: [Your explanation]\n"\
                  "(...and so on for all other items)\n"\
                  "4. Based on your reasoning and analysis, you are to make two mutually exclusive yes/no judgements on the bundle. $criteria$" \
                  "5. Now, you are to analyse the bundle: \n"

bad_bundle_criteria = "\nFollow this criteria when evaluating the bundle:\n"\
                "1-2 - Poor: Some items or no items are connected thematically, but more than one modification needs to be made to guarantee the acceptability of the bundle.\n"\
                "3-5 - Acceptable: Items are connected thematically, and complement each other.\n"

middle_bundle_criteria = "\nFollow this criteria when evaluating the bundle:\n"\
                "1-3 - Needs Improvement: One or more modifications needs to be made in order to guarantee the acceptability of the bundle.\n"\
                "4-5 - Good Quality: No modifications need to be performed as this bundle is extremely well designed.\n"

good_bundle_criteria = "\nFollow this criteria when evaluating the bundle:\n"\
                "1-3 - Needs Improvement: One or more modifications needs to be made in order to guarantee the acceptability of the bundle.\n"\
                "4-5 - High Quality: No modifications need to be performed to be accepted by the end user.\n"


initial_importance_prompt_bad_bundle = initial_importance_prompt.replace("$criteria$",
                                                         bad_bundle_criteria).replace("$quality$", "1-2")

initial_importance_prompt_middle_bundle = initial_importance_prompt.replace("$criteria$",
                                                         middle_bundle_criteria).replace("$quality$", "1-3")

initial_importance_prompt_good_bundle = initial_importance_prompt.replace("$criteria$",
                                                         good_bundle_criteria).replace("$quality$", "4-5")


# Role -> initial (importance-graph) evaluator prompt.
ROLE_INITIAL_PROMPT = {
    "bad": initial_importance_prompt_bad_bundle,
    "middle": initial_importance_prompt_middle_bundle,
    "good": initial_importance_prompt_good_bundle,
}


convincing_prompt = """You are a senior analyst and adjudicator. Your job is to review a disagreement between a junior AI analyst and a human expert to determine if the expert's reasoning is sound and should be used as a training example.

--- CONTEXT ---

1.  **Bundle in Question:**
    $error_case$

2.  **Disagreement Summary:**
    $llm_judge$

3.  **Junior AI's Reasoning:**
    $llm_reasoning$

4.  **Expert's Score and Reasoning:**
    - Score: $true_score$/5
    - Reasoning: $annotation$

--- YOUR TASK ---

Perform your task in two parts.

**Part 1: Written Analysis**
First, write a brief analysis. Is the expert's argument convincing, logical, and specific? Treat the expert's reasoning as more accurate AI's reasoning unless it seems that expert overlooked a detail that the AI considered?
Second, if the expert seems correct, then you must identify what is the most important point to the expert, and what was most important to the AI which led to a disagreement.

**Part 2: Final Verdict (JSON)**
After your analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object with your final verdict. The verdict must be either "yes" (the expert is convincing) or "no".

**JSON Schema:**
```json
{{
    "verdict": str, yes/no response, do not provide anything else.
}}
"""

inferring_reasons = "I'm trying to refine a prompt that evaluated the following bundle incorrectly: $error_case$.\n"\
                    "$llm_judge$ The reasoning given was: $llm_reasoning$ \n"\
                    "The expert gave the bundle a score of $true_score$, with the reasoning:\n $annotation$ \n"\
                    "First, write a brief analysis. Is the expert's argument convincing, logical, and specific? Treat the expert's reasoning as more accurate AI's reasoning unless it seems that expert overlooked a detail that the AI considered?\n"\
                    "Second, if the expert seems correct, then you must identify what is the most important point to the expert, and what was most important to the AI which led to a disagreement.\n"\
                    "Summarise the difference in reasoning between the LLM and the expert, and incorporate what specific criteria and information did the expert use that the LLM overlooked.\n"\
                    "Give $num_feedbacks$ reasons of why the LLM got this example wrong.\n"\
                    "Wrap each reason with <START> and <END>"

inferring_reasons_no_experts =  "I'm trying to refine a prompt that evaluated the following bundle incorrectly: $error_case$.\n"\
                                "$llm_judge$ The reasoning given was: $llm_reasoning$ \n"\
                                "Give $num_feedbacks$ reasons of why the LLM could have gotten this example wrong.\n"\
                                "Wrap each reason with <START> and <END>"

refining_prompts = "I'm trying to write a zero-shot evaluation prompt.\n"\
                    "My current prompt is \"$prompt$\"\n"\
                    "But this prompt gets the following example wrong: $error_case$\n"\
                    "$llm_judge$."\
                    "Based on these example the problem with this prompt is that $reasons$.\n"\
                    "Based on the above information, please write an improved prompt without including any bundle items.\n"\
                    "The prompt should be wrapped with <START> and <END>.\n"\
                    "The new prompt is:"

augmenting_prompts = "Generate a variation of the following instruction while maintaining the semantic meaning.\n"\
                      "Input: $refined_prompt$\n"\
                      "The prompt should be wrapped with <START> and <END>.\n"\
                      "Output:"


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #

def _text_dir():
    return config.EGPO_DIR / "Dataset" / "bundle" / "Text"


def load_domain_data(domain):
    """Load (train, validation) annotated bundles for a domain.

    Files live under ``config.EGPO_DIR/Dataset/bundle/Text`` as
    ``<domain>_train_50.json`` and ``<domain>_valid.json``. (The notebook read a bare
    ``valid.json`` for the electronic split; that filename does not exist in the repo,
    so we use the per-domain ``<domain>_valid.json`` that does — see the confirmed
    listing in the module notes.)"""
    import json

    text_dir = _text_dir()
    with open(text_dir / f"{domain}_train_50.json", "r") as json_file:
        train_data = json.load(json_file)
    with open(text_dir / f"{domain}_valid.json", "r") as json_file:
        val_data = json.load(json_file)
    return train_data, val_data


# --------------------------------------------------------------------------- #
# Core APO logic (ported from notebook cells 6 and 7).                         #
# --------------------------------------------------------------------------- #

async def get_top_3_prompts(beam_candidate, val_data, reward_model, result_table):
    """Score every prompt in ``beam_candidate`` on the validation data and return the
    3 with the highest reward."""
    sample_data = list(val_data.values())
    reward_prompt_pairs = []

    # 1. Calculate the reward for each prompt and pair them up.
    for prompt in beam_candidate:
        reward = await reward_model.calculate_reward(prompt, sample_data)

        print(reward)
        reward_prompt_pairs.append((reward, prompt))
        if result_table is not None:
            result_table.add_data(prompt, reward)

    # 2. Sort the (reward, prompt) tuples in descending order by reward.
    sorted_pairs = sorted(reward_prompt_pairs, key=lambda item: item[0], reverse=True)

    # 3. Extract just the prompts from the top 3 pairs.
    top_3_prompts = [prompt for reward, prompt in sorted_pairs[:3]]

    return top_3_prompts


async def structure_refinement(char, initial_prompt, inferring_reasons, refining_prompts,
                               augmenting_prompts, training_data, validation_data,
                               constant_metrics="", metastructure=None, mode="single",
                               consideration="", convincing="", search_depth=None):
    """Run the APO loop for one evaluator config and pickle the top-3 refined prompts.

    Output goes to ``config.OUTPUT_DIR/final_prompts/Refined_{char}.pkl`` (was a Google
    Drive path in the notebook). ``search_depth`` overrides ``build_config``'s default
    when given; it must be >= 3 because the top-3 selection pools ``all_prompts[-4:]``."""
    # The engine package. Imported here (not at module top) so a bare
    # ``import stage1_egpo`` stays cheap and works even without the engine's deps.
    from opt.request import Request
    from opt.reward import Reward
    from opt.improve import Improve
    from opt.select import Select

    god_given_criteria =  "You are an expert bundle evaluator. Use this criteria to help you evaluate the bundle:\n"\
                          "- 1 - Poor: Items do not correlate with each other or the intent.\n"\
                          "- 2 - Needs Improvement: Some items are connected, but multiple modifications are needed to ensure the bundle's acceptability.\n"\
                          "- 3 - Almost: Only one modification is needed to guarantee the acceptability of the bundle.\n"\
                          "- 4 - Acceptable: Not perfect, but the intended user would find the bundle appealing.\n"\
                          "- 5 - Excellent: No flaws with the bundle.\n"

    if consideration == "":
        json_addition = "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"\
                        "**JSON Schema:**\n"\
                        "```json\n"\
                        "{{\n"\
                        "score: float, bundle score out of 5 given to 2 decimal places\n"\
                        "}}"

    elif consideration == "1-2":
        json_addition = "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"\
                        "**JSON Schema:**\n"\
                        "```json\n"\
                        "{{\n"\
                        "score: float, bundle quality out of 5.\n"\
                        "is_poor_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"\
                        "is_acceptable_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"\
                        "}}"

    elif consideration == "3":
        json_addition = "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"\
                        "**JSON Schema:**\n"\
                        "```json\n"\
                        "{{\n"\
                        "score: float, bundle quality out of 5.\n"\
                        "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"\
                        "is_good_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"\
                        "score: float, bundle quality out of 5.\n"\
                        "}}"

    elif consideration == "4-5":

        json_addition = "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"\
                        "**JSON Schema:**\n"\
                        "```json\n"\
                        "{{\n"\
                        "score: float, bundle quality out of 5.\n"\
                        "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"\
                        "is_high_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"\
                        "}}"

    else:
        json_addition = "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"\
                        "**JSON Schema:**\n"\
                        "```json\n"\
                        "{{\n"\
                        "score: float, bundle quality out of 5.\n"\
                        "verdict: str, yes/no response, do not provide anything other than yes or no.\n"\
                        "}}"

    print(initial_prompt)
    print(metastructure)

    # Build the engine config directly (the repo's assets/*.yaml are missing, so the
    # upstream init_config cannot run). See build_config for the key inventory.
    conf = build_config()

    conf['initial_prompt'] = initial_prompt
    conf['inferring_reasons'] = inferring_reasons
    conf['refining_prompts'] = refining_prompts
    conf['augmenting_prompts'] = augmenting_prompts
    conf['json_addition'] = json_addition
    conf['metrics'] = constant_metrics
    conf['opt_type'] = mode
    conf['base_reward'] = 3
    conf['threshold'] = 0.95  # legacy: from when we optimised for a single perfect prompt.
    conf['case'] = consideration
    conf['convince'] = convincing

    conf['initialise_struct'] = "You are an expert in summarising thorough analyses. Please use the given template to organise the given analysis.\n"
    conf['initialise_judge'] = god_given_criteria
    conf['operation'] = """These are the operations:
Add: introduce another item to the bundle.
Remove: remove an item from the bundle.
Replace: replace an item in the bundle with another item.
Decompose: split the bundle into 2 or more bundles.
"""

    # Keys come from config.py (were google.colab.userdata in the notebook).
    conf['wandb_api_key'] = config.WANDB_KEY
    conf['openai_api_key'] = config.OPENAI_API_KEY

    # Shallow-loop override for --mock (falls back to the engine's yaml default).
    if search_depth is not None:
        conf['search_depth'] = search_depth

    opt_request = Request(conf)

    if conf.get('use_wandb'):
        import wandb
        wandb.login(key=conf['wandb_api_key'])
        run = wandb.init(
            project=f"PO4ISR_{conf['dataset']}_tune",
            config=conf,
        )
        text_table = wandb.Table(columns=["Input", "Prompt", "Reason", "Improved prompt", "Augumented prompt"])
        reward_table = wandb.Table(columns=["Prompt", "Reward"])
    else:
        text_table = None
        reward_table = None
    print("parameter initialization is complete")

    train_data = training_data
    val_data = validation_data

    val_data_values = list(val_data.values())
    val_input = [data['input'] for data in val_data_values]
    val_scores = [data['target_score'] for data in val_data_values]

    prompt_data = [{"prompts": data + "\n" + json_addition} for data in val_input]

    beam_candidate = []

    random.seed(conf['seed'])

    opt_reward = Reward(conf, opt_request)
    opt_improve = Improve(inferring_reasons, refining_prompts, augmenting_prompts, train_data, conf, opt_request)
    opt_select = Select(train_data, conf, opt_reward)

    print()
    print()
    print("==============")
    print("The apo algorithm is running...")
    print("==============")

    if conf['opt_type'] == 'single':
        beam_candidate.append(initial_prompt)
    elif conf['opt_type'] == 'structure':
        beam_candidate.append(metastructure)

    all_prompts = []
    runnings = 0
    keep_going = True

    while keep_going:
        print("Search depth: " + str(int(runnings + 1)))

        all_expanded_candidates = []
        beam_count = 1

        if conf['opt_type'] == 'single':
            for prompt in beam_candidate:
                print()
                print(f"beam_count: {beam_count}/{len(beam_candidate)}")
                print()
                # Expand.
                expanded_prompts = await opt_improve.run(prompt, table=text_table)
                all_expanded_candidates.extend(expanded_prompts)
                beam_count += 1

        # Select. ``top_1_prompt`` is legacy (single best candidate).
        if conf['opt_type'] == 'single':
            beam_candidate, top_1_prompt = await opt_select.run(all_expanded_candidates)

        all_prompts.append(beam_candidate)

        if runnings < 4:
            conf['threshold'] += -0.1

        runnings += 1

        # Safeguard against infinite loops.
        if runnings == conf['search_depth']:
            keep_going = False

    try:
        last_2_iterations = [item for sublist in all_prompts[-4:] for item in sublist]
    except Exception:
        last_2_iterations = all_prompts[-1]

    if conf['opt_type'] == 'single':
        top_3 = await get_top_3_prompts(last_2_iterations, val_data, opt_reward, reward_table)

    out_dir = config.OUTPUT_DIR / "final_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"Refined_{char}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump([top_3, all_prompts], f)

    print(f"Saved {filename}")
    return


# --------------------------------------------------------------------------- #
# Driver (ported from the commented notebook cell 13).                         #
# --------------------------------------------------------------------------- #

def build_evaluator_configs(domain):
    """The three role evaluator configs for a domain (bad / middle / good), each using
    the matching importance-graph prompt and consideration gate."""
    configs = []
    for role in config.ROLES:  # ["bad", "middle", "good"]
        configs.append({
            "char_base": f"{role}_evaluator_{domain}_importance",
            "initial_prompt": ROLE_INITIAL_PROMPT[role],
            "consideration": ROLE_CONSIDERATION[role],
        })
    return configs


async def run(domains, use_experts_modes, search_depth=None):
    """Loop over domains x {expert, no-expert} x {bad, middle, good} and refine each.

    ``use_experts`` toggles whether the expert annotation + adjudication signal
    (``inferring_reasons`` + ``convincing_prompt``) is used, or the score-only signal
    (``inferring_reasons_no_experts``, no convincing); the latter appends the
    ``_no_experts`` suffix to ``char`` (matching the notebook's naming)."""
    for domain in domains:
        train_data, val_data = load_domain_data(domain)

        for use_experts in use_experts_modes:
            if use_experts:
                reasons_template = inferring_reasons
                convincing = convincing_prompt
                suffix = ""
            else:
                reasons_template = inferring_reasons_no_experts
                convincing = ""
                suffix = "_no_experts"

            for cfg in build_evaluator_configs(domain):
                char = cfg["char_base"] + suffix
                print(f"\n########## EGPO refining: {char} ##########")
                await structure_refinement(
                    char=char,
                    initial_prompt=cfg["initial_prompt"],
                    inferring_reasons=reasons_template,
                    refining_prompts=refining_prompts,
                    augmenting_prompts=augmenting_prompts,
                    training_data=train_data,
                    validation_data=val_data,
                    constant_metrics=adding_metrics,
                    metastructure=None,
                    mode="single",
                    consideration=cfg["consideration"],
                    convincing=convincing,
                    search_depth=search_depth,
                )


def main():
    # NOTE: a full run is expensive (many LLM calls per config x 3 roles x 2 signal
    # modes x N domains) and requires an OpenAI key (config.OPENAI_API_KEY), the engine's
    # assets/*.yaml, and optionally a wandb key (config.WANDB_KEY). Use --mock to smoke
    # test the plumbing cheaply.
    parser = argparse.ArgumentParser(description="LLM4BEAR EGPO prompt optimization (stage 1)")
    parser.add_argument("--mock", action="store_true", help="cheap, shallow smoke test via mock_configs")
    args = parser.parse_args()

    if args.mock:
        domains = mock_configs.DOMAINS
        # Only the expert-signal variant, and a shallow beam search. search_depth MUST be
        # >= 3: the top-3 selection pools all_prompts[-4:], so a too-shallow run has too
        # few candidates to choose from (mock_configs.EGPO_SEARCH_DEPTH is 3).
        use_experts_modes = [True]
        search_depth = mock_configs.EGPO_SEARCH_DEPTH
    else:
        domains = config.DOMAINS
        use_experts_modes = [True, False]  # expert-guided and score-only
        search_depth = config.EGPO_SEARCH_DEPTH  # canonical depth (>= 3)

    asyncio.run(run(domains, use_experts_modes, search_depth))


if __name__ == "__main__":
    main()
