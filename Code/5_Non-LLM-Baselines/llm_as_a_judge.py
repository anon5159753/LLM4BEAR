"""Localized LLM-as-a-Judge -- de-Colab'd port of Comparing_Baselines.ipynb.

MECHANICAL changes only; the evaluation logic is preserved verbatim:
  * removed drive.mount / google.colab.userdata / !git clone / !pip / autotime;
  * API key read from os.environ (API_KEY or OPENAI_API_KEY);
  * hardcoded /content/LLM4BEAR/... and /content/drive/MyDrive/evaluating_bundles/
    paths repointed to the local repo;
  * single_request / openai_request / separate_consideration_scores /
    intent_evaluation_module and the 1-5 intent-scoring loop are unchanged.

The judge model is OpenAI gpt-4o-mini (temperature 0, seed 42), exactly as the
notebook.  Our three non-LLM baselines feed make_bundle_strings -> input_strings
-> intent_evaluation_module identically to the existing gpt-4o-mini/claude/etc.
baselines -- they are just additional prediction dicts.

Entry point: see run_llm_judge.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import random
import re

import numpy as np
import openai
from openai import AsyncOpenAI

# --------------------------------------------------------------------------- #
# Paths (repointed from /content/... to the local tree)
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))
# Repo root is two levels up (Code/5_Non-LLM-Baselines -> Code -> repo root),
# where "4_Bundle Generation" lives.
LB = os.path.join(ROOT, "..", "..", "4_Bundle Generation")
DATA_DIR = os.path.join(LB, "data", "bundlerec")          # <domain>/test_set.npy
FINAL_PROMPTS = os.path.join(LB, "final_prompts")
OUR_OUTPUT = os.path.join(ROOT, "output")                 # <baseline>/<domain>.npy
EVAL_OUT = os.path.join(ROOT, "judge_scores")             # our score pickles land here
os.makedirs(EVAL_OUT, exist_ok=True)

# --------------------------------------------------------------------------- #
# API client (key from environment, not Colab userdata).  Constructed lazily so
# the module imports without a key for offline steps (e.g. make_bundle_strings
# schema verification).
# --------------------------------------------------------------------------- #
def _load_dotenv(path=os.path.join(ROOT, ".env")):
    """Minimal .env loader (no dependency). Populates os.environ from KEY=VALUE
    lines if not already set. The .env file is gitignored -- never committed."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
_async_client = None


def get_client():
    global _async_client
    if _async_client is None:
        key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set API_KEY or OPENAI_API_KEY (e.g. in .env).")
        _async_client = AsyncOpenAI(api_key=key)
    return _async_client

# per-domain evaluator-prompt selection indices (from notebook cell 8)
CHARIZARD_INDICES = {
    "electronic": [1, 1, 2],
    "clothing":   [2, 2, 0],
    "food":       [0, 2, 0],
}
BANDS = ["bad", "middle", "good"]

# --------------------------------------------------------------------------- #
# Prompt constants (verbatim from the notebook)
# --------------------------------------------------------------------------- #
json_bad = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n"
    "```json\n"
    "{{\n"
    "is_poor_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_acceptable_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "score: float, bundle quality out of 5.\n"
    "}}"
)
json_middle = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n"
    "```json\n"
    "{{\n"
    "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_good_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "score: float, bundle quality out of 5.\n"
    "}}"
)
json_good = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n"
    "```json\n"
    "{{\n"
    "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_high_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "score: float, bundle quality out of 5.\n"
    "}}"
)
adding_metrics = (
    "\nFunctionality Integration: Describe how a user would utilize this collection of items to achieve their primary goal. Considering the entire workflow, is this a complete and logical set of items for the task, or is there an irrelevant or missing item? \n"
    "Similarity: What is the common theme or category that connects these items?\n"
    "Complementarity: Are these items more valuable together than they would be if sold separately? Does the presence of one item create a clear reason to buy the other(s)?\n"
    "Diversity: Does the variety of items in this bundle cater to a broad set of related needs for a single user, or does the mix of items seem unfocused and random?\n"
)


# --------------------------------------------------------------------------- #
# Extraction / string helpers (verbatim)
# --------------------------------------------------------------------------- #
def load_pkl(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return None


def infer_intent_prompt(input_string):
    output_format = f"""
Analyze the following list of products and infer the specific underlying intent or use-case they represent:

{input_string}
**## TASK:**
1. **Analyze:** Look for logical connections (e.g., a specific hobby, a repair task, a seasonal event, or a targeted lifestyle need).
2. **Think Step-by-Step:** Briefly reason about why these items are grouped together.
3. **Summarize Intent:** Create a hyper-specific 3-4 word theme.

**## CONSTRAINTS for 'intent':**
- Must be **specific** (e.g., "Organic Backyard Tomato Gardening" instead of "Gardening Supplies").
- Avoid broad categories (e.g., do not use "Electronics" or "Kitchenware").
- **Strictly Prohibited:** Do not include filler phrases like "and accessories," "and tools," "related items," or "and equipment."

**## OUTPUT FORMAT:**
Print your reasoning first. Then, print the separator string `===JSON_START===` on a new line. Finally, provide a single JSON object:

```json
{{
  "intent": "string (3-4 words max)"
}}"""
    return output_format


def input_strings(item_list):
    string_list = []
    for i in range(len(item_list)):
        items = item_list[i]
        item_str = "\n".join([f"{k + 1}. {title}" for k, title in enumerate(items)])
        string_list.append(f"Intent: \nBundle Items:\n{item_str}\n")
    return string_list


def extract_json_simple_replace(response_text):
    if response_text is None:
        return None
    try:
        if "===JSON_START===" not in response_text:
            json_part = response_text
        else:
            json_part = response_text.split("===JSON_START===")[1]
        first_brace = json_part.find("{")
        last_brace = json_part.rfind("}")
        if first_brace == -1 or last_brace == -1:
            return None
        return json.loads(json_part[first_brace:last_brace + 1].strip())
    except Exception as e:
        print(f"Extraction error: {e}")
        return None


def extract_inferred_intent(response):
    js = extract_json_simple_replace(response)
    if js and "intent" in js:
        return str(js["intent"]).strip()
    return None


def extract_bundle_score(response):
    js = extract_json_simple_replace(response)
    if js is not None:
        try:
            return float(js["score"])
        except Exception:
            return None
    return None


def extract_bundle_verdict(response, consideration):
    js = extract_json_simple_replace(response)
    if js is None:
        return None
    try:
        if consideration == "1-2":
            return [js["is_poor_quality_bundle"].lower(), js["is_acceptable_quality_bundle"].lower()]
        if consideration == "3":
            return [js["needs_improvement_bundle"].lower(), js["is_good_quality_bundle"].lower()]
        if consideration == "4-5":
            return [js["needs_improvement_bundle"].lower(), js["is_high_quality_bundle"].lower()]
        return None
    except Exception:
        return None


def convert_products_to_indices(product_names):
    indices = []
    pattern = re.compile(r"(\d+)")
    for name in product_names:
        m = pattern.search(str(name).strip())
        if m:
            indices.append(int(m.group(1)) - 1)
    return indices


def make_bundle_strings(path, predictions_dict):
    """path is the domain dir containing test_set.npy (trailing sep included)."""
    test_set = np.load(os.path.join(path, "test_set.npy"), allow_pickle=True).tolist()
    all_test_keys = list(test_set.keys())
    new_dict, bundle_size_dict = {}, {}
    for key in all_test_keys:
        value = predictions_dict.get(key)
        session_items = test_set[key].split("|split|")
        empty_box, size_box = [], []
        if value is not None and isinstance(value, dict):
            for _bundle_id, content in value.items():
                try:
                    item_ids = convert_products_to_indices(content)
                    bundle_items = [session_items[j] for j in item_ids if j < len(session_items)]
                    if bundle_items:
                        empty_box.append(bundle_items)
                        size_box.append(len(bundle_items))
                except Exception:
                    continue
        new_dict[key] = input_strings(empty_box) if empty_box else []
        bundle_size_dict[key] = size_box if size_box else []
    return new_dict, bundle_size_dict


def clean_nones(input_list):
    return [item for item in input_list if item is not None]


def dict_to_list(input_dict):
    """{key: {bundleN: [strings]}} -> per-key flat list of bundle strings."""
    out = []
    for k in input_dict:
        per = []
        for j in input_dict[k]:
            for s in input_dict[k][j]:
                per.append(s)
        out.append(per)
    return out


# --------------------------------------------------------------------------- #
# Domain evaluator prompts
# --------------------------------------------------------------------------- #
def load_domain_prompts(domain):
    """Replicates cell 8: for each band load Refined_<band>_evaluator_<domain>_
    importance_no_experts.pkl -> (top_3_prompts, _) and pick the tuned index."""
    idx = CHARIZARD_INDICES[domain]
    prompts = []
    for b, band in enumerate(BANDS):
        fn = os.path.join(FINAL_PROMPTS,
                          f"Refined_{band}_evaluator_{domain}_importance_no_experts.pkl")
        top_3_prompts, _ = load_pkl(fn)
        prompts.append(top_3_prompts[idx[b]])
    return prompts  # [bad_prompt, middle_prompt, good_prompt]


# --------------------------------------------------------------------------- #
# Async request + scoring (verbatim logic; pickle path repointed local)
# --------------------------------------------------------------------------- #
async def single_request(user, system=None, seed_value=None):
    if system:
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        message = [{"role": "user", "content": user}]
    client = get_client()
    for delay_secs in (2 ** x for x in range(0, 3)):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=message,
                temperature=0,
                max_tokens=1500,
                seed=seed_value,
            )
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as e:
            sleep_dur = delay_secs + random.randint(0, 1000) / 1000.0
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            await asyncio.sleep(sleep_dur)
    return None


async def openai_request(prompts, system=None, batch_size=128, delay=0):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tasks = [single_request(d["prompts"], system=system, seed_value=42) for d in batch]
        results.extend(await asyncio.gather(*tasks))
        print(f"batch {i // batch_size + 1} sent; sleeping {delay}s")
        await asyncio.sleep(delay)
    return results


async def separate_consideration_scores(char, initial_prompt, sample_data, json_addition,
                                         constant_metrics="", consideration="", save=""):
    if consideration == "":
        return
    prompt_list = [{"prompts": data + "\n" + json_addition} for data in sample_data]
    responses = await openai_request(prompt_list, initial_prompt + constant_metrics)
    verdicts = [extract_bundle_verdict(i, consideration) for i in responses]
    scores = [extract_bundle_score(i) for i in responses]
    filename = os.path.join(EVAL_OUT, f"{save}{char}.pkl")
    with open(filename, "wb") as f:
        pickle.dump([responses, verdicts, scores], f)
    return responses, verdicts, scores


async def intent_evaluation_module(charizards, prompts, input_strings_list):
    intent_prompt_list = [{"prompts": infer_intent_prompt(data)} for data in input_strings_list]
    intent_responses = await openai_request(intent_prompt_list)

    inferred_data = []
    for i in range(len(intent_responses)):
        intent_text = extract_inferred_intent(intent_responses[i])
        original_text = input_strings_list[i]
        if intent_text:
            inferred_data.append(original_text.replace("Intent:", f"Intent: {intent_text}"))
        else:
            inferred_data.append(original_text)

    bad = await separate_consideration_scores(
        char=charizards[0], initial_prompt=prompts[0], sample_data=inferred_data,
        json_addition=json_bad, constant_metrics=adding_metrics, consideration="1-2", save="intent_")
    middle = await separate_consideration_scores(
        char=charizards[1], initial_prompt=prompts[1], sample_data=inferred_data,
        json_addition=json_middle, constant_metrics=adding_metrics, consideration="3", save="intent_")
    good = await separate_consideration_scores(
        char=charizards[2], initial_prompt=prompts[2], sample_data=inferred_data,
        json_addition=json_good, constant_metrics=adding_metrics, consideration="4-5", save="intent_")
    return list(bad), list(middle), list(good)


# --------------------------------------------------------------------------- #
# High-level driver for one baseline x domain
# --------------------------------------------------------------------------- #
def _flatten_bundle_strings(bundle_string_dict):
    """{key: [prompt_str, ...]} -> single flat list of all bundle prompt strings."""
    flat = []
    for key in bundle_string_dict:
        for s in bundle_string_dict[key]:
            flat.append(s)
    return flat


def generate_bundle_stats(input_dict):
    """Verbatim port of notebook cell 16. input_dict = {run_key: {test_key: [sizes]}}."""
    avg_num_bundles_per_sess, avg_num_items_per_bundle, num_bundles_generated, session_count = [], [], [], []
    for run_keys in list(input_dict.keys()):
        num_bundles, bundle_sizes, count = [], [], 0
        run_dict = input_dict[run_keys]
        for j in list(run_dict.keys()):
            values = run_dict[j]
            if values and len(values) > 0:
                num_bundles.append(len(values))
                for i in values:
                    bundle_sizes.append(i)
                count += 1
        avg_num_bundles_per_sess.append(sum(num_bundles) / count if count else 0)
        avg_num_items_per_bundle.append(sum(bundle_sizes) / sum(num_bundles) if num_bundles else 0)
        session_count.append(count)
        num_bundles_generated.append(sum(num_bundles))
    return avg_num_bundles_per_sess, avg_num_items_per_bundle, num_bundles_generated, session_count


async def evaluate_baseline(baseline, domain, predictions_dict):
    """Run the 3-band intent evaluation for one baseline x domain.
    Returns dict with band_scores/average/n plus the bundle strings + sizes (for
    the notebook-style end-of-run pickle dump)."""
    domain_dir = os.path.join(DATA_DIR, domain)
    bundle_strings, sizes = make_bundle_strings(domain_dir + os.sep, predictions_dict)
    flat = _flatten_bundle_strings(bundle_strings)
    if not flat:
        return {"band_scores": [None, None, None], "average": None, "n": 0,
                "bundle_strings": bundle_strings, "sizes": sizes}

    prompts = load_domain_prompts(domain)
    charizards = [f"{baseline}_{domain}_bad", f"{baseline}_{domain}_middle", f"{baseline}_{domain}_good"]
    bad, middle, good = await intent_evaluation_module(charizards, prompts, flat)

    band_scores = []
    for band in (bad, middle, good):
        s = clean_nones(band[2])
        band_scores.append(sum(s) / len(s) if s else None)
    valid = [b for b in band_scores if b is not None]
    return {"band_scores": band_scores, "average": (sum(valid) / len(valid)) if valid else None,
            "n": len(flat), "bundle_strings": bundle_strings, "sizes": sizes}


def band_pkls_exist(baseline, domain):
    """Resume helper: True if all 3 band score pkls already exist for this run."""
    return all(os.path.exists(os.path.join(EVAL_OUT, f"intent_{baseline}_{domain}_{band}.pkl"))
               for band in BANDS)
