"""Stage 3 — BundleRefinement (canonical ``bundle_refinement_workflow``).

Iteratively improves seed bundles: three LLM evaluators score each bundle, a filter
decides retain/modify, a diagnosis picks an add/remove operation, and the bundle is
edited (remove one item, or retrieve+add the best candidate item) before being
re-evaluated. History is snapshotted every iteration and pickled at the end.

This is the graph-free canonical workflow (the notebook's ``no_graphs_workflow``); the
four ablation variants and the importance-graph add prompt have been removed. The add
prompt used here is ``prompts.create_add_item_prompt`` (was ``*_no_graphs``).

Fix vs. the notebook: ``evaluation_module`` now returns 5 values (including
``min_scores``) and passes both the average and min score into ``drugs_prompt`` — the
notebook's cell-8 version returned 4 and would ``ValueError`` inside this workflow.

Runnable but not run by default (needs an OpenAI key + GPU-ish embedding model). Use
``python stage3_refinement.py --mock`` for a cheap smoke test.
"""

import argparse
import asyncio
import copy
import json
import pickle
import sys

import numpy as np
import pandas as pd

import config
import mock_configs
from common import (
    extract_bundle_score,
    extract_bundle_verdict,
    extract_json_from_response,
    extract_json_simple_replace,
    extract_reasoning_part,
    input_strings,
    matrix_cosine_line_jit,
    openai_request,
    stochastic_decision,
)
from prompts import (
    adding_metrics,
    create_add_item_prompt,
    create_expand_candidates_prompt,
    create_remove_item_prompt,
    drugs_prompt,
    json_bad,
    json_good,
    json_middle,
)

# Domain -> index used by all the per-domain data lists.
DOMAIN_INDEX = {"clothing": 0, "electronic": 1, "food": 2}

# Category-key lists (used to turn per-item category one-hot dicts into product-type lists).
ELECTRONIC_CATEGORY_KEYS = [
    "Camera and Accessories", "Computers and Accessories", "Audio Equipment",
    "Tablets and Accessories", "Storage Solutions", "Networking Equipment",
    "Mobile Devices and Accessories", "Travel Accessories", "Gaming",
    "Home Entertainment Systems", "Miscellaneous Electronics", "Power Solutions",
    "Cables and Connectors", "Security Systems", "Car Technology and Accessories",
    "Photography and Camera Equipment", "Adapters and Cables",
    "Television and Accessories", "AV Setup", "GPS and Navigation Accessories",
    "Mobile Device Protection", "Streaming and Media", "PC Building and Assembly",
    "General Electronics", "Walkie Talkies and Communication Devices",
]
CLOTHING_CATEGORY_KEYS = [
    "Footwear", "Accessories", "Costumes and Themed Apparel",
    "Lingerie and Underwear", "Baby and Kids Clothing", "Activewear and Sportswear",
    "Fashion Accessories", "Seasonal and Thematic Products", "Electronics",
    "Children's Items", "Carrying Items", "Maintenance", "Clothing",
]
FOOD_CATEGORY_KEYS = [
    "Snacks", "Beverages", "Cooking Ingredients", "Breakfast Foods",
    "Sweets and Desserts", "Health Foods", "Canned and Packaged Foods", "Baby Food",
    "Condiments and Sauces", "Fruits and Vegetables", "Specialty Foods",
    "Dried and Preserved Foods", "Grains and Pasta", "Miscellaneous",
    "Gift Baskets and Food Gifts", "Dietary Specific Items",
    "Cooking Tools and Kitchen Goods", "Sweeteners", "Nuts and Seeds", "Coffee/Tea",
    "Vegetables and Beans", "Health-Conscious Options", "Prepared and Ready-Made Meals",
    "Ethnic and Specialty Foods", "Culinary Specialties",
]


class Data:
    """Container for all per-domain catalog data, populated by ``load_data``.

    Kept out of import time so ``import stage3_refinement`` is cheap and does not need
    the repo present. Lists are ordered [clothing, electronic, food] to match
    ``DOMAIN_INDEX``.
    """

    loaded = False
    model = None
    bundle_items_list = intents = metadata = None
    session_bundle = session_item = item_names = None
    metadata_jsons = domain_items = super_embeddings = None
    category_indices = all_products = None


D = Data()


def _bundlerec(domain, name):
    return config.BUNDLEREC_DATA / domain / name


def _category_indices(cat_guys, category_keys):
    indices = []
    for raw in cat_guys:
        dictionary = json.loads(raw)
        data = [dictionary.get(key, 0) for key in category_keys]
        indices.append([idx for idx, value in enumerate(data) if value == 1])
    return indices


def load_data():
    """Load every catalog artifact the workflow needs (embeddings, metadata, sessions).

    Heavy: reads many CSV/pickle files and loads a SentenceTransformer. Called once by
    ``main``; safe to call again (idempotent)."""
    if D.loaded:
        return
    from sentence_transformers import SentenceTransformer

    D.model = SentenceTransformer(config.EMBED_MODEL)

    order = ["clothing", "electronic", "food"]
    D.bundle_items_list = [pd.read_pickle(_bundlerec(d, "bundle_list_items.pkl")) for d in order]
    D.intents = [pd.read_csv(_bundlerec(d, "bundle_intent.csv")) for d in order]
    D.metadata = [pd.read_csv(_bundlerec(d, "merged_metadata.csv")) for d in order]
    D.session_bundle = [pd.read_csv(_bundlerec(d, "session_bundle.csv")) for d in order]
    D.session_item = [pd.read_csv(_bundlerec(d, "session_item.csv")) for d in order]
    D.item_names = [pd.read_csv(_bundlerec(d, "item_titles.csv")) for d in order]

    bd = config.BUNDLEREC_DATA
    with open(bd / "specific_clothing_product_metadata.pkl", "rb") as f:
        meta_clothing = pickle.load(f)
    with open(bd / "specific_electronic_product_metadata.pkl", "rb") as f:
        meta_electronic = pickle.load(f)
    with open(bd / "specific_food_product_metadata.pkl", "rb") as f:
        meta_food = pickle.load(f)
    D.metadata_jsons = [meta_clothing, meta_electronic, meta_food]

    with open(bd / "enriched_outputs_clothing.pkl", "rb") as f:
        n_clothing = len(pickle.load(f))
    with open(bd / "enriched_outputs_electronic.pkl", "rb") as f:
        n_electronic = len(pickle.load(f))
    with open(bd / "enriched_outputs_food.pkl", "rb") as f:
        n_food = len(pickle.load(f))

    D.domain_items = [
        [D.metadata[0].iloc[i]["titles"] for i in range(n_clothing)],
        [D.metadata[1].iloc[i]["titles"] for i in range(n_electronic)],
        [D.metadata[2].iloc[i]["titles"] for i in range(n_food)],
    ]

    rd = config.REFINEMENT_DIR
    D.super_embeddings = [
        np.load(rd / "clothing_item_token_embeddings.npy"),
        np.load(rd / "electronic_item_token_embeddings.npy"),
        np.load(rd / "food_item_token_embeddings.npy"),
    ]

    # Per-item category one-hots -> per-item candidate product-type lists.
    def load_cat(domain, keys, patch=None):
        with open(bd / f"all_product_types_{domain}.pkl", "rb") as f:
            all_products = pickle.load(f)
        with open(bd / f"item_categories_{domain}.pkl", "rb") as f:
            cat_guys = pickle.load(f)
        cat_guys = [extract_json_from_response(i) for i in cat_guys]
        if patch:
            for idx, val in patch.items():
                cat_guys[idx] = val
        return all_products, _category_indices(cat_guys, keys)

    # The notebook hand-patches one malformed food category entry (index 3255).
    food_patch = {3255: json.dumps({k: (1 if k == "Snacks" else 0) for k in FOOD_CATEGORY_KEYS})}
    all_clothing, clothing_idx = load_cat("clothing", CLOTHING_CATEGORY_KEYS)
    all_electronic, electronic_idx = load_cat("electronic", ELECTRONIC_CATEGORY_KEYS)
    all_food, food_idx = load_cat("food", FOOD_CATEGORY_KEYS, patch=food_patch)
    D.all_products = [all_clothing, all_electronic, all_food]
    D.category_indices = [clothing_idx, electronic_idx, food_idx]

    D.loaded = True


# --- Data-coupled helpers (read from D) ---
def making_product_type_list(category_indices, all_products):
    product_list = []
    for i in category_indices:
        product_list = product_list + all_products[i]
    return list(set(product_list))


def find_k_neighbours(new_item, embeddings, domain_items, k):
    new_embedding = D.model.encode(new_item, convert_to_tensor=False)
    sim_line = matrix_cosine_line_jit(embeddings, new_embedding)
    top_k_indices = np.argsort(sim_line)[-k:][::-1]
    nearest_items = [domain_items[i] for i in top_k_indices]
    return nearest_items, top_k_indices


def _token_string_parts(metasetter, k):
    """Build the '[part][part]...' metadata token string for one item."""
    if k in (0, 1):  # clothing / electronic
        standard_keys = ["product_type", "brand", "design_focus", "cost_tier"]
        list_keys = ["key_features"]
    else:  # food
        standard_keys = ["product_type", "brand", "flavor_profile", "cost_tier"]
        list_keys = ["dietary_considerations", "key_features"]
    parts = []
    for key in standard_keys:
        value = metasetter.get(key)
        if value:
            parts.append(str(value))
    for key in list_keys:
        feature_list = metasetter.get(key, [])
        if isinstance(feature_list, list):
            parts.extend(feature_list)
    return "".join([f"[{part}]" for part in parts]) if parts else ""


def making_product_type_list_for(domain, bundle_id):
    k = DOMAIN_INDEX[domain]
    return making_product_type_list(D.category_indices[k][bundle_id], D.all_products[k])


def bundle_info_provider(bundle_ID, domain):
    k = DOMAIN_INDEX[domain]
    original_session_ID = D.session_bundle[k].iloc[bundle_ID]["session ID"]
    item_ids = D.session_item[k][D.session_item[k]["session ID"] == original_session_ID]["item ID"].values

    session_item_titles, existing = [], []
    for item_id in item_ids:
        item_titles = D.item_names[k][D.item_names[k]["item ID"] == item_id]["titles"]
        if not item_titles.empty and item_titles.values[0] not in existing:
            session_item_titles.append(item_titles.values[0])
            existing.append(item_titles.values[0])

    session_item_ids = [D.metadata[k][D.metadata[k]["titles"] == names].index.tolist()[0] for names in session_item_titles]
    bundle_list = D.bundle_items_list[k]
    bundle_item_ids = [D.metadata[k][D.metadata[k]["titles"] == names].index.tolist()[0] for names in bundle_list[bundle_ID]]
    bundle_item_titles = [D.metadata[k].iloc[i]["titles"] for i in bundle_item_ids]
    bundle_intent = D.intents[k].iloc[bundle_ID]["intent"]
    return bundle_intent, bundle_item_ids, bundle_item_titles, session_item_ids, session_item_titles


def bundle_token_strings(intent_list, item_list, index_list, domain):
    k = DOMAIN_INDEX[domain]
    string_list = []
    for i in range(len(intent_list)):
        intent, items, indexes = intent_list[i], item_list[i], index_list[i]
        meta_list = [D.metadata_jsons[k][j] for j in indexes]
        token_list = [_token_string_parts(json.loads(meta_list[m]), k) for m in range(len(indexes))]
        item_str = "\n".join([f"[ID: {indexes[j]}]. {items[j]}\n{token_list[j]}\n" for j in range(len(indexes))])
        string_list.append(f"Intent: {intent}\nBundle Items:\n\n{item_str}\n")
    return string_list


def candidate_token_strings(item_list, index_list, domain):
    k = DOMAIN_INDEX[domain]
    token_list = []
    for i in range(len(index_list)):
        metasetter = json.loads(D.metadata_jsons[k][index_list[i]])
        token_list.append(_token_string_parts(metasetter, k))
    item_str = "\n".join([f"[ID: {index_list[j]}]. {item_list[j]}\n{token_list[j]}\n" for j in range(len(index_list))])
    return f"Candidate Items:\n\n{item_str}\n"


def filter_decision(bad_verdicts, bad_scores, middle_verdicts, middle_scores,
                    good_verdicts, good_scores, score_acceptability=[3.4, 3.9]):
    """Classify each bundle as retain vs modify from the three evaluators' yes/no
    verdicts, falling back to a score gate for the ambiguous ones."""
    l = len(bad_verdicts)
    unsure, modify, retain = [], [], []
    all_evaluators, all_verdicts = [], []
    min_scores = [min(bad_scores[i], middle_scores[i], good_scores[i]) for i in range(l)]
    average_scores = [np.round((bad_scores[i] + middle_scores[i] + good_scores[i]) / 3, 2) for i in range(l)]

    for i in range(l):
        all_evaluators.append([bad_scores[i], middle_scores[i], good_scores[i]])
        all_verdicts.append([bad_verdicts[i], middle_verdicts[i], good_verdicts[i]])

        if bad_verdicts[i][0] == "yes" and bad_verdicts[i][1] == "no":
            bad_verd = "yes"
        elif bad_verdicts[i][0] == "no" and bad_verdicts[i][1] == "yes":
            bad_verd = "no"
        else:
            bad_verd = None
            print("bad verdict error")

        if middle_verdicts[i][0] == "yes" and middle_verdicts[i][1] == "no":
            middle_verd = "yes"
        elif middle_verdicts[i][0] == "no" and middle_verdicts[i][1] == "yes":
            middle_verd = "no"
        else:
            middle_verd = None
            print("middle verdict error")

        if good_verdicts[i][0] == "no" and good_verdicts[i][1] == "yes":
            good_verd = "yes"
        elif good_verdicts[i][0] == "yes" and good_verdicts[i][1] == "no":
            good_verd = "no"
        else:
            good_verd = None
            print("good verdict error")

        if bad_verd == "no" and middle_verd == "no" and good_verd == "yes":
            retain.append(i)
        else:
            unsure.append(i)

    for i in unsure:
        if min_scores[i] > score_acceptability[0] and average_scores[i] > score_acceptability[1]:
            retain.append(i)
        else:
            modify.append(i)

    return modify, retain, all_evaluators, all_verdicts, average_scores, min_scores


def get_drugs(summary_response, bundle_items, bundle_intent, randomise=""):
    reasoning_part = extract_reasoning_part(summary_response)
    json_schema = extract_json_simple_replace(summary_response)
    if json_schema is None:
        return reasoning_part, stochastic_decision(bundle_items), bundle_intent
    try:
        decision = json_schema["operation"].lower()
        intended_direction = json_schema["new_bundle_intent"]
        if decision not in ["add", "remove"] or randomise == "yes":
            return reasoning_part, stochastic_decision(bundle_items), intended_direction
        return reasoning_part, decision, intended_direction
    except Exception:
        return reasoning_part, stochastic_decision(bundle_items), bundle_intent


def bundle_item_remover(bundle_items, bundle_indices, remove_response):
    json_schema = extract_json_simple_replace(remove_response)
    if json_schema is None:
        return bundle_items, bundle_indices, None
    try:
        decision = json_schema["remove_item"].lower()
        if decision == "yes":
            item_to_remove = json_schema["item_to_remove"]
            item_to_remove_id = json_schema["item_to_remove_id"]
            new_bundle_intent = json_schema["new_bundle_intent"]
            if item_to_remove in bundle_items:
                idx = bundle_items.index(item_to_remove)
                bundle_items.pop(idx)
                bundle_indices.pop(idx)
                return bundle_items, bundle_indices, new_bundle_intent
            elif item_to_remove_id in bundle_indices:
                idx = bundle_indices.index(item_to_remove_id)
                bundle_items.pop(idx)
                bundle_indices.pop(idx)
                return bundle_items, bundle_indices, new_bundle_intent
            return bundle_items, bundle_indices, None
        elif decision == "no":
            return bundle_items, bundle_indices, None
    except Exception:
        return bundle_items, bundle_indices, None
    return bundle_items, bundle_indices, None


def extended_candidates_extractor(candidate_responses, domain, num_candidates=2):
    k = DOMAIN_INDEX[domain]
    candidate_jsons = [extract_json_simple_replace(r) for r in candidate_responses]
    collected_items, collected_ids = [], []
    for i in range(len(candidate_responses)):
        json_string = candidate_jsons[i]
        collated_items, collated_ids = [], []
        if json_string is None:
            print(f"Warning: No valid JSON found for response index {i}. Skipping.")
            collected_items.append([])
            collected_ids.append([])
            continue
        for metasetter in json_string.get("suggestions", []):
            search_string = _token_string_parts(metasetter, k)
            if search_string:
                nearest_items, top_k_indices = find_k_neighbours(
                    search_string, D.super_embeddings[k], D.domain_items[k], num_candidates
                )
                collated_items = collated_items + nearest_items
                collated_ids = collated_ids + list(top_k_indices)
        collected_items.append(collated_items)
        collected_ids.append(collated_ids)
    return collected_items, collected_ids


def extended_candidates_mixer(bundle_items, session_items, session_ids, extended_items=None, extended_ids=None):
    if extended_items is None:
        extended_items = []
    if extended_ids is None:
        extended_ids = []
    candidate_items = copy.deepcopy(session_items)
    candidate_ids = copy.deepcopy(session_ids)
    for i in range(len(extended_items)):
        if extended_items[i] not in candidate_items:
            candidate_items.append(extended_items[i])
            candidate_ids.append(extended_ids[i])
    vetted_items, vetted_ids = [], []
    for i in range(len(candidate_items)):
        if candidate_items[i] not in bundle_items:
            vetted_items.append(candidate_items[i])
            vetted_ids.append(candidate_ids[i])
    return vetted_items, np.array(vetted_ids).tolist()


def bundle_item_adder(bundle_intent, bundle_items, bundle_indices, add_response, domain):
    k = DOMAIN_INDEX[domain]
    json_schema = extract_json_simple_replace(add_response)
    if json_schema is None:
        return bundle_intent, bundle_items, bundle_indices
    try:
        chosen_item_to_add = json_schema["chosen_item_to_add"]
        chosen_item_id = json_schema["chosen_item_id"]
        new_bundle_intent = json_schema["new_bundle_intent"]
    except Exception:
        return bundle_intent, bundle_items, bundle_indices

    matching_rows = D.metadata[k][D.metadata[k]["titles"] == chosen_item_to_add]
    if not matching_rows.empty:
        if chosen_item_to_add not in bundle_items:
            bundle_items.append(chosen_item_to_add)
            bundle_indices.append(matching_rows.index[0])
            return new_bundle_intent, bundle_items, bundle_indices
    try:
        new_bundle_item = D.metadata[k].iloc[chosen_item_id]["titles"]
        if new_bundle_item not in bundle_items:
            bundle_items.append(new_bundle_item)
            bundle_indices.append(chosen_item_id)
        return new_bundle_intent, bundle_items, bundle_indices
    except Exception:
        return new_bundle_intent, bundle_items, bundle_indices


# --- LLM evaluation ---
async def separate_consideration_scores(char, initial_prompt, sample_data, json_addition,
                                        constant_metrics="", consideration=""):
    if consideration == "":
        return
    prompt_list = [{"prompts": data + "\n" + json_addition} for data in sample_data]
    responses = await openai_request(prompt_list, initial_prompt + constant_metrics)
    verdicts = [extract_bundle_verdict(i, consideration) for i in responses]
    scores = [extract_bundle_score(i) for i in responses]
    out = config.ensure_output_dir() / "filter_results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{char}.pkl", "wb") as f:
        pickle.dump([responses, verdicts, scores], f)
    return responses, verdicts, scores


async def evaluation_module(charizards, prompts, input_strings, score_acceptability):
    """Score every bundle with the three evaluators, filter retain/modify, and produce a
    diagnosis summary for the bundles to modify.

    Returns 5 values: modify, retain, all_scores (average), summary_responses, min_scores.
    (The notebook's cell-8 version returned 4 and omitted min_scores — fixed here.)"""
    bad_responses, bad_verdicts, bad_scores = await separate_consideration_scores(
        char=charizards[0], initial_prompt=prompts[0], sample_data=input_strings,
        json_addition=json_bad, constant_metrics=adding_metrics, consideration="1-2")
    middle_responses, middle_verdicts, middle_scores = await separate_consideration_scores(
        char=charizards[1], initial_prompt=prompts[1], sample_data=input_strings,
        json_addition=json_middle, constant_metrics=adding_metrics, consideration="3")
    good_responses, good_verdicts, good_scores = await separate_consideration_scores(
        char=charizards[2], initial_prompt=prompts[2], sample_data=input_strings,
        json_addition=json_good, constant_metrics=adding_metrics, consideration="4-5")

    modify, retain, all_evaluators, all_verdicts, all_scores, min_scores = filter_decision(
        bad_verdicts, bad_scores, middle_verdicts, middle_scores, good_verdicts, good_scores,
        score_acceptability=score_acceptability)

    # Diagnosis for the bundles to modify (drugs_prompt now gets both avg and min score).
    summary_prompts = [
        drugs_prompt(
            input_strings[i],
            extract_reasoning_part(bad_responses[i]),
            extract_reasoning_part(middle_responses[i]),
            extract_reasoning_part(good_responses[i]),
            all_scores[i],
            min_scores[i],
        )
        for i in modify
    ]
    summary_prompts = [{"prompts": p} for p in summary_prompts]
    summary_responses = await openai_request(summary_prompts, "")
    return modify, retain, all_scores, summary_responses, min_scores


async def bundle_refinement_workflow(domain, char, prompts, starting_bundle_intents,
                                     starting_bundle_item_ids, starting_bundle_item_titles,
                                     session_items_ids=None, session_item_titles=None,
                                     starting_product_list=None, randomise="",
                                     num_iterations=1, score_acceptability=[3.9, 4]):
    """The canonical LLM4BEAR refinement loop (was ``no_graphs_workflow``)."""
    flags = [True for _ in range(len(starting_bundle_intents))]
    working_indices = [index for index, flag in enumerate(flags) if flag]

    bundle_intents = copy.deepcopy(starting_bundle_intents)
    bundle_items = copy.deepcopy(starting_bundle_item_titles)
    bundle_indices = copy.deepcopy(starting_bundle_item_ids)
    product_types = copy.deepcopy(starting_product_list) if starting_product_list is not None else None

    bundle_input_strings = input_strings(bundle_intents, bundle_items)
    modify, retain, all_scores, summary_responses, min_scores = await evaluation_module(
        charizards=[f"starting_workflow_bad_{char}", f"starting_workflow_middle_{char}", f"starting_workflow_good_{char}"],
        prompts=prompts, input_strings=bundle_input_strings, score_acceptability=score_acceptability)

    if summary_responses:
        print(summary_responses[0])

    for i in retain:
        flags[working_indices[i]] = False

    historical_intents = [copy.deepcopy(bundle_intents)]
    historical_bundle_items = [copy.deepcopy(bundle_items)]
    historical_bundle_indices = [copy.deepcopy(bundle_indices)]
    historical_bundle_scores = [copy.deepcopy(all_scores)]
    historical_min_scores = [copy.deepcopy(min_scores)]
    historical_flags = [copy.deepcopy(flags)]

    working_indices = [index for index, flag in enumerate(flags) if flag]

    for num_iter in range(num_iterations):
        print(f"\n===========================\n\nIteration {num_iter + 1}\n\n===========================\n")

        remove_indices, remove_summary, add_indices, add_summary = [], [], [], []

        for i in range(len(working_indices)):
            summary, operation, intent = get_drugs(
                summary_responses[i], bundle_items[working_indices[i]],
                bundle_intents[working_indices[i]], randomise)
            if operation == "remove":
                remove_indices.append(working_indices[i])
                remove_summary.append(summary)
                bundle_intents[working_indices[i]] = intent
            elif operation == "add":
                add_indices.append(working_indices[i])
                add_summary.append(summary)
                bundle_intents[working_indices[i]] = intent

        remove_scores = [all_scores[i] for i in remove_indices]
        remove_min_scores = [min_scores[i] for i in remove_indices]

        remove_intents = [bundle_intents[i] for i in remove_indices]
        remove_items = [bundle_items[i] for i in remove_indices]
        remove_ids = [bundle_indices[i] for i in remove_indices]
        add_intents = [bundle_intents[i] for i in add_indices]
        add_items = [bundle_items[i] for i in add_indices]
        add_ids = [bundle_indices[i] for i in add_indices]

        remove_strings = bundle_token_strings(remove_intents, remove_items, remove_ids, domain)
        add_strings = bundle_token_strings(add_intents, add_items, add_ids, domain)

        # --- REMOVE path ---
        remove_prompts = [create_remove_item_prompt(remove_strings[i], remove_summary[i], remove_scores[i], remove_min_scores[i])
                          for i in range(len(remove_indices))]
        remove_prompts = [{"prompts": p} for p in remove_prompts]
        remove_responses = await openai_request(remove_prompts, "")
        for i in range(len(remove_indices)):
            post_remove_items, post_remove_ids, _ = bundle_item_remover(remove_items[i], remove_ids[i], remove_responses[i])
            bundle_items[remove_indices[i]] = post_remove_items
            bundle_indices[remove_indices[i]] = post_remove_ids

        # --- ADD path (retrieve candidate items, then let the LLM pick the best) ---
        candidate_strings = []
        if product_types is not None:
            expand_prompts = [create_expand_candidates_prompt(add_strings[i], add_summary[i], product_types=product_types[add_indices[i]], domain=domain)
                              for i in range(len(add_indices))]
            expand_prompts = [{"prompts": p} for p in expand_prompts]
            expand_responses = await openai_request(expand_prompts, "")
            expanded_items, expanded_ids = extended_candidates_extractor(expand_responses, domain, num_candidates=2)
            for i in range(len(add_indices)):
                if session_items_ids is not None:
                    candidate_items, candidate_ids = extended_candidates_mixer(
                        add_items[i], session_item_titles[add_indices[i]], session_items_ids[add_indices[i]],
                        expanded_items[i], expanded_ids[i])
                else:
                    candidate_items, candidate_ids = extended_candidates_mixer(
                        add_items[i], expanded_items[i], expanded_ids[i])
                candidate_strings.append(candidate_token_strings(candidate_items, candidate_ids, domain))
        else:
            for i in range(len(add_indices)):
                candidate_items, candidate_ids = extended_candidates_mixer(
                    add_items[i], session_item_titles[add_indices[i]], session_items_ids[add_indices[i]])
                candidate_strings.append(candidate_token_strings(candidate_items, candidate_ids, domain))

        add_prompts = [create_add_item_prompt(add_strings[i], add_summary[i], candidate_strings[i])
                       for i in range(len(add_indices))]
        add_prompts = [{"prompts": p} for p in add_prompts]
        add_responses = await openai_request(add_prompts, "")
        for i in range(len(add_indices)):
            _, post_add_items, post_add_ids = bundle_item_adder(add_intents[i], add_items[i], add_ids[i], add_responses[i], domain)
            bundle_items[add_indices[i]] = post_add_items
            bundle_indices[add_indices[i]] = post_add_ids

        # --- Re-evaluate the still-working bundles ---
        working_indices = [index for index, flag in enumerate(flags) if flag]
        modified_intents = [bundle_intents[i] for i in working_indices]
        modified_items = [bundle_items[i] for i in working_indices]
        modified_input_strings = input_strings(modified_intents, modified_items)

        modify, retain, adjusted_scores, summary_responses, adjusted_min_scores = await evaluation_module(
            charizards=[f"iterative_workflow_bad_{char}_{num_iter + 1}", f"iterative_workflow_middle_{char}_{num_iter + 1}", f"iterative_workflow_good_{char}_{num_iter + 1}"],
            prompts=prompts, input_strings=modified_input_strings, score_acceptability=score_acceptability)

        for i in range(len(adjusted_scores)):
            all_scores[working_indices[i]] = adjusted_scores[i]
            min_scores[working_indices[i]] = adjusted_min_scores[i]

        for i in retain:
            flags[working_indices[i]] = False

        working_indices = [index for index, flag in enumerate(flags) if flag]

        # A degenerate bundle (<2 items) is flagged back to "keep working".
        for i in range(len(bundle_items)):
            if len(bundle_items[i]) < 2:
                flags[i] = True

        historical_intents.append(copy.deepcopy(bundle_intents))
        historical_bundle_items.append(copy.deepcopy(bundle_items))
        historical_bundle_indices.append(copy.deepcopy(bundle_indices))
        historical_bundle_scores.append(copy.deepcopy(all_scores))
        historical_min_scores.append(copy.deepcopy(min_scores))
        historical_flags.append(copy.deepcopy(flags))

    out = config.ensure_output_dir()
    with open(out / f"historical_bundle_changes_{domain}.pkl", "wb") as f:
        pickle.dump([historical_intents, historical_bundle_items, historical_bundle_indices,
                     historical_bundle_scores, historical_min_scores, historical_flags], f)

    return (historical_intents, historical_bundle_items, historical_bundle_indices,
            historical_bundle_scores, historical_min_scores, historical_flags)


# --- Prompt loading (stage-2 selection, with commented repo-supplied fallback) ---
def load_domain_prompts(domain):
    """Return [bad, middle, good] evaluator prompts for a domain.

    Canonical path: consume stage 2's best-ICC selection (output/selected_best_prompts.pkl).
    """
    with open(config.OUTPUT_DIR / "selected_best_prompts.pkl", "rb") as f:
        selected = pickle.load(f)
    return selected[domain]

    # --- Fallback: use the repo-supplied refined prompts directly, hand-picking the
    #     known-good top-3 index per role (electronic [1,1,2], clothing [2,2,0],
    #     food [0,2,0]). Uncomment this block (and comment the two lines above) to
    #     borrow the original prompts instead of a fresh stage-2 run.
    #
    # role_index = {
    #     "electronic": [1, 1, 2],
    #     "clothing": [2, 2, 0],
    #     "food": [0, 2, 0],
    # }[domain]
    # roles = ["bad", "middle", "good"]
    # prompts_out = []
    # for r, idx in zip(roles, role_index):
    #     char = f"{r}_evaluator_{domain}_importance_no_experts"
    #     with open(config.REFINEMENT_FINAL_PROMPTS / f"Refined_{char}.pkl", "rb") as f:
    #         top_3_prompts, _ = pickle.load(f)
    #     prompts_out.append(top_3_prompts[idx])
    # return prompts_out


def build_starting_inputs(domain, subset=None):
    """Build the per-seed starting inputs (intents, item ids/titles, session items,
    product-type lists) for a domain, optionally limited to the first ``subset`` seeds."""
    k = DOMAIN_INDEX[domain]
    n = len(D.bundle_items_list[k])
    if subset is not None:
        n = min(subset, n)

    intents, item_ids, item_titles, session_ids, session_titles, product_lists = [], [], [], [], [], []
    for i in range(n):
        bundle_intent, bundle_item_ids, bundle_item_titles, session_item_id, session_item_title = bundle_info_provider(i, domain)
        intents.append(bundle_intent)
        item_ids.append(bundle_item_ids)
        item_titles.append(bundle_item_titles)
        session_ids.append(session_item_id)
        session_titles.append(session_item_title)
        product_lists.append(making_product_type_list_for(domain, i))
    return intents, item_ids, item_titles, session_ids, session_titles, product_lists


async def run(domains, num_iterations, subset=None):
    load_data()
    for domain in domains:
        print(f"\n########## Refining domain: {domain} ##########")
        prompts = load_domain_prompts(domain)
        intents, item_ids, item_titles, session_ids, session_titles, product_lists = build_starting_inputs(domain, subset)
        await bundle_refinement_workflow(
            domain=domain,
            char=f"complete_{domain}_run",
            prompts=prompts,
            starting_bundle_intents=intents,
            starting_bundle_item_ids=item_ids,
            starting_bundle_item_titles=item_titles,
            session_items_ids=session_ids,
            session_item_titles=session_titles,
            starting_product_list=product_lists,
            randomise="",
            num_iterations=num_iterations,
        )
        print(f"Saved output/historical_bundle_changes_{domain}.pkl")


def main():
    parser = argparse.ArgumentParser(description="LLM4BEAR BundleRefinement (stage 3)")
    parser.add_argument("--mock", action="store_true", help="cheap smoke test via mock_configs")
    args = parser.parse_args()

    if args.mock:
        config.BATCH_SIZE = mock_configs.BATCH_SIZE
        domains = mock_configs.DOMAINS
        num_iterations = mock_configs.NUM_ITERATIONS
        subset = mock_configs.SUBSET
    else:
        domains = config.DOMAINS
        num_iterations = config.REFINEMENT_NUM_ITERATIONS
        subset = config.REFINEMENT_SUBSET

    asyncio.run(run(domains, num_iterations, subset))


if __name__ == "__main__":
    main()
