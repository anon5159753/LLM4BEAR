"""Shared, stateless helpers used across the LLM4BEAR pipeline stages.

Contains the OpenAI async plumbing (with retry/backoff + batching), the
``===JSON_START===`` response parsers, bundle/score/verdict extractors, the JIT
cosine-similarity kernel, and a couple of small utilities. Nothing here depends on
the per-domain catalog data (that lives in the stage modules).
"""

import asyncio
import json
import random

import numpy as np
from numba import jit

import config

# --- OpenAI client (lazily built so importing this module never needs a key) ---
_async_client = None


def get_async_client():
    """Return a process-wide AsyncOpenAI client, building it on first use."""
    global _async_client
    if _async_client is None:
        from openai import AsyncOpenAI

        key = config.OPENAI_API_KEY
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it in a .env file or the environment "
                "before running a stage that calls the OpenAI API."
            )
        _async_client = AsyncOpenAI(api_key=key)
    return _async_client


async def single_request(user, system=None, seed_value=None):
    """One chat completion with exponential-backoff retry."""
    import openai

    client = get_async_client()
    if system:
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        message = [{"role": "user", "content": user}]

    for delay_secs in (2 ** x for x in range(0, 3)):
        try:
            response = await client.chat.completions.create(
                model=config.MODEL,
                messages=message,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                seed=seed_value,
            )
            return response.choices[0].message.content.strip()
        except openai.OpenAIError as e:
            jitter = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + jitter
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            await asyncio.sleep(sleep_dur)

    return None  # all retries failed


async def openai_request(prompts, system=None, batch_size=config.BATCH_SIZE, delay=0):
    """Run a list of ``{"prompts": <user text>}`` dicts in batches, seed=42."""
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tasks = [single_request(d["prompts"], system=system, seed_value=config.SEED) for d in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"✅ Sending batch {i // batch_size + 1} — sleeping for {delay}s...\n")
        await asyncio.sleep(delay)
    return results


# --- Response parsing (the LLM emits a ===JSON_START=== separator) ---
def extract_json_simple_replace(response_text):
    """Parse the JSON object that follows a '===JSON_START===' separator."""
    try:
        json_part = response_text.split("===JSON_START===")[1]
        first_brace = json_part.find("{")
        last_brace = json_part.rfind("}")
        json_string = json_part[first_brace: last_brace + 1]
        return json.loads(json_string)
    except IndexError:
        print("Error: The separator '===JSON_START===' was not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Could not find or parse a valid JSON object after the separator.")
        return None


def extract_reasoning_part(response_text):
    """Return the analysis text that appears before the '===JSON_START===' separator."""
    return response_text.split("===JSON_START===")[0].strip()


def extract_json_from_response(response_text):
    """Return the substring from the first '{' to the last '}', or None."""
    start_index = response_text.find("{")
    end_index = response_text.rfind("}")
    if start_index != -1 and end_index != -1:
        return response_text[start_index: end_index + 1]
    return None


def extract_bundle_score(response):
    json_schema = extract_json_simple_replace(response)
    if json_schema is not None:
        try:
            return float(json_schema["score"])
        except Exception:
            return None
    return None


def extract_bundle_verdict(response, consideration):
    json_schema = extract_json_simple_replace(response)
    if json_schema is None:
        return None
    try:
        if consideration == "1-2":
            v1 = json_schema["is_poor_quality_bundle"]
            v2 = json_schema["is_acceptable_quality_bundle"]
        elif consideration == "3":
            v1 = json_schema["needs_improvement_bundle"]
            v2 = json_schema["is_good_quality_bundle"]
        elif consideration == "4-5":
            v1 = json_schema["needs_improvement_bundle"]
            v2 = json_schema["is_high_quality_bundle"]
        else:
            return None
        return [v1.lower(), v2.lower()]
    except Exception:
        return None


def input_strings(intent_list, item_list):
    """Format (intent, items) pairs into 'Intent: ...\\nBundle Items:\\n1. ...' strings."""
    string_list = []
    for i in range(len(intent_list)):
        intent = intent_list[i]
        items = item_list[i]
        item_str = "\n".join([f"{j + 1}. {title}" for j, title in enumerate(items)])
        string_list.append(f"Intent: {intent}\nBundle Items:\n{item_str}\n")
    return string_list


@jit(nopython=True)
def matrix_cosine_line_jit(embeddings, new_embedding):
    """Cosine similarity of ``new_embedding`` against every row of ``embeddings``."""
    n = embeddings.shape[0]
    norms = np.sqrt(np.sum(embeddings ** 2, axis=1))
    new_norm = np.sqrt(np.sum(new_embedding ** 2))
    cosine_similarity_line = np.zeros((n), dtype=np.float32)
    for i in range(n):
        cosine_similarity_line[i] = np.dot(embeddings[i], new_embedding) / (norms[i] * new_norm)
    return cosine_similarity_line


def stochastic_decision(bundle_items):
    """Fallback add/remove decision when the LLM's operation is unusable."""
    l = len(bundle_items)
    if l == 2:
        return "add"
    elif l > 8:
        return "remove"
    else:
        chance_to_add = 0.65 + (3 - l) * 0.1
        return "add" if random.random() < chance_to_add else "remove"
