"""Prompt templates for the BundleRefinement stage.

These are pure string builders (no data dependencies). The add-candidates prompt here
is the graph-free canonical version (``create_add_item_prompt``, ported from the
notebook's ``create_add_item_prompt_no_graphs``). The importance-graph variant used by
the removed ablation workflows is intentionally omitted.
"""

# Appended to every evaluator prompt (the four bundle-quality metrics).
adding_metrics = (
    "\nFunctionality Integration: Describe how a user would utilize this collection of items to achieve their primary goal. Considering the entire workflow, is this a complete and logical set of items for the task, or is there an irrelevant or missing item? \n"
    "Similarity: What is the common theme or category that connects these items?\n"
    "Complementarity: Are these items more valuable together than they would be if sold separately? Does the presence of one item create a clear reason to buy the other(s)?\n"
    "Diversity: Does the variety of items in this bundle cater to a broad set of related needs for a single user, or does the mix of items seem unfocused and random?\n"
)

# JSON output schemas appended to each evaluator (bad / middle / good considerations).
json_bad = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n```json\n{{\n"
    "is_poor_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_acceptable_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "score: float, bundle quality out of 5.\n}}"
)

json_middle = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n```json\n{{\n"
    "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_good_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "score: float, bundle quality out of 5.\n}}"
)

json_good = (
    "After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.\n"
    "**JSON Schema:**\n```json\n{{\n"
    "needs_improvement_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "is_high_quality_bundle: str, yes/no response, do not provide anything other than yes or no.\n"
    "score: float, bundle quality out of 5.\n}}"
)


def drugs_prompt(bundle_str, reasoning1, reasoning2, reasoning3, average_score, min_score):
    """Diagnosis prompt: synthesize the evaluators' feedback into a problem statement,
    rate the four metrics, and decide an add/remove operation + new intent."""
    prompt = f"""You are an Expert Bundle Designer with years of experience creating successful and coherent product packages. Your task is to analyze a failing bundle and capture all nuances given by the experts.

**## CONTEXT:**
You have received a bundle that was reviewed by three different evaluators. The bundle's quality score is not high enough.


{bundle_str}
* **Average Score:** {average_score}
* **Min Score:** {min_score}

* **Evaluator 1 Reasoning:**
{reasoning1}

* **Evaluator 2 Reasoning:**
{reasoning2}

* **Evaluator 3 Reasoning:**
{reasoning3}

---

**## YOUR TASK: Problem Analysis**

You must perform the following four steps:

1.  **Synthesize the Core Problem:** Read all three evaluator reasonings and combine them into a single, concise **Problem Statement**. This statement should identify the bundle's single biggest flaw (e.g., "The bundle lacks a cohesive theme," "It's missing an essential item," or "An item is incompatible with the rest").

2.  **List the Top 3 Flaws:** Based on your analysis and the evaluator feedback, list the three most significant reasons that contribute to the core problem you identified.

3. **Evaluate Metrics:** Give verdict of low, medium, high for each metric below.
{adding_metrics}

4. **Operation Decision:** Decide whether to add or remove items from the bundle depending on the verdict of each metric.
If low Functionality Integration: Add or remove items to be able to fulfill intended purpose of bundle.
If low Similarity: Add more items that are thematic to the intent.
If low Complementarity: Remove the noisy item.
If low Diversity: Add a more niche item that is thematic to the intent.

---

**## OUTPUT FORMAT:**
After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions. Do not include any other text after the separator.

**JSON Schema:**
```json
{{
  "problem_summary": "string",
  "Functionality Integration: string, must be low/medium/high",
  "Similarity: string, must be low/medium/high",
  "Complementarity: string, must be low/medium/high",
  "Diversity: string, must be low/medium/high",
  "operation": "string, must be add/remove ONLY PICK ONE",
  "new_bundle_intent": "string,  a concise 3 or 4 word summary of the intended direction of the new, improved bundle's theme and purpose. The intent must be specific and not a broad category. Do not include phrases like 'and accessories' or 'and tools'."
}}
````
"""
    return prompt


def create_remove_item_prompt(bundle_str, summary, average_score, min_score):
    """Prompt: decide whether removing a single item fixes the bundle, and which one."""
    prompt = f"""You are an Expert Bundle Designer with years of experience creating successful and coherent product packages. Your task is to analyze a failing bundle and decide if the REMOVE operation should be applied.

**## CONTEXT:**
You have received a bundle that was reviewed by three different evaluators. The bundle's quality score is not high enough.


{bundle_str}
* **Average Score:** {average_score}
* **Min Score:** {min_score}

* **Evaluator Summary:**
{summary}

---

**## YOUR TASK:**
You must perform the following two steps.

**Part 1: Summarize the Core Problem**
First, look at the evaluator reasoning summary and try to determine whether or not removing a **SINGLE** item would fix each of the main problems of the bundle.

**Part 2: Decide whether or not to apply Remove Operation**
Based on your analysis of the problem, decide whether or not to: `REMOVE` (yes/no).

---

**## OUTPUT FORMAT:**
After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions. Do not include any other text after the separator.

**JSON Schema:**
```json
{{
  "problem_summary": "string",
  "remove_item": "string, must be yes/no",
  "item_to_remove": "string, required for REMOVE, otherwise null if remove item is unnecessary",
  "item_to_remove_id": "int, required for REMOVE, otherwise null if operation is ADD",
  "new_bundle_intent": "string, a concise 3-5 word summary of the new, improved bundle's theme and purpose."
}}
```
"""
    return prompt


def create_expand_candidates_prompt(bundle_str, summary, product_types, domain):
    """Prompt: propose three ideal single-item additions (product-type + profile) drawn
    from the bundle's candidate product-type list. Food uses a slightly different schema."""
    k = {"clothing": 0, "electronic": 1, "food": 2}[domain]
    product_types_str = "\n".join([f"- {j}" for j in product_types])

    if k in (0, 1):
        schema_obj = """{{
          "product_type": "string",
          "brand": "string",
          "design_focus": "string",
          "target_user": "string",
          "cost_tier": "string",
          "key_features": ["string"]
        }}"""
    else:  # food
        schema_obj = """{{
          "product_type": "string",
          "brand": "string",
          "dietary_considerations": ["string"],
          "flavor_profile": "string",
          "cost_tier": "string",
          "key_features": ["string"]
        }}"""

    prompt = f"""You are an Expert Bundle Designer with years of experience creating successful and coherent product packages.

    **## CONTEXT:**
    Experts have decided that the following bundle requires a new item to be added.

    {bundle_str}

    Experts have provided an analysis summary for this bundle:
    {summary}

    ---

    **## YOUR TASK:**
    Your goal is to propose **three different and independent suggestions** for a single item to add to this bundle. Each suggestion should represent a potentially different way to improve the bundle.

    1. **Propose Three Product Types:** Based on the context, choose three distinct product types from the list below that would be good single additions.
    2. **Generate Ideal Characteristics:** For each of the three product types you chose, generate a full profile of characteristics (brand, design focus, etc.) that would make it a perfect fit for the existing bundle.

    **Candidate Product Types:**
    {product_types_str}

    ---

    **## OUTPUT FORMAT:**
    After your analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your three suggestions. Do not include any other text after the separator.

    **JSON Schema:**
    The output must be a JSON object with a single key "suggestions" which contains a list of three objects. Each object represents one ideal product to add.
    ```json
    {{
      "suggestions": [
        {schema_obj},
        {schema_obj},
        {schema_obj}
      ]
    }}
    ```"""
    return prompt


def create_add_item_prompt(bundle_str, summary, additional_items):
    """Prompt: pick the single best item to add from the candidate list and set the new
    intent. Graph-free canonical version (was ``create_add_item_prompt_no_graphs``)."""
    prompt = f"""You are an Expert Bundle Designer tasked with improving an existing bundle.

**## CONTEXT:**
You are analyzing a bundle that needs one new item.

{bundle_str}

Experts have provided analysis to aid you in the addition operation:
{summary}

{additional_items}

---

**## YOUR TASK:**
You must perform the following two parts.

**Part 1: Multi-Scenario Analysis**
For **EACH** item in the 'Candidate Items to Add', provide a thorough analysis of how the addition of each item would change the overall value of the bundle.


**Part 2: Final Recommendation**
After you have created a thorough analysis for every candidate, compare the outcomes. Decide which addition creates the most cohesive and valuable bundle overall. Summarize your final decision in the JSON format below.

---

**## OUTPUT FORMAT:**
First, provide your full written analysis from Part 1, showing the multiple analyses. After that is complete, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object with your final recommendation from Part 2. Do not include any other text after the JSON.

**JSON Schema:**
```json
{{
  "reasoning_for_choice": "string, your detailed justification that explains WHY you chose the final item after considering all scenarios.",
  "chosen_item_to_add": "string, the single best item you selected from the candidate list.",
  "chosen_item_id": "int, the ID of the chosen item.",
  "new_bundle_intent": "string, a concise 3-5 word summary of the new, improved bundle's theme and purpose."
}}
```"""
    return prompt
