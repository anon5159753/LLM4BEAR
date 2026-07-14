# Code Contents in Order

1. Making_Surveys.ipynb: Code process for how the json surveys were made.
2. Human_Evaluation_Result_Analysis.ipynb: Use code to analyse the results of the item bundle comparison tasks done by human partipants.
3. Claude_evaluating_surveyed_bundles.ipynb: Claude to use same item bundle comparison task instructions to rate bundles.
4. Case_Study.ipynb: Double check for reasons why LLM and human ratings disagreed.

# Human/Claude Participant Instructions for Item Bundle Comparison


```text
Task: Compare original and modified item bundles (intended to be purchased together). Bundle positions are randomised (A/B).

Part 1 - Choose the better designed bundle: Consider:
- Was the bundle designed with a clear idea in mind?
- Do the items make sense to purchase together?
Part 2 - Rate each bundle (1-5)

{bundles_here}
Score:
1-2: Low-quality: No/weak relations between bundle items.
3: Needs improvement: One/two modifications needed.
4-5: High-quality: Reasonable bundle to purchase.

**## OUTPUT FORMAT:**
Print your reasoning first. Then, print the separator string `===JSON_START===` on a new line. Finally, provide a single JSON object:

```json
{{
  "preferred_bundle": "string (single letter A/B)"
  "Bundle_A_rating: ": integer (1-5)
  "Bundle_B_rating": integer (1-5)
}}
```
