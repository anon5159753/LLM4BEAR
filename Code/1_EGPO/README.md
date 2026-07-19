## Code Contents in Order

1. EGPO.ipynb: Running the EGPO mechanism
2. Determining_Best_Prompts.ipynb: Determining which mode of EGPO performed the best (i.e., with/without natural language metrics, with/without natural language expert explanations, with/without importance graph reasoning)
3. Prompt_score_distributions.ipynb: Inter-evaluator agreement and showing score distributions.

## EGPO Prompts


### Initial "Harsh" Electronic Prompt:
```text
You are an expert bundle strategist tasked distinguishing a 1--2/5 quality bundle. 
Your response should be a single judgement, is this bundle a 1--2/5 bundle?: yes/no.

Your task is to diligently complete subtasks that can help guide your analysis:
1. Based on the stated intent, come up with how combinations of items can interact to fulfill the intent.
2. If from your understanding, the combinations do not meet the stated intent, develop a new intent that your combinations can fulfill.
3. Design a importance graph analysis for each bundle in the format:
a. [Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]
Reason: [Your explanation for this item's rank in this specific scenario]
b. [2nd Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]
Reason: [Your explanation]
(...and so on for all other items)
4. Based on your reasoning and analysis, you are to make two mutually exclusive yes/no judgements on the bundle.
5. Now, you are to analyse the bundle:

Follow this criteria when evaluating the bundle:
1-2 - Poor: Some items or no items are connected thematically, but more than one modification needs to be made to guarantee the acceptability of the bundle.
3-5 - Acceptable: Items are connected thematically, and complement each other.
```

#### Natural Language Metrics Added to Each Prompt:
```text
Functionality Integration: Describe how a user would utilize this collection of items to achieve their primary goal. Considering the entire workflow, is this a complete and logical set of items for the task, or is there an irrelevant or missing item? 
Similarity: What is the common theme or category that connects these items?
Complementarity: Are these items more valuable together than they would be if sold separately? Does the presence of one item create a clear reason to buy the other(s)?
Diversity: Does the variety of items in this bundle cater to a broad set of related needs for a single user, or does the mix of items seem unfocused and random?
```

#### Harsh Prompt Json for Consistent Output (Changes depending on if Harsh, Neutral or Lenient):
```text
After you have completed your full written analysis, print the separator string `===JSON_START===` on a new line. Then, on the next line, provide a single JSON object that summarizes your conclusions from Part 1. Do not include any other text after the separator.
**JSON Schema:**
```json
{{
score: float, bundle quality out of 5.
is_poor_quality_bundle: str, yes/no response, do not provide anything other than yes or no.
is_acceptable_quality_bundle: str, yes/no response, do not provide anything other than yes or no.
}}
```


### Infer Reasons Prompt:
```text
I'm trying to refine a prompt that evaluated the following bundle incorrectly: $error_case$
$llm_judge=(LLM score was too high/too low...)$ The reasoning given was: $llm_reasoning$

The expert gave the bundle a score of $true_score$, with the reasoning: $annotation$
First, write a brief analysis. Is the expert's argument convincing, logical, and specific? Treat the expert's reasoning as more accurate AI's reasoning unless it seems that expert overlooked a detail that the AI considered?
Second, if the expert seems correct, then you must identify what is the most important point to the expert, and what was most important to the AI which led to a disagreement.
Summarise the difference in reasoning between the LLM and the expert, and incorporate what specific criteria and information did the expert use that the LLM overlooked.
Give $num_feedbacks$ reasons of why the LLM got this example wrong.
Wrap each reason with <start> and <end>
```
### Refine Prompt:
```text
I'm trying to write a zero-shot evaluation prompt.
My current prompt is \"$prompt$\"
But this prompt gets the following example wrong: $error_case$
$llm_judge$.
Based on these example the problem with this prompt is that $reasons$.
Based on the above information, please write an improved prompt without including any bundle items.
The prompt should be wrapped with <START> and <END>.
The new prompt is:
```
### Augment Prompt:
```text
Generate a variation of the following instruction while maintaining the semantic meaning.
Input: $refined_prompt$
The prompt should be wrapped with <START> and <END>.
Output:
```

## Optimised Prompts

### "Harsh" Electronic Prompt:
```text
You are tasked with evaluating a bundle to determine if it qualifies for a 1-2/5 quality rating. Please provide a clear answer: does this bundle meet the standards for the 1-2/5 rating? Respond with "yes" or "no."

To assist in your assessment, please follow these steps:

1. **Functionality Evaluation**: Analyze the intended purpose of the bundle and assess how well the items work together to fulfill that purpose. Consider the broader context of the intended function, ensuring that you evaluate the functional roles of each item in relation to that purpose, including any complementary features that may enhance the overall utility.

2. **Functionality Reassessment**: If you determine that the items do not adequately meet the intended function, suggest an alternative use that the items could collectively serve, based on their shared characteristics and compatibility. Ensure that this alternative use is relevant and logical within the context of the items.

3. **Importance Ranking**: For each item in the bundle, establish an importance ranking using the following format:
   a. [Most Critical Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Explain the item's significance in this context, considering its importance to the overall function and purpose.]
   b. [2nd Most Critical Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Describe how this item contributes to the bundle's intended use.]
   (Continue this format for all items in the bundle.)

4. **Assessment Criteria**: Based on your analysis, provide two separate yes/no evaluations of the bundle according to these criteria:
   - 1-2: Poor - Few or no items are thematically connected, necessitating numerous adjustments for the bundle to be considered acceptable.
   - 3-5: Acceptable - Items are thematically cohesive and enhance one another.

5. **Comprehensive Review**: After completing the previous steps, present a thorough evaluation of the bundle, considering how effectively the items work together to achieve the intended use and any shortcomings in their combined effectiveness. Ensure your assessment reflects a detailed understanding of each item's role in the overall objective of the bundle, avoiding biases related to specific features or capabilities.

Your evaluation should demonstrate a nuanced understanding of the specific context of the intended use and the functional roles of each item within that context. Be diligent in your analysis to prevent misclassification based on narrow interpretations of the items' purposes. Consider all aspects of the intended use, including any relevant complementary functions that may enhance the overall effectiveness of the bundle. Additionally, be mindful of potential biases that may arise from preferences for modern technology or assumptions about the items' contexts.
```

### "Neutral" Electronic Prompt:
```text
You are assigned the responsibility of assessing bundles to ascertain whether they qualify for a 1-3/5 quality classification. Your answer should be a definitive yes or no regarding the bundle's compliance with the criteria for a 1-3/5 classification.

To assist you in your evaluation, please follow these steps:

1. **Understand the Purpose**: Carefully examine the intent behind the bundle. Grasp the specific context and meanings of the relevant terms, as well as the overall aim of the bundle. Pay attention to subtle details to prevent misinterpretation.

2. **Analyze Item Interactions**: With the clarified purpose in mind, evaluate how the items within the bundle interact to fulfill that intent. Determine if the items work together effectively to support the intended goal or if they diverge from it. Consider the broader implications of the intent beyond just the individual items.

3. **Evaluate Compatibility**: Consider whether the items are appropriate for achieving the stated purpose. Assess if they collaborate smoothly or if there are conflicts that could impede their collective objective. Look for synergies or inconsistencies that could affect overall effectiveness.

4. **Suggest Alternative Purposes**: If the current combinations of items do not align with the stated intent, propose a new purpose that the items could successfully achieve. This should reflect a clearer understanding of the roles of the items involved and how they might serve a different function.

5. **Create Importance Graphs**: For each item in the bundle, construct an importance graph formatted as follows:  
   a. [Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale for this item's ranking in this context]  
   b. [2nd Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale]  
   (...continue this format for all remaining items)

6. **Establish Judgment Criteria**: Based on your analysis, make two mutually exclusive yes/no determinations regarding the bundle using the following criteria:  
   - 1-3 - Needs Improvement: One or more modifications are required for the bundle to be considered acceptable.  
   - 4-5 - Good Quality: The bundle is well-constructed and does not require any changes.

7. **Provide Final Evaluation**: After completing the previous steps, present your final assessment of the bundle, utilizing the insights gained from your evaluations. Ensure that your analysis is thorough and demonstrates a deep understanding of both the intent and the items involved, avoiding biases related to individual product features, brand reputation, or unusual product types. Your evaluation should concentrate solely on the overall effectiveness of the bundle in achieving its intended purpose, considering the specific context of the intent and steering clear of unrelated concepts.

Be cautious of potential misinterpretations of terms and ensure that your evaluation captures the true essence of the intent behind the bundle. Your final decision should reflect a comprehensive understanding of how the items align with the stated intent and their collective capacity to fulfill that purpose.
```

### "Lenient" Electronic Prompt:
```text
You are a seasoned bundle analyst assigned to assess the effectiveness of a bundle designed for a particular purpose. Your response should provide a definitive answer: does this bundle qualify for a 4-5/5 quality rating? Please respond with "yes" or "no."

To guide your evaluation, please adhere to the following steps meticulously:

1. **Clarify the Purpose**: Accurately define the specific aim of the bundle. Concentrate on the distinct requirements associated with this aim, ensuring clarity and precision. Steer clear of vague terminology and clarify any ambiguous phrases to avoid misinterpretation.

2. **Evaluate Compatibility**: Investigate whether the components of the bundle are suitable for one another and pertinent to the defined aim. Analyze the variety of the items; if they are excessively alike or from the same brand or product line, they may not effectively achieve the intended purpose. If the current combinations do not align with the specified aim, suggest an alternative objective that the items could fulfill, ensuring that key terms are clearly defined.

3. **Rank Priorities**: For each item in the bundle, establish a priority ranking using the following format:
   a. [Top Priority Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale for this item's ranking in this context]
   b. [Next Priority Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale]
   (Continue this format for all remaining items in the bundle.)

4. **Criteria for Assessment**: Based on your evaluation, make two mutually exclusive yes/no judgments regarding the bundle. Use the following criteria for your decision:
   - 1-3: Requires Improvement - One or more changes are necessary for the bundle to meet its intended aim.
   - 4-5: High Quality - The bundle is ready for user acceptance without any changes.

5. **Conclusive Judgment**: After completing the preceding steps, provide your final evaluation of the bundle based on the defined aim and the compatibility of the items. Ensure that your assessment considers not only the individual components but also their collective ability to create a cohesive and effective solution for the stated aim. Highlight the diversity and distinctiveness of the items involved.

Be careful to avoid assumptions based on common associations or generalizations that may misinterpret the specific intent of the aim. Clarify any ambiguous terms to ensure accurate evaluations, especially when distinguishing between related concepts such as "memory" in the context of cognitive function versus data storage.
```

### "Harsh" Clothing Prompt:
```text
You are a skilled bundle evaluator responsible for assessing whether a bundle qualifies for a 1-2/5 quality rating. Your answer should be a straightforward decision: does this bundle meet the criteria for a 1-2/5 quality rating? Please respond with "yes" or "no."

To assist in your evaluation, please follow these detailed steps:

1. **Clarify Purpose**: Determine the specific goal of the bundle. Consider who the intended audience is and the context in which the items will be utilized. Ensure that the purpose is explicitly defined and directly addresses the required types of items. Pay close attention to the clarity of the intent.

2. **Review Item Categories**: Confirm that the items included in the bundle align with the specified categories relevant to the purpose. Make sure that all items strictly fit within the defined category. For instance, if the purpose is "WOMEN'S FOOTWEAR," verify that all items are types of footwear and not unrelated items.

3. **Evaluate Thematic Cohesion**: Analyze how the items are thematically connected and whether they enhance the overall purpose. Ensure that the thematic relationships do not undermine the necessity for the correct item categories.

4. **Establish Importance Ranking**: For each item in the bundle, assign an importance ranking as follows:  
   a. [Top Priority Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Explain why this item holds this ranking]  
   b. [Next Priority Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Provide your rationale]  
   (...continue this for all items)

5. **Define Evaluation Criteria**: After your analysis, make two separate yes/no determinations regarding the bundle. Use these evaluation criteria:  
   - 1-2 - Unsatisfactory: Few or no items meet the specific type requirements, and substantial changes are necessary for the bundle to be deemed acceptable.  
   - 3-5 - Satisfactory: All items fulfill the specific type requirements and exhibit thematic cohesion.

6. **Conclude Findings**: Wrap up your evaluation by summarizing your insights and delivering a final judgment on whether the bundle qualifies for a 1-2/5 quality rating.

Make sure your assessment considers the specific item types requested and the overall thematic unity of the bundle. Be careful not to misinterpret the purpose based solely on thematic links or current trends. Concentrate on the exact nature of each item in relation to the defined intent, and remain vigilant about distinguishing between closely related categories, such as different types of footwear. Remember that the intent may encompass a variety of styles and purposes within the broader category.
```

### "Neutral" Clothing Prompt:
```text
As a seasoned bundle analyst, your role is to evaluate whether a given bundle qualifies for a 1-3/5 quality classification. Please provide a straightforward answer: does this bundle meet the criteria for a 1-3/5 classification? Respond with either yes or no.

To assist in your evaluation, please follow these steps meticulously:

1. **Intent Analysis**: Determine the specific purpose of the bundle. Take into account the intended audience, context of usage, and expected outcomes. Ensure that the intent is well-defined and corresponds with the characteristics of the items included. Be careful to recognize that terms may have overlapping meanings (e.g., "bangle" and "bracelet") and that variations in style or material should not detract from the overarching category.

2. **Item Interaction**: Examine how the items in the bundle interact to achieve the identified intent. Assess whether the items enhance each other and contribute effectively to the overall goal of the bundle. Confirm that all items are relevant to the stated purpose and align with the intent, acknowledging that various styles or types can coexist within a larger category without compromising the bundle's coherence.

3. **Revised Intent Development**: If the items do not align with the initial intent, formulate a new intent that the items can successfully fulfill. This should reflect a clearer comprehension of the items' characteristics and their potential applications, ensuring that the new intent is distinct from the original.

4. **Importance Graph Analysis**: For each item in the bundle, construct an importance graph using this format:
   a. [Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale for this item's ranking in this context]
   b. [2nd Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale]
   (...continue this format for all remaining items)

5. **Judgment Criteria**: Based on your analysis and reasoning, make two mutually exclusive yes/no decisions regarding the bundle. Use the following criteria for your assessment:
   - 1-3 - Needs Improvement: One or more adjustments are necessary to ensure the bundle's acceptability.
   - 4-5 - Good Quality: The bundle is well-organized and requires no modifications.

6. **Final Analysis**: After completing the aforementioned steps, deliver a conclusive evaluation of the bundle based on your comprehensive analysis. Ensure that your assessment considers all facets of the bundle's intent, item interactions, and overall quality, while being mindful of any biases or misinterpretations related to the specific types or styles of items involved. Recognize that variations in item descriptions should not lead to an incorrect assessment of their collective intent.

Make sure your evaluation is thorough and accurately represents the intent and quality of the bundle, particularly in acknowledging the variety of item types and styles within the broader category.
```

### "Lenient" Clothing Prompt:
```text
You are a skilled bundle evaluator responsible for determining if a specific bundle qualifies as a 4-5/5 quality bundle. Your answer should be a straightforward judgment: does this bundle qualify as a 4-5/5 bundle? Respond with "yes" or "no."

To assist you in your evaluation, please follow these steps:

1. **Intent Clarification**: Accurately articulate the primary intent of the bundle. Ensure that this intent is precise and clear, taking into account the overall context of the items' functions, styles, and intended audiences. Pay special attention to any specifications regarding gender and age demographics.

2. **Item Cohesion Assessment**: Evaluate whether the items within the bundle successfully fulfill the defined intent. If the items do not align with the intent, suggest an alternative intent that the items could serve, considering the variety of styles, functions, and target audiences represented.

3. **Importance Ranking**: For each item in the bundle, develop an importance ranking in the following structure:
   a. [Top Priority Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Justify the ranking of this item concerning the intent, taking into account its function, style, and suitability for the target audience.]
   b. [Second Priority Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Justify the ranking of this item, emphasizing its role in supporting the overall intent and its relevance to the target audience.]
   (Continue this format for all items in the bundle.)

4. **Evaluation Criteria**: Based on your analysis, make two mutually exclusive judgments regarding the bundle:
   - 1-3: Requires Improvement - One or more adjustments are needed for the bundle to meet the acceptance criteria.
   - 4-5: High Quality - The bundle is acceptable as is, with no changes necessary.

5. **Conclusive Evaluation**: After completing the previous steps, review the bundle based on your insights and provide your final judgment.

Ensure that your assessment is comprehensive and considers the specific needs of the intent, avoiding any misinterpretations related to item categories, themes, or demographics. Be mindful of the diversity of styles and functions represented in the bundle, and remain vigilant against biases that may affect your evaluation.
```

### "Harsh" Food Prompt:
```text
You are a skilled bundle evaluator responsible for assessing the quality of a specific bundle by examining its thematic consistency and overall purpose. Your answer should be a straightforward judgment: does this bundle fall within the 1-2/5 quality range? Respond with "yes" or "no."

To assist in your evaluation, please adhere to the following steps:

1. **Clarify the Intended Purpose**: Clearly define the bundle's intended purpose. Analyze how the selected items collaborate to achieve that purpose, focusing on the specific category of items (e.g., baby formulas, snacks, drinks). Your assessment should be grounded in a comprehensive understanding of what characterizes that category, including any significant distinctions within it.

2. **Evaluate Alignment**: If you find that the items do not sufficiently align with the stated purpose, suggest an alternative purpose that the items could serve based on their features and potential interactions. Ensure that this new purpose is pertinent to the items' characteristics and context.

3. **Importance Graph Analysis**: Conduct an importance graph analysis for each item in the bundle, structured as follows:  
   a. [Most Significant Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale for this item's ranking in this context]  
   b. [Second Most Significant Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your rationale]  
   (...and continue this format for all remaining items)

4. **Evaluation Criteria**: Based on your analysis, provide two separate yes/no assessments regarding the bundle. Use these criteria for your evaluation:  
   - 1-2: Poor - Few or no items are thematically connected, and numerous adjustments are necessary for the bundle to be deemed acceptable.  
   - 3-5: Acceptable - Items are thematically related and enhance each other.

5. **Contextual Assessment**: Lastly, evaluate the bundle considering your findings, ensuring you account for the broader context of the items, their specific attributes, and their cultural relevance. Clearly distinguish between different types of items and their intended purposes, especially in situations where distinctions are crucial for understanding compatibility.

Your objective is to deliver a comprehensive and insightful evaluation that demonstrates an understanding of the subtleties involved in assessing the quality of a bundle, particularly regarding the specific category of items in question. Be attentive to recognizing essential differences within categories that may influence the overall assessment.
```

### "Neutral" Food Prompt:
```text
You are an expert bundle strategist tasked with evaluating the quality of a bundle based on its alignment with a specified intent. Your response should be a single judgment: is this bundle a 1-3/5 quality bundle? Answer with yes or no.

Your task is to diligently complete the following subtasks to guide your analysis step by step:

1. **Intent Analysis**: Clearly define the intent associated with the bundle. Analyze how the items within the bundle can interact to fulfill this intent. Consider the overarching purpose of the bundle and how each item contributes to achieving that purpose. Ensure that your definition of intent is specific and relevant to the items provided, avoiding assumptions about the need for variety in food groups unless explicitly stated.

2. **Intent Reevaluation**: If you determine that the current combinations do not adequately meet the stated intent, propose a new intent that the items could collectively fulfill. Be explicit about how this new intent aligns with the items, ensuring that it reflects the potential culinary applications and interactions of the items rather than a requirement for distinct food categories.

3. **Importance Graph Analysis**: For each item in the bundle, create an importance graph in the following format:
   a. [Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your explanation for this item's rank in this specific scenario, focusing on its relevance to the intent and its culinary applications]
   b. [2nd Most Important Bundle Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your explanation]
   (...and so on for all other items)

4. **Judgment Criteria**: Based on your reasoning and analysis, make two mutually exclusive yes/no judgments on the bundle. Follow this evaluation criteria:
   - 1-3: Needs Improvement - One or more modifications need to be made to guarantee the acceptability of the bundle.
   - 4-5: Good Quality - No modifications are necessary as this bundle is extremely well designed.

5. **Final Analysis**: After completing the above steps, provide a comprehensive analysis of the bundle, synthesizing your findings and justifying your final judgment. 

Ensure that your analysis is thorough and considers both the individual items and their collective contribution to the stated intent. Pay special attention to the relevance of each item to the intent, providing clear definitions and examples where necessary to avoid ambiguity. Avoid biases based on brand or type, and recognize that multiple items can coexist under a single intent. Be vigilant against misinterpretations that may arise from broad or vague definitions of intent, particularly regarding the diversity of culinary applications and flavor profiles. Remember that different forms of the same ingredient can serve distinct purposes in cooking, and evaluate the bundle accordingly.
```

### "Lenient" Food prompt:
```text
You are tasked with evaluating a bundle to determine if it meets the criteria for a 4-5/5 quality rating based on its alignment with the specified intent. Please provide a clear answer: does this bundle qualify as a 4-5/5 bundle? Your response should be either "yes" or "no."

To guide your assessment, please adhere to the following steps:

1. **Purpose Identification**: Clearly define the primary objective of the bundle based on the specified intent. Focus on the category or type of items that should be included, emphasizing how these items collectively serve the intended function rather than their individual characteristics.

2. **Item Synergy**: Evaluate how well the items in the bundle work together to achieve the defined purpose. Assess whether the items enhance one another and collaborate effectively to fulfill the common goal related to their intended use.

3. **Relevance Assessment**: Identify any items that do not align with the defined intent. If certain items are not suitable for the intended purpose, explain why and suggest alternative purposes that the items could fulfill, ensuring they remain relevant to the included items.

4. **Importance Assessment**: For each item in the bundle, create an importance assessment using the following structure:  
   a. [Most Important Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your justification for this item's significance]  
   b. [Second Most Important Item] — [Role: Primary/Secondary/Tertiary]  
   Reason: [Your justification]  
   (...and continue this for all other items)

5. **Evaluation Criteria**: Based on your analysis, make two distinct yes/no evaluations regarding the bundle:  
   - 1-3: Requires Improvement – One or more modifications are necessary for the bundle to meet the standards.  
   - 4-5: High Quality – The bundle is satisfactory as it stands for the intended user.

6. **Conclusive Evaluation**: After completing the previous steps, provide your final assessment of the bundle based on the defined criteria, ensuring that your judgment reflects the overall purpose and the interaction of the items.

When evaluating, be mindful of the specific intent behind the bundle and ensure that your analysis accurately reflects the relevance and synergy of the items included. Now, proceed to evaluate the bundle:
```
