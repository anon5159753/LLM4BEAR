import random
from opt.utils import detect_error, extract_edit_prompt, extract_bundle_score, extract_reasoning_part, extract_bundle_verdict

class Improve():
    def __init__(self,
                inferring_reasons, 
                refining_prompts, 
                augmenting_prompts, 
                train_data,
                config,
                request_model):
        self.inferring_reasons = inferring_reasons
        self.refining_prompts = refining_prompts
        self.augmenting_prompts = augmenting_prompts
        self.train_data = train_data
        self.config = config
        self.request = request_model
        self.used_data = []
    
    async def evaluate_collect_error(self, system_prompt, data, metastructure=None, mode='single'):
        if mode == 'single':
            errors_list = []
            prompt_list = [{"prompts": val['input'] + "\n" + self.config['json_addition']} for val in data]
            
            # print("System Prompt:\n", system_prompt)
            # print()
            # print("User Prompt:\n", prompt_list[0]["prompts"])
            consideration = self.config['case']


            responses = await self.request.openai_request(prompt_list, system_prompt + self.config['metrics'])

            print()
            print(responses[0])
            print()



            validation_targets = [val['target_score'] for val in data]

            big_bundle_scores = [extract_bundle_score(i) for i in responses]



            verdicts = [extract_bundle_verdict(i, consideration) for i in responses]

            for i in range(len(responses)):
                print(validation_targets[i], big_bundle_scores[i], "||", verdicts[i])
            print()

            if consideration == "1-2":
                for i in range(len(responses)):

                    if validation_targets[i] > 2 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM was **OVERLY HARSH** and thought this bundle was poor whilst the expert thought the bundle was more **HIGH QUALITY** at {validation_targets[i]}/5."
                        errors_list.append(error)

                    elif validation_targets[i] < 3 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM thought this bundle was acceptable whilst the expert thought the bundle was very poor at {validation_targets[i]}/5."
                        errors_list.append(error)

            elif consideration == "4-5":
                for i in range(len(responses)):

                    if validation_targets[i] > 3 and verdicts[i][1] == "no" and verdicts[i][0] == "yes":

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM was **OVERLY HARSH** and thought this bundle was more poor whilst the expert thought the bundle was **EXCELLENT** at {validation_targets[i]}/5."
                        errors_list.append(error)

                    elif validation_targets[i] < 4 and verdicts[i][1] == "yes" and verdicts[i][0] == "no":

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM thought this bundle was excellent whilst the expert thought the bundle was quite poor at {validation_targets[i]}/5."
                        errors_list.append(error)


            elif consideration == "3":
                for i in range(len(responses)):

                    if validation_targets[i] > 3 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM was **OVERLY HARSH** and thought this bundle was not acceptable and needed modification whilst the expert thought it was an acceptable {validation_targets[i]}/5 bundle."
                        errors_list.append(error)

                    elif validation_targets[i] < 4 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM though this bundle was acceptable, needing no modifications whilst the expert thought the bundle needed modifications, evaluating it as a {validation_targets[i]}/5 bundle."
                        errors_list.append(error)

            elif consideration == "":


                for i in range(len(responses)):
                    given_score = extract_bundle_score(responses[i])
                    validation_score = validation_targets[i]
      

                    if validation_score > 3.6 and given_score < 3.4:

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = given_score
                        error['true_score'] = validation_score
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = "poor and needed refining whilst the expert thought the bundle was acceptable"
                        errors_list.append(error)
            
                    elif validation_score < 3.4 and given_score > 3.6:

                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = given_score
                        error['true_score'] = validation_score
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = "acceptable and did not need refining whilst the expert thought the bundle needed modification"
                        errors_list.append(error)
        
            return errors_list

        elif mode == 'structure':


            consideration = self.config['case']

            errors_list = []


            prompt_list = [{"prompts": val['input'] + "\n" + metastructure + "\n" + self.config['json_addition']} for val in data]
            
            responses = await self.request.openai_request(prompt_list, system_prompt + self.config['metrics'])
            validation_targets = [val['target_score'] for val in data]

            verdicts = [extract_bundle_verdict(i, consideration) for i in responses]

            # prompt_list = [{"prompts": val['input']} for val in data]


            # print("System Prompt:\n", system_prompt)
            # print()
            # print("User Prompt:\n", prompt_list[0]["prompts"])
            

            # initial_reasoning = await self.request.openai_request(prompt_list, system_prompt + self.config['metrics'])

            # print(metastructure)
            # print()
            # print('========================================')
            # print()

            # structure_list = [{"prompts": data[i]['input'] + initial_reasoning[i]} for i in range(len(prompt_list))]

            # meta_structure_enforced = await self.request.openai_request(structure_list, system=self.config['initialise_struct'] + metastructure + self.config['operation'])

            # print(meta_structure_enforced[0])
            # print()
            # print('========================================')
            # print()

            # judging_list = [{"prompts":  data[i]['input'] + "\n" + meta_structure_enforced[i] + "\n" + self.config['json_addition']} for i in range(len(prompt_list))]

            # judgements = await self.request.openai_request(judging_list, system=self.config['initialise_judge'])
            
            # print(judgements[0])
            # print()
            # print('========================================')
            # print()

            # validation_targets = [val['target_score'] for val in data]


            if consideration == "1-2":
                for i in range(len(responses)):

                    if validation_targets[i] > 2 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":
                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM thought this bundle was poor whilst the expert thought the bundle was more acceptable at {validation_targets[i]}/5."
                        errors_list.append(error)

                    elif validation_targets[i] < 3 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":
                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM thought this bundle was acceptable whilst the expert thought the bundle was very poor at {validation_targets[i]}/5."
                        errors_list.append(error)

            elif consideration == "4-5":
                for i in range(len(responses)):

                    if validation_targets[i] > 3 and verdicts[i][1] == "no" and verdicts[i][0] == "yes":
                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM thought this bundle was more poor whilst the expert thought the bundle was more excellent at {validation_targets[i]}/5."
                        errors_list.append(error)

                    elif validation_targets[i] < 4 and verdicts[i][1] == "yes" and verdicts[i][0] == "no":
                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM thought this bundle was excellent whilst the expert thought the bundle was quite poor at {validation_targets[i]}/5."
                        errors_list.append(error)


            elif consideration == "3":
                for i in range(len(responses)):

                    if validation_targets[i] > 3 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":
                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM thought this bundle was almost acceptable and evaluated it as 3/5 whilst the expert thought it was a {validation_targets[i]}/5 bundle."
                        errors_list.append(error)

                    elif validation_targets[i] < 4 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":
                        error = {}
                        error['reasoning'] = extract_reasoning_part(responses[i])
                        error['input'] = data[i]['input']
                        error['output'] = responses[i]
                        error['given_score'] = verdicts[i]
                        error['true_score'] = validation_targets[i]
                        error['annotation'] = data[i]['annotations']
                        error['llm_judge'] = f"The LLM did not think this bundle was almost acceptable whilst the expert thought the bundle was very close to acceptability at 3/5."
                        errors_list.append(error)


            # for i in range(len(judgements)):
            #     given_score = extract_bundle_score(judgements[i])
            #     validation_score = validation_targets[i]

            #     if validation_score > 3.6 and given_score < 3.4:

            #         error = {}
            #         error['reasoning'] = meta_structure_enforced[i]
            #         error['input'] = data[i]['input']
            #         error['output'] = judgements[i]
            #         error['given_score'] = given_score
            #         error['true_score'] = validation_score
            #         error['annotation'] = data[i]['annotations']
            #         error['llm_judge'] = "poor and needed refining whilst the expert thought the bundle was acceptable"
            #         errors_list.append(error)
        
            #     elif validation_score < 3.4 and given_score > 3.6:

            #         error = {}
            #         error['reasoning'] = meta_structure_enforced[i]
            #         error['input'] = data[i]['input']
            #         error['output'] = judgements[i]
            #         error['given_score'] = given_score
            #         error['true_score'] = validation_score
            #         error['annotation'] = data[i]['annotations']
            #         error['llm_judge'] = "acceptable and did not need refining whilst the expert thought the bundle needed modification"
            #         errors_list.append(error)
        
            return errors_list          


    async def generate_similar_prompt(self, prompt_list):
        # Step 1: Augment each prompt once to create a temporary list
        tmp = self.augmenting_prompts
        augmented_prompts_once = [
            tmp.replace("$refined_prompt$", prompt) 
            for prompt in prompt_list
        ]

        # Step 2: Create a flattened list where each augmented prompt is repeated
        # 'addition_sample' times, which is what your proposed line does.
        flattened_prompts = [
            p for p in augmented_prompts_once 
            for _ in range(self.config['addition_sample'])
        ]
        
        # Step 3: Format the flattened list into the dictionary format
        formatted_prompts = [{"prompts": p} for p in flattened_prompts]

        # print("Asking us to make an augmented prompt. Actually check if we've used the augmenting prompt and replaced the real prompt\n")

        # print(formatted_prompts[0]["prompts"])

        # Step 4: Send all formatted prompts in one asynchronous batch
        # We assume the augmentation prompt has no system message, hence system=None
        responses = await self.request.openai_request(prompts=formatted_prompts)
    
        return responses



    async def run(self, prompt, metastructure=None, table=None):
  

        train_data_values = list(self.train_data.values())
    
        # Sample from the list of values to get a list of dictionaries
        batch_data = random.sample(train_data_values, self.config['batch_size'])
      
        # batch_data = dict(random.sample(list(self.train_data.items()), self.config['batch_size']))
        
        self.used_data += batch_data

        # print("Evaluating Bundles and collecting errors")

        errors_list = await self.evaluate_collect_error(prompt, batch_data, metastructure=metastructure, mode=self.config['opt_type']) 

        if self.config['convince'] != "":

            convincing_prompt = self.config['convince']

            convince_prompts = [convincing_prompt.replace("$error_case$", error['input']).replace("$given_score$", str(error['given_score'])).replace("$true_score$", str(error['true_score'])).replace("$annotation$", error['annotation'].replace("$llm_reasoning$", error['reasoning']).replace("$llm_judge$", error['llm_judge'])) for error in errors_list]

            convince_prompts = [{"prompts": prompt} for prompt in convince_prompts]


            convince_list = await self.request.openai_request(convince_prompts, system='')

            try:
                print(convince_list[0])
            except:
                print("no convinces")

            convinces = [extract_bundle_verdict(i, self.config['case']) for i in convince_list]


            convincing_errors = [
                errors_list[i] for i in range(len(convinces)) if convinces[i] == "yes"
            ]
            if len(convincing_errors) != 0:
                error_list = convincing_errors


        try:
            errors_group = random.sample(errors_list, self.config['error_batch_size'])
        except:
            errors_group = errors_list

        # inferring_reasons = self.inferring_reasons.replace("$prompt$", prompt).replace("$num_feedbacks$", str(self.config['num_feedbacks'])) 




        inferring_reasons = self.inferring_reasons.replace("$num_feedbacks$", str(self.config['num_feedbacks'])) 

        tmp_infer_prompt = inferring_reasons
        # error_prompts = [tmp_infer_prompt.replace("$error_case$", error['input']).replace("$given_score$", str(error['given_score'])).replace("$true_score$", str(error['true_score'])) for error in errors_group]
        error_prompts = [tmp_infer_prompt.replace("$error_case$", error['input']).replace("$given_score$", str(error['given_score'])).replace("$true_score$", str(error['true_score'])).replace("$annotation$", error['annotation'].replace("$llm_reasoning$", error['reasoning']).replace("$llm_judge$", error['llm_judge'])) for error in errors_group]

        # print()
        # print("Trying to infer reasons for errors")

        error_prompts = [{"prompts": prompt} for prompt in error_prompts]

        # print("Sending in trying to find errors in example, there are no system prompts, please check our shit. I've added the scores and later expert annotations\n\n", error_prompts[0]["prompts"])

        gradients = await self.request.openai_request(error_prompts, system='')



        if self.config['opt_type'] == 'single':
            refining_prompts = self.refining_prompts.replace("$prompt$", prompt)
        elif self.config['opt_type'] == 'structure':
            refining_prompts = self.refining_prompts.replace("$prompt$", metastructure)


        tmp_refine_prompt = refining_prompts
        tmp_refine_prompts = [tmp_refine_prompt.replace("$error_case$", error['input'].replace("$llm_judge$", error['llm_judge'])) for error in errors_group]
        contents = [tmp_refine_prompts[i].replace("$reasons$", gradients[i]) for i in range(len(errors_group))]

        contents = [{"prompts": prompt} for prompt in contents]

        # print("Trying to refine prompts after injecting reasons\n")

        # print("No system prompts, but we've got reasons for why it might suck\n", contents[0]["prompts"])

        # Corrected: Passed a formatted list to the openai_request method
        edit_prompts_response = await self.request.openai_request(contents, system='')

        # Flatten the list of lists that extract_edit_prompt returns
        edit_prompt_list_flat = [
            item for sublist in [extract_edit_prompt(response) for response in edit_prompts_response] 
            for item in sublist
        ]

        # print(edit_prompt_list_flat[0])



        # print("We've refined prompts, now we're augmenting the refined prompts")

        more_candidate_prompts = await self.generate_similar_prompt(edit_prompt_list_flat)

        # print(more_candidate_prompts[0])

        augmented_prompt_list_flat = [
            item for sublist in [extract_edit_prompt(response) for response in more_candidate_prompts] 
            for item in sublist
        ]

        # print(augmented_prompt_list_flat[0])

        potential_candidates = edit_prompt_list_flat + augmented_prompt_list_flat

        # Randomly sampled #num successor candidates per parent prompt
        try:
            sample_candidate_prompts = random.sample(potential_candidates, self.config['num_candidates'])
        except:
            sample_candidate_prompts = potential_candidates
        return sample_candidate_prompts
    
    def get_used_data(self):
        return self.used_data