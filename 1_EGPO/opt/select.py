import random
import math

class Select():
    def __init__(self,
                train_data,
                config,
                reward_model):
        self.train_data = train_data
        self.config = config
        self.reward_model = reward_model
        self.used_data = []

    async def ucb(self, prompt_list):
        numbers_of_selections = [0] * len(prompt_list)
        sums_of_reward = [0] * len(prompt_list)
        index_list = [i for i in range(len(prompt_list))]

        for t in range(1, self.config['time_steps']+1):
            # print()
            print(f"Iteration {t} (Finding the top-b prompts):")


            train_data_values = list(self.train_data.values())

            sample_data = random.sample(train_data_values, self.config['sample_num'])


            self.used_data += sample_data
            if t == 1:
                select_prompt_index = random.choice(index_list)
            else:
                explore_param = self.config['explore_param']
                results = [q_value + explore_param*math.sqrt(math.log(t)/(n+1)) for q_value, n in zip(sums_of_reward, numbers_of_selections)]
                max_result = max(results)
                select_prompt_index = results.index(max_result)
            select_prompt = prompt_list[select_prompt_index]

            select_prompt_reward = await self.reward_model.calculate_reward(select_prompt, sample_data)

            # Update N and Q
            numbers_of_selections[select_prompt_index] += self.config['sample_num']
            sums_of_reward[select_prompt_index] += select_prompt_reward / numbers_of_selections[select_prompt_index]
        
        pairs = list(zip(sums_of_reward, prompt_list))
        pairs.sort(reverse=True)
        top_1_prompt = pairs[0][1]

        # Return top b prompts
        if self.config['beam_width'] > len(prompt_list):
            top_b_prompt = prompt_list
            # raise Exception("The value of beamwidth needs to be less than the length of the prompt list")

        else:
            top_b_prompt = [pair[1] for pair in pairs[:self.config['beam_width']]]
        
        return top_b_prompt, top_1_prompt

    async def struct_ucb(self, prompt, structure_list):
        numbers_of_selections = [0] * len(structure_list)
        sums_of_reward = [0] * len(structure_list)
        index_list = [i for i in range(len(structure_list))]

        for t in range(1, self.config['time_steps']+1):
            # print()
            print(f"Iteration {t} (Finding the top-b structures):")


            train_data_values = list(self.train_data.values())

            sample_data = random.sample(train_data_values, self.config['sample_num'])


            self.used_data += sample_data
            if t == 1:
                select_struct_index = random.choice(index_list)
            else:
                explore_param = self.config['explore_param']
                results = [q_value + explore_param*math.sqrt(math.log(t)/(n+1)) for q_value, n in zip(sums_of_reward, numbers_of_selections)]
                max_result = max(results)
                select_struct_index = results.index(max_result)
            select_struct = structure_list[select_struct_index]

            select_struct_reward = await self.reward_model.calculate_reward(prompt, sample_data, metastructure=select_struct, mode='structure')

            # Update N and Q
            numbers_of_selections[select_struct_index] += self.config['sample_num']
            sums_of_reward[select_struct_index] += select_struct_reward / numbers_of_selections[select_struct_index]
        
        pairs = list(zip(sums_of_reward, structure_list))
        pairs.sort(reverse=True)
        top_1_prompt = pairs[0][1]

        # Return top b prompts
        if self.config['beam_width'] > len(structure_list):
            top_b_prompt = structure_list
            # raise Exception("The value of beamwidth needs to be less than the length of the prompt list")

        else:
            top_b_prompt = [pair[1] for pair in pairs[:self.config['beam_width']]]
        
        return top_b_prompt, top_1_prompt
    
    async def run(self, prompt_list):
        top_b_prompt, top_1_prompt = await self.ucb(prompt_list)

        return top_b_prompt, top_1_prompt

    async def structure_run(self, prompt, structure_list):
        top_b_structure, top_1_structure = await self.struct_ucb(prompt, structure_list)
        
        return top_b_structure, top_1_structure
    
    def get_used_data(self):
        return self.used_data



    
        