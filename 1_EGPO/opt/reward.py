import numpy as np
from opt.utils import detect_error, extract_bundle_score, extract_bundle_verdict


# Tried different reward functions when trying to optimise for a prompt. These a largely useless after we decided on the ensemble strategy
# These functions may optimise for prompts which induce Reward hacking behaviour, such as score compression, i.e.,changing score range to 2-4 from 1-5.


def rmse(given_score, true_score):
    res = np.sqrt((given_score - true_score)**2)
    
    return res

def true_rmse(score_list, true_list):
    # Ensure lists are NumPy arrays to perform vectorized operations
    score_array = np.array(score_list, dtype=object)
    true_array = np.array(true_list, dtype=object)

    # Create a boolean mask to find all non-None scores
    valid_indices = score_array != None

    # Apply the mask to both arrays to get only the valid scores
    valid_scores = score_array[valid_indices].astype(float)
    valid_trues = true_array[valid_indices].astype(float)

    # Check if there are any valid scores to prevent ZeroDivisionError
    if len(valid_scores) == 0:
        return 0.0

    return np.sqrt(np.mean((valid_scores - valid_trues)**2))


# def exponential_reward(collective_rmse, base_reward=1.0, k=0.5):
#     """
#     Calculates reward from a collective error using exponential decay.
#     'k' controls the steepness of the decay.
#     """
#     return base_reward * np.exp(-k * collective_rmse)

def inverse_reward(collective_rmse, base_reward=1.0, epsilon=0.1):
    """
    Calculates reward from a collective error using an inverse function.
    'epsilon' prevents division by zero.
    """
    return base_reward / (collective_rmse + epsilon)

# def logistic_separation_reward(rmse, base_reward=1.0, center=1.2, k=10):
#     """
#     Creates a steep reward drop-off around a 'center' value.
#     """
#     return base_reward / (1 + np.exp(k * (rmse - center)))

# def plateau_and_cliff_reward(error, base_reward=1.0, plateau_threshold=0.95, cutoff_threshold=1.5, center = 1.2, k=10):
#     """
#     A hybrid reward function with a flat plateau for excellent scores
#     and a logistic cliff for mediocre scores.
#     """
#     if error < plateau_threshold:
#         # Region 1: The high-reward plateau for excellent scores
#         return base_reward * (1 + (1 - error))
#     elif error < cutoff_threshold:
#         # Region 2: The logistic cliff for mediocre scores
#         # Center the cliff in the middle of the ramp for a smooth transition
#         return base_reward / (1 + np.exp(k * (error - center)))
#     else:
#         # Region 3: The floor for unacceptable scores
#         return 0


# def ramp(given_score, true_score, base_reward):
#     d = abs(given_score - true_score)
#     if d < 0.2:
#         return base_reward
#     elif d >= 0.2 and d < 0.6:
#         return base_reward * 0.5
#     elif d >= 0.6 and d < 1.1:
#         return base_reward * 0.1
#     else:
#         return 0



# def inverse_mae_reward(given_score, true_score, base_reward, epsilon=0.5):
#     """
#     Calculates a reward based on the inverse of the absolute error.
#     """
#     d = abs(given_score - true_score)
#     # Epsilon prevents division by zero
#     return base_reward / (d + epsilon)



class Reward():
    def __init__(self, config, request_model) -> None:
        self.config = config
        self.reward_func = config['reward_func']
        self.request = request_model

    async def calculate_reward(self, system_prompt, sample_data, metastructure=None, mode='single'): # need to change this to prompt_list and send multiple at once

        if mode == 'single':
            reward = 0


            prompt_list = [{"prompts": data['input'] + "\n" + self.config['json_addition']} for data in sample_data]

            # print("System prompt:\n", system_prompt + self.config['metrics'])

            # print()

            # print("Supposed to be sample data", prompt_list[0]["prompts"])


            # print("Sending in prompts for da reward\n")

            responses = await self.request.openai_request(prompt_list, system_prompt + self.config['metrics'])
            target_scores = [data['target_score'] for data in sample_data]

            consideration = self.config['case']

            if consideration == "":
                given_scores = [extract_bundle_score(i) for i in responses]
            else:
                verdicts = [extract_bundle_verdict(i, consideration) for i in responses]

            

            if consideration == "1-2":
                for i in range(len(responses)):

                    if target_scores[i] > 2 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":
                        reward += 1
                    elif target_scores[i] < 3 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":
                        reward += 1

            elif consideration == "4-5":
                for i in range(len(responses)):

                    if target_scores[i] > 3 and verdicts[i][1] == "yes" and verdicts[i][0] == "no":
                        reward += 1
                    elif target_scores[i] < 4 and verdicts[i][1] == "no" and verdicts[i][0] == "yes":
                        reward += 1

            elif consideration == "3":
                for i in range(len(responses)):

                    if target_scores[i] < 4 and verdicts[i][0] == "yes" and verdicts[i][1] == "no":
                        reward += 1
                    elif target_scores[i] > 3 and verdicts[i][0] == "no" and verdicts[i][1] == "yes":
                        reward += 1

            elif consideration == "":

                collective_rmse = true_rmse(given_scores, target_scores)

                reward = inverse_reward(collective_rmse, base_reward=self.config['base_reward'], epsilon=0.5)

            print(reward)

            return reward

