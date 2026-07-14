from tqdm import tqdm
from opt.metrics import Metric
from opt.request import Request
from opt.utils import extract_bundle_score
import asyncio
# from opt.asyncrequest import AsyncRequest

class Eval():
    def __init__(self, config, data, text_table):
        self.conf = config
        self.request = Request(config) # change to async request
        self.data = data
        self.text_table = text_table
        self.error_list = []
        self.target_score_list = []
        self.given_score_list = []
    
    def run(self, prompt):
        self.normal_eval(prompt)
        metric = Metric(self.given_score_list, self.target_score_list, self.conf)
        result = metric.rmse()

        return result, self.target_score_list, self.error_list
    
    def record_error(self, data, response):
        tmp = {}
        tmp['response'] = response
        tmp['input'] = data['input']
        tmp['target_score'] = data['target_score']
        
        return tmp

    async def normal_eval(self, prompt):
        threshold = 1.1

        # Step 1: Run all initial evaluations concurrently
        eval_prompts = [{"prompts": d['input'] + "\n" + self.conf['json_addition']} for d in self.data]
        responses = await self.request.openai_request(prompts=eval_prompts, system=prompt)
        result_list = [extract_bundle_score(res) for res in responses]

        # Step 2: Identify and retry only the failures concurrently
        none_indices = [i for i, score in enumerate(result_list) if score is None]
        if none_indices:
            # Create a list of tasks for retries
            retry_tasks = [
                self.request.openai_request(prompts=[eval_prompts[i]], system=prompt) 
                for i in none_indices
            ]
            # Gather all retry responses
            retry_responses_lists = await asyncio.gather(*retry_tasks)
            
            # Flatten the list of lists
            retry_responses = [res[0] for res_list in retry_responses_lists for res in res_list]
            
            # Update the original response and result lists with successful retries
            for i, response in zip(none_indices, retry_responses):
                retried_score = extract_bundle_score(response)
                if retried_score is not None:
                    result_list[i] = retried_score
                    responses[i] = response

        # Step 3: Log and collect errors from the final results
        for i in range(len(responses)):
            self.text_table.add_data(self.data[i]['input'], self.data[i]['target_score'], responses[i])
            self.given_score_list.append(result_list[i])

            if (result_list[i] is None) or (abs(result_list[i] - self.data[i]['target_score']) > threshold):
                error = self.record_error(self.data[i], responses[i])
                self.error_list.append(error)
                self.target_score_list.append(self.data[i]['target_score'])
