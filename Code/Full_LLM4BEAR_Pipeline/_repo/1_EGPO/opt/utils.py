import random
import time
import openai
import re
import json

def extract_reasoning_part(response_text: str) -> str:
    """
    Extracts the reasoning text that appears before a "===JSON_START===" separator.

    Args:
        response_text: The full string from the LLM.

    Returns:
        The text before the separator, with leading/trailing whitespace removed.
        Returns the full string if the separator is not found.
    """
    # .split(...)[0] is always safe and does not need a try/except block.
    reasoning_part = response_text.split("===JSON_START===")[0]
    return reasoning_part.strip()



def extract_json_simple_replace(response_text):
    """
    Extracts a JSON object from a string that has a "===JSON_START===" separator.

    This function isolates the JSON by finding the first '{' and last '}'
    to ensure it works correctly even with markdown fences or extra whitespace.

    Args:
        response_text (str): The full string containing the separator and JSON.

    Returns:
        dict: The parsed JSON object as a Python dictionary, or None if an error occurs.
    """
    try:
        # 1. Get the text after the separator
        json_part = response_text.split("===JSON_START===")[1]

        # 2. Find the boundaries of the JSON object
        first_brace = json_part.find('{')
        last_brace = json_part.rfind('}')

        # 3. Slice the string to get only the valid JSON
        # This will fail gracefully in the json.loads() if a brace isn't found
        json_string = json_part[first_brace : last_brace + 1]

        # 4. Parse the clean string
        parsed_json = json.loads(json_string)
        # print("I'M GOING")
        return parsed_json

    except IndexError:
        print("Error: The separator '===JSON_START===' was not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Could not find or parse a valid JSON object after the separator.")
        return None


def extract_bundle_score(response):
    json_schema = extract_json_simple_replace(response)
    if json_schema is not None:
        try:
            given_score = float(json_schema['score'])
        except:
            given_score = None
    else:
        given_score = None
    
    return given_score

def extract_bundle_verdict(response, consideration):
    json_schema = extract_json_simple_replace(response)
    if json_schema is not None:
        try:
            if consideration == "1-2":
                given_verdict_1 = json_schema['is_poor_quality_bundle']
                given_verdict_2 = json_schema['is_acceptable_quality_bundle']
                given_verdict = [given_verdict_1.lower(), given_verdict_2.lower()]
            elif consideration == "3":
                given_verdict_1 = json_schema['needs_improvement_bundle']
                given_verdict_2 = json_schema['is_good_quality_bundle']
                given_verdict = [given_verdict_1.lower(), given_verdict_2.lower()]
            elif consideration == "4-5":
                given_verdict_1 = json_schema['needs_improvement_bundle']
                given_verdict_2 = json_schema['is_high_quality_bundle']
                given_verdict = [given_verdict_1.lower(), given_verdict_2.lower()]
            else:
                given_verdict = None
        except:
            given_verdict = None
    else:
        given_verdict = None

    return given_verdict


def detect_error(bundle_score, target_score, mode='improve', threshold=0.55):
    
    if bundle_score is not None:
        if mode == 'improve':
            if abs(bundle_score - target_score) > threshold:
                return False
            elif abs(bundle_score - target_score) <= threshold:
                return True
        elif mode == 'select':
            return True
    else:
        return False



def extract_edit_prompt(response):
    pattern = r'<START>\s*(.*?)\s*<END>'
    result_list = re.findall(pattern, response, re.DOTALL)
    if len(result_list) == 0:
        pattern = r'<START>(.*?)<END>'
        result_list = re.findall(pattern, response, re.DOTALL)
    return result_list 

def load_eval_data(config):
    with open(f"{config['data_path']}{config['dataset']}/ID/test_seed_{config['seed']}.json", 'r') as json_file:
        test_data = json.load(json_file)
    return test_data


