import re
import ast
import json
import html # Added for entity decoding
from collections import defaultdict

# Helper function for checking strict validity (already defined in your code flow)
def is_strictly_valid(d: dict) -> bool:
    return all(isinstance(v, str) and v.strip() for v in d.values())

def recover_list_intents(d: dict) -> dict:
    recovered = {}
    for k, v in d.items():
        # Check for the common error: single-item list containing a string
        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], str):
            recovered[k] = v[0].strip() # Unwrap the string from the list
        else:
            recovered[k] = v # Keep the original value (it's either correct or unrecoverable)
    return recovered

def clean_input_string(s):
    """Decodes HTML entities and removes problematic characters before parsing."""
    if not isinstance(s, str):
        return s
        
    # 1. Decode HTML entities (e.g., &amp; to &, &quot; to ", &#153; to ™)
    s = html.unescape(s)
    
    # 2. Remove known parsing breakers (important for ast.literal_eval)
    s = s.replace('®', '').replace('™', '').replace('©', '').replace('°', '')
    
    # 3. Standardize whitespace
    s = s.replace('\n', '').strip()

    return s


def rebuild_malformed_duplicate_dict(malformed_str):
    """
    Aggressively rebuilds a string that contains illegal duplicate dictionary keys
    (e.g., {'k': 'v1', 'k': 'v2', ...}) into a legal dictionary of lists:
    {'k': ['v1', 'v2'], ...}. This handles the LLM's common duplicate key error.
    """
    final_dict = defaultdict(list)
    
    # Remove all comments and extract the content between the main braces { ... }
    cleaned_str = re.sub(r'//.*', '', malformed_str)
    match = re.search(r'\{(.*?)\}', cleaned_str, re.DOTALL)
    if not match:
        raise ValueError("No dictionary content found for rebuilding.")
    
    content = match.group(1).strip()
    
    # Split content into individual 'key': 'value' entries by comma
    raw_entries = content.split(',')
    
    for entry in raw_entries:
        entry = entry.strip()
        if not entry:
            continue
            
        # Use regex to split only on the first colon, separating the key from the value
        parts = re.split(r'\s*:\s*', entry, maxsplit=1)
        
        if len(parts) == 2:
            key_raw, value_raw = parts
            
            # Clean and normalize key (remove quotes and whitespace)
            key = key_raw.strip().strip("'\"")
            
            # Clean and normalize value (remove quotes and whitespace)
            value = value_raw.strip().strip("'\"")

            if key and value:
                # Add the value to the list associated with the key
                final_dict[key].append(value)
        else:
            raise ValueError(f"Malformed entry during rebuild: {entry}")

    if not final_dict:
        raise ValueError("Reassembly yielded an empty dictionary.")
        
    return dict(final_dict)

def output_parser(response_str, type='bundle'):
    # Initialize response variables
    response_dict = {}
    response_str = clean_input_string(response_str)

    # 1. ATTEMPT TO FIND DICTIONARY STRUCTURE
    match_str = re.search(r'{.*}', response_str, re.DOTALL)
    if match_str:
        response_str = match_str.group()
    else:
        return {'state_code': 404, 'output': {}} 

    # Further cleanup of complex nesting/joins
    if "}{" in response_str:
        response_str = response_str.replace("}{", ", ")

    # 2. ATTEMPT ROBUST PARSING (With Duplicate Key Recovery)
    
    # Try JSON first (Strict)
    try:
        response_dict = json.loads(response_str)
    except json.JSONDecodeError:
        # Fallback 1: Try ast.literal_eval (Handles Python dict syntax like single quotes)
        try:
            response_dict = ast.literal_eval(response_str)
        except (SyntaxError, ValueError, IndexError, TypeError):
            # Fallback 2: Try aggressive rebuild for duplicate key errors
            try:
                response_dict = rebuild_malformed_duplicate_dict(response_str)
            except ValueError:
                # Parsing failed entirely even with aggressive recovery
                return {'state_code': 404, 'output': {}}

    # At this point, response_dict is a valid Python dictionary object.
    
    # 3. TYPE-SPECIFIC VALIDATION AND RECOVERY

    if type == 'bundle':
        recovered_dict = {}
        
        for key, value in response_dict.items():
            # Bundle-specific recovery: Handle value as string (e.g., 'p1, p2, p3')
            if isinstance(value, str):
                # Split the string by comma, strip whitespace, and filter out empty strings
                items = [
                    item.strip() 
                    for item in value.split(',') 
                    if item.strip()
                ]
                recovered_dict[key] = items
                
            elif isinstance(value, list):
                # Handle the standard list output, ensuring elements are clean strings
                recovered_dict[key] = [str(item).strip() for item in value]
            else:
                # Value is an unexpected type (int, None, bool)
                return {'state_code': 404, 'output': {}}

        # Final check: only keep valid bundles (non-empty lists)
        final_bundle_dict = {
            k: v for k, v in recovered_dict.items() if isinstance(v, list) and len(v) > 0
        }
        
        return {'state_code': 200, 'output': final_bundle_dict}

    elif type == 'intent':
        # Intent-specific validation: Values MUST be non-empty strings (intents)
        if all(isinstance(v, str) and v.strip() for v in response_dict.values()):
            return {'state_code': 200, 'output': response_dict}
        else:
            recovered_dict = recover_list_intents(response_dict)
            
            # 3. Check the recovered dictionary against the strict criteria
            if all(isinstance(v, str) and v.strip() for v in recovered_dict.values()):
                # Recovery was successful for all items! Return the clean dictionary.
                return {'state_code': 200, 'output': recovered_dict}


            # 4. Final Failure: Log and return 404
            else:
                # --- START: Debugging Block (Logs the final failure reason) ---
                malformed_intents = {}
                for key, value in recovered_dict.items():
                    if not (isinstance(value, str) and value.strip()):
                        malformed_intents[key] = value
                
                print("❌ Malformed Intent Data Detected (Recovery Failed):")
                print(f"LLM Response: {response_dict}")

                print(f"Failed Keys/Values: {malformed_intents}")
                # --- END: Debugging Block ---
                
                # Parsing failed entirely because some intent values were fundamentally malformed 
                # (e.g., [1, 2, 3] or None).
                return {'state_code': 404, 'output': {}}

   
    elif type == 'score':
          
        if not response_str.startswith('{'):
            match_str = re.search(r'{.*}', response_str, re.DOTALL)
            match_str = match_str.group().replace('\n', '')
            response_str = match_str
        if "}{" in response_str:
            response_str = response_str.replace("}{", ", ")
        response_dict = ast.literal_eval(response_str)
        return {'state_code': 200, 'output': response_dict}
                
    # Fallback for unexpected type argument
    return {'state_code': 404, 'output': {}}
    # return {'state_code': 200, 'output': response_dict}

    
def process_results(bundle_res):
    invalid_id = []
    for testid, bundles in bundle_res.items():
        c = 0
        for b,items in bundles.items():
            if len(items)==1:
                c+=1
        # print(c, len(bundles))
        if c==len(bundles):
            # print(test_id)
            invalid_id.append(testid)
    print(invalid_id)

    remove_invalid_res = {}
    for test_id, bundles in bundle_res.items():
        if test_id in invalid_id:
            continue
        format_bundles = {}
        for bid, items in bundles.items():
            if len(items)>1:
                format_bundles[bid] = items
        remove_invalid_res[test_id] = format_bundles

    return remove_invalid_res