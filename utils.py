import json
from models import GeminiModel

async def process_raw_responses(raw_results_file, model):
    """Process raw response data into a standardized format"""
    with open(raw_results_file, 'r') as f:
        raw_data = json.load(f)
    
    processed_results = {}
    
    # Check if we have multiple modes
    if any(mode in raw_data for mode in ['no_tools', 'with_tools']):
        for mode in ['no_tools', 'with_tools']:
            if mode in raw_data:
                processed_results[mode] = {
                    'test_results': process_single_run(raw_data[mode]['test_results'])
                }
    else:
        # Get the single mode that exists
        mode = next(iter(raw_data))  # Gets the first (and only) key
        processed_results = {
            mode: {
                'test_results': process_single_run(raw_data[mode]['test_results'])
            }
        }
    
    return processed_results

def process_single_run(results):
    """Process a single run of test results"""
    processed_records = []
    
    for record in results:  # Iterate through the test_results array
        processed_record = {
            'id': record['id'],
            'user_query': record['user_query'],
            'ground_truth': record['ground_truth'],
            'model_function_call': None,
            'model_text': None
        }
        
        # Extract response from model output
        if 'model_response' in record:
            response = record['model_response']
            if 'candidates' in response and response['candidates']:
                candidate = response['candidates'][0]
                if 'content' in candidate and 'parts' in candidate['content']:
                    parts = candidate['content']['parts']
                    if parts:
                        part = parts[0]
                        if 'function_call' in part:
                            processed_record['model_function_call'] = {
                                'name': part['function_call']['name'],
                                'arguments': part['function_call']['args']
                            }
                        elif 'text' in part:
                            processed_record['model_text'] = part['text']
        
        processed_records.append(processed_record)
    
    return processed_records 