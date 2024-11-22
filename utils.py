import json
from models import GeminiModel

async def process_raw_responses(raw_results_file, model):
    """Process raw response data into a standardized format"""
    with open(raw_results_file, 'r') as f:
        raw_data = json.load(f)
    
    # Process the test results directly without mode wrapping
    return {
        'test_results': process_single_run(raw_data['test_results'])
    }

def process_single_run(results):
    """Process a single run of test results"""
    processed_records = []
    
    for record in results:
        processed_record = {
            'id': record['id'],
            'type': record['type'],
            'user_query': record['user_query'],
            'ground_truth': record['ground_truth'],
            'model_function_call': None,
            'model_text': None
        }
        
        # Extract response from model output
        if 'model_response' in record:
            response = record['model_response']
            # Handle Gemini response
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
            # Handle OpenAI response
            else:
                if "model_function_call" in response and response["model_function_call"]:
                    processed_record['model_function_call'] = response["model_function_call"]
                elif "full_model_response" in response and response["full_model_response"]:
                    processed_record['model_text'] = response["full_model_response"]
        
        processed_records.append(processed_record)
    
    return processed_records 