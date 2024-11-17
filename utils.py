import json
from models import GeminiModel

async def process_raw_responses(raw_responses_file, model_instance):
    """Process raw responses into standardized format"""
    with open(raw_responses_file, 'r') as f:
        raw_data = json.load(f)

    processed_results = []
    
    for record in raw_data['test_results']:
        processed_record = {
            'id': record.get('id'),
            'user_query': record['user_query'],
            'ground_truth': record.get('ground_truth'),
            'model_function_call': None,
            'model_text': None
        }

        if 'model_response' in record and record['model_response']:
            if isinstance(model_instance, GeminiModel):
                response = record['model_response']
                if 'candidates' in response and response['candidates']:
                    candidate = response['candidates'][0]  # Get first candidate
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if parts:  # If there are parts
                            part = parts[0]  # Get first part
                            if 'function_call' in part:
                                function_call = part['function_call']
                                processed_record['model_function_call'] = {
                                    'name': function_call['name'],
                                    'arguments': function_call['args']
                                }
                            elif 'text' in part:
                                processed_record['model_text'] = part['text']
            
            else:  # OpenAI model
                response = record['model_response']
                if 'model_function_call' in response and response['model_function_call']:
                    function_call = response['model_function_call']
                    try:
                        arguments = (function_call['arguments'] 
                                   if isinstance(function_call['arguments'], dict)
                                   else json.loads(function_call['arguments']))
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                    
                    processed_record['model_function_call'] = {
                        'name': function_call['name'],
                        'arguments': arguments
                    }
                else:
                    processed_record['model_text'] = response['model_response']
        
        processed_results.append(processed_record)
    
    return {'test_results': processed_results} 