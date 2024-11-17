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

        # Check if we have a model response
        if 'model_response' in record:
            response = record['model_response']
            
            if isinstance(model_instance, GeminiModel):
                # Process Gemini response
                if 'candidates' in response and response['candidates']:
                    candidate = response['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if parts:
                            part = parts[0]
                            if 'function_call' in part:
                                function_call = part['function_call']
                                processed_record['model_function_call'] = {
                                    'name': function_call['name'],
                                    'arguments': function_call['args']
                                }
                            elif 'text' in part:
                                processed_record['model_text'] = part['text']
            else:
                # Process OpenAI response
                # OpenAI responses are already in the correct format from models.py
                processed_record['model_function_call'] = response.get('model_function_call')
                processed_record['model_text'] = response.get('full_model_response')
                # if 'error' in response:
                #     processed_record['error'] = response.get('error')
        
        processed_results.append(processed_record)
    
    return {'test_results': processed_results} 