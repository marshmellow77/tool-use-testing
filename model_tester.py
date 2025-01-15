import asyncio
import json
import logging
from tools.functions import ALL_FUNCTIONS
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    ToolConfig,
    Content,
    Part
)
from models import GeminiModel
from google.protobuf.json_format import MessageToDict
from tenacity import retry, wait_exponential, retry_if_exception_type, before_sleep_log

# Set up logging
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model, test_dataset):
        logger.info(f"Initializing ModelTester")
        self.model = model
        self.test_dataset = test_dataset
        
        # Initialize tools based on model type
        model_type = "gemini" if isinstance(model, GeminiModel) else "openai"
        logger.info(f"Creating tool configuration for {model_type} model")
        self.tools = ALL_FUNCTIONS.get_functions_for_model(model_type)
        logger.info(f"Initialized {len(self.tools) if isinstance(self.tools, list) else 'Gemini'} tools")

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),  # Exponential backoff between 4-60 seconds
        retry=retry_if_exception_type((Exception)),  # Retry on any exception
        before_sleep=before_sleep_log(logger, logging.WARNING)  # Log before retrying
    )
    async def process_test_case(self, index, record):
        """Process a single test case and return raw response"""
        test_case = index + 1
        user_query = record['user_query']
        record_type = record['type']
        expected_response_type = record['ground_truth']['expected_response_type']

        logger.info(f"Processing test case {record['id']} (type: {record_type})")
        
        try:
            response = await self.model.generate_response(
                user_query, 
                tool=self.tools
            )
            
            # Handle different model responses
            if isinstance(self.model, GeminiModel):
                try:
                    response_dict = {'model_response': MessageToDict(response._pb)}
                except (AttributeError, Exception) as e:
                    # Fallback: manually construct the response structure
                    response_dict = {
                        'model_response': {
                            'candidates': [],
                            'usage_metadata': None
                        }
                    }
                    
                    if hasattr(response, 'candidates'):
                        for candidate in response.candidates:
                            candidate_dict = {
                                'content': {
                                    'role': candidate.content.role,
                                    'parts': []
                                },
                                'finish_reason': candidate.finish_reason,
                                'avg_logprobs': candidate.avg_logprobs
                            }

                            if hasattr(candidate.content, 'parts'):
                                for part in candidate.content.parts:
                                    part_dict = {}
                                    # Handle text responses
                                    if hasattr(part, 'text') and part.text is not None:
                                        part_dict['text'] = part.text
                                    # Handle function calls
                                    if hasattr(part, 'function_call') and part.function_call is not None:
                                        try:
                                            part_dict['function_call'] = {
                                                'name': part.function_call.name,
                                                'args': part.function_call.args
                                            }
                                        except AttributeError:
                                            # If function call exists but is malformed, skip it
                                            pass
                                    # Only append if we captured either text or function call
                                    if part_dict:
                                        candidate_dict['content']['parts'].append(part_dict)

                            response_dict['model_response']['candidates'].append(candidate_dict)
                    
                    if hasattr(response, 'usage_metadata'):
                        response_dict['model_response']['usage_metadata'] = {
                            'prompt_token_count': response.usage_metadata.prompt_token_count,
                            'candidates_token_count': response.usage_metadata.candidates_token_count,
                            'total_token_count': response.usage_metadata.total_token_count
                        }
            else:  # OpenAI model
                response_dict = {}
                response_dict["model_response"] = response

            return index, response_dict

        except Exception as e:
            logger.error(f"Error processing test case {test_case}: {str(e)}")
            raise  # Re-raise the exception to trigger retry

    async def run_tests(self):
        logger.info(f"\nStarting test execution with {len(self.test_dataset)} test cases")

        # Create tasks for all test cases
        tasks = [
            self.process_test_case(index, record)
            for index, record in enumerate(self.test_dataset)
        ]

        # Run all tasks in parallel
        responses = await asyncio.gather(*tasks)
        
        # Sort responses by index and extract just the model responses
        sorted_responses = sorted(responses, key=lambda x: x[0])
        model_responses = [r[1] for r in sorted_responses]

        logger.info("\nAll test cases processed")
        
        # Combine model responses with the test dataset
        combined_results = [
            {**record, **response}  # Combine dataset record with model response
            for record, response in zip(self.test_dataset, model_responses)
        ]
        return combined_results