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

# Set up logging
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model, test_dataset, test_mode='function_call'):
        logger.info(f"Initializing ModelTester with mode: {test_mode}")
        self.model = model
        self.test_dataset = test_dataset
        self.test_mode = test_mode
        
        # For Gemini, we need to create a Tool instance with proper function declarations
        if isinstance(model, GeminiModel):
            logger.info("Creating Tool instance for Gemini model")
            self.tool = Tool(function_declarations=ALL_FUNCTIONS)
        else:
            self.tool = None

    async def process_test_case(self, index, record, use_tools):
        """Process a single test case"""
        test_case = index + 1
        user_query = record['user_query']
        logger.info(f"Processing test case {test_case}/{len(self.test_dataset)}")
        
        try:
            # Get the full prompt that will be sent to the model
            full_prompt = await self.model.get_full_prompt(user_query, use_tools=use_tools)
            
            if self.test_mode == 'function_call':
                response = await self.model.generate_response(
                    user_query, 
                    use_tools=use_tools,
                    tool=self.tool
                )
                
                result = {
                    'test_case': test_case,
                    'user_query': user_query,
                    'full_prompt': full_prompt,
                    'full_model_response': str(response),
                    'model_function_call': None,  # Will be updated if function call exists
                    'text': None  # Initialize text field
                }
                
                # Extract function call based on model type
                if isinstance(self.model, GeminiModel):
                    if (hasattr(response, 'candidates') and response.candidates and 
                        hasattr(response.candidates[0], 'function_calls') and 
                        response.candidates[0].function_calls):
                        
                        function_call = response.candidates[0].function_calls[0]
                        result['model_function_call'] = {
                            'name': function_call.name,
                            'arguments': function_call.args
                        }
                    else:
                        # Extract text from Gemini response when no function call is made
                        if hasattr(response, 'candidates') and response.candidates:
                            result['text'] = response.candidates[0].content.parts[0].text
                else:  # OpenAI model
                    choice = response['choices'][0]['message']
                    if 'function_call' in choice:
                        function_call = choice['function_call']
                        try:
                            arguments = json.loads(function_call['arguments'])
                        except json.JSONDecodeError:
                            logger.error("Failed to parse function arguments")
                            arguments = {}
                        result['model_function_call'] = {
                            'name': function_call['name'],
                            'arguments': arguments
                        }
                    else:
                        # Extract text from OpenAI response when no function call is made
                        result['text'] = choice.get('content', '')
                
                return index, result
                
            else:
                # Handle no_function mode
                response = await self.model.generate_response(
                    user_query, 
                    use_tools=use_tools,
                    tool=self.tool
                )
                return index, {
                    'test_case': test_case,
                    'user_query': user_query,
                    'full_prompt': full_prompt,
                    'model_response': response.text if hasattr(response, 'text') else str(response),
                    'full_model_response': str(response)
                }

        except Exception as e:
            logger.error(f"Error processing test case {test_case}: {str(e)}")
            return index, {
                'test_case': test_case,
                'user_query': user_query,
                'full_prompt': full_prompt if 'full_prompt' in locals() else None,
                'model_function_call': None,
                'full_model_response': None,
                'error': str(e)
            }

    async def run_tests(self, use_tools=False):
        logger.info(f"\nStarting test execution with {len(self.test_dataset)} test cases")
        logger.info(f"Mode: {self.test_mode}, Use tools: {use_tools}")

        # Create tasks for all test cases
        tasks = [
            self.process_test_case(index, record, use_tools)
            for index, record in enumerate(self.test_dataset)
        ]

        # Run all tasks in parallel
        responses = await asyncio.gather(*tasks)
        
        # Sort responses by index and extract just the model responses
        sorted_responses = sorted(responses, key=lambda x: x[0])
        model_responses = [r[1] for r in sorted_responses]

        logger.info("\nAll test cases processed")
        return {
            'test_dataset': self.test_dataset,
            'model_responses': model_responses,
            'test_mode': self.test_mode
        }
