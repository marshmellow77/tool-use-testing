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
        
        # Initialize tools based on model type
        model_type = "gemini" if isinstance(model, GeminiModel) else "openai"
        logger.info(f"Creating tool configuration for {model_type} model")
        self.tools = ALL_FUNCTIONS.get_functions_for_model(model_type)
        logger.info(f"Initialized {len(self.tools) if isinstance(self.tools, list) else 'Gemini'} tools")

    async def process_test_case(self, index, record, use_tools):
        """Process a single test case"""
        test_case = index + 1
        user_query = record['user_query']
        
        logger.info(f"\nProcessing test case {test_case}/{len(self.test_dataset)}")
        logger.info(f"User query: {user_query}")
        
        try:
            # Get the expected function from the assistant_response
            expected_function = None
            if 'assistant_response' in record and 'function_call' in record['assistant_response']:
                expected_function = record['assistant_response']['function_call']['name']
            
            logger.info(f"Expected function: {expected_function}")
            logger.info(f"Use tools: {use_tools}")
            
            current_tool = None
            if use_tools:
                if self.test_mode == 'no_function':
                    # Always provide tools in no_function mode if use_tools is True
                    if isinstance(self.model, GeminiModel):
                        current_tool = self.tools
                    else:  # OpenAI
                        current_tool = self.tools  # Pass all tools for OpenAI
                elif expected_function:
                    # In function_call mode, only pass the expected tool
                    if isinstance(self.model, GeminiModel):
                        current_tool = self.tools
                    else:  # OpenAI
                        current_tool = next(
                            (tool for tool in self.tools 
                             if tool["name"] == expected_function),
                            None
                        )
            
            response = await self.model.generate_response(
                user_query, 
                use_tools=use_tools,
                tool=current_tool
            )
            
            result = {
                'test_case': test_case,
                'user_query': user_query,
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
                if 'model_function_call' in response and response['model_function_call']:
                    function_call = response['model_function_call']
                    try:
                        # Check if arguments is already a dict
                        if isinstance(function_call['arguments'], dict):
                            arguments = function_call['arguments']
                        else:
                            arguments = json.loads(function_call['arguments'])
                    except (json.JSONDecodeError, TypeError):
                        logger.error("Failed to parse function arguments")
                        arguments = {}
                    result['model_function_call'] = {
                        'name': function_call['name'],
                        'arguments': arguments
                    }
                else:
                    # Extract text from OpenAI response when no function call is made
                    result['text'] = response['full_model_response']
            
            return index, result
            
        except Exception as e:
            # logger.error(f"Error processing test case {test_case}: {str(e)}")
            return index, {
                'test_case': test_case,
                'user_query': user_query,
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
