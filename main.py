import os
import json
import asyncio
import argparse
import json
import os
from models import OpenAIModel, GeminiModel
from evaluator import Evaluator
from model_tester import ModelTester
from tools.functions import ALL_FUNCTIONS
from datetime import datetime
import csv
import logging
from vertexai.generative_models import Tool
from vertexai.generative_models import FunctionDeclaration
from vertexai.generative_models import ToolConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    ToolConfig,
    GenerationConfig
)

# Suppress urllib3 connection pool warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.MaxRetryError)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)
logging.getLogger('google.auth.transport.requests').setLevel(logging.ERROR)
logging.getLogger('google.oauth2').setLevel(logging.ERROR)

# Suppress Vertex AI engine message
logging.getLogger('google.ai.generativelanguage.generative_models._async_engine').setLevel(logging.WARNING)

# Add these lines to suppress OpenAI client logs
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        return json.load(f)

async def main():
    parser = argparse.ArgumentParser(description='Run model tests')
    parser.add_argument('--model-type', choices=['gemini', 'openai'], required=True, help='Type of model to test')
    parser.add_argument('--mode', choices=['function_call', 'no_function'], 
                       default='function_call', help='Testing mode')
    parser.add_argument('--dataset', help='Path to test dataset', required=True)
    parser.add_argument('--openai-model-name', default='gpt-4o-mini', help='OpenAI model name')
    parser.add_argument('--gemini-model-id', default='gemini-1.5-flash-002', help='Gemini model ID')
    parser.add_argument('--openai-api-key', help='OpenAI API key')
    parser.add_argument('--semantic-judge-model', default='gpt-4', help='Model name for semantic judge')
    parser.add_argument('--semantic-judge-prompt', help='Path to semantic judge prompt file')
    parser.add_argument('--run-both-tool-modes', 
                       action='store_true',
                       help='In no_function mode, run both with and without tools')
    args = parser.parse_args()

    logger.info(f"\nStarting test run with {args.model_type} model in {args.mode} mode")
    logger.info(f"Loading dataset from: {args.dataset}")
    test_dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(test_dataset)} test cases")

    if args.model_type == 'openai':
        if not args.openai_api_key:
            logger.error("Error: OpenAI API key is required for OpenAI model.")
            return
        logger.info(f"Initializing OpenAI model: {args.openai_model_name}")
        model = OpenAIModel(
            model_name=args.openai_model_name,
            api_key=args.openai_api_key,
            temperature=0
        )
    elif args.model_type == 'gemini':
        logger.info(f"Initializing Gemini model: {args.gemini_model_id}")
        model = GeminiModel(
            model_id=args.gemini_model_id,
            temperature=0
        )
    else:
        logger.error("Invalid model type.")
        return

    logger.info("Starting test execution...")
    tester = ModelTester(
        model=model,
        test_dataset=test_dataset,
        test_mode=args.mode
    )

    # Step 1: Run tests and get raw results
    if args.mode == 'no_function' and args.run_both_tool_modes:
        # Run tests without tools
        logger.info("\n=== Running tests WITHOUT tools ===")
        no_tools_results = await tester.run_tests(use_tools=False)
        
        # Run tests with tools
        logger.info("\n=== Running tests WITH tools ===")
        with_tools_results = await tester.run_tests(use_tools=True)
        
        # Combine results
        raw_results = {
            'no_tools': {
                'test_dataset': test_dataset,
                'model_responses': no_tools_results['model_responses'],
                'test_mode': args.mode,
                'run_type': 'no_tools'
            },
            'with_tools': {
                'test_dataset': test_dataset,
                'model_responses': with_tools_results['model_responses'],
                'test_mode': args.mode,
                'run_type': 'with_tools'
            }
        }
    else:
        # If in no_function mode, use tools by default, otherwise follow function_call mode setting
        use_tools = True if args.mode == 'no_function' else (args.mode == 'function_call')
        raw_results = await tester.run_tests(use_tools=use_tools)

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"test_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save test parameters
    test_parameters = {
        "timestamp": timestamp,
        "test_mode": args.mode,
        "dataset_path": args.dataset,
        "function_call_model_id": args.gemini_model_id if args.model_type == 'gemini' else args.openai_model_name,
        "semantic_judge_model_id": args.semantic_judge_model,
        "generation_config": {
            "temperature": 0.0
        }
    }
    
    parameters_file = os.path.join(results_dir, "test_parameters.json")
    with open(parameters_file, 'w') as f:
        json.dump(test_parameters, f, indent=2)

    # Save raw results
    raw_results_file = os.path.join(results_dir, "raw_responses.json")
    with open(raw_results_file, 'w') as f:
        json.dump(raw_results, f, indent=2)

    # Step 2: Evaluate results
    evaluator = Evaluator(
        test_mode=args.mode,
        semantic_judge_model_name=args.semantic_judge_model,
        semantic_judge_prompt=args.semantic_judge_prompt,
        run_both_tool_modes=args.run_both_tool_modes
    )
    evaluation_results = await evaluator.evaluate_results(raw_results)
    
    # Save evaluation results
    evaluator.save_results(results_dir)

    # Log summary
    if args.mode == 'no_function' and args.run_both_tool_modes:
        for run_type, results in evaluation_results.items():
            logger.info(f"""
Evaluation Summary ({run_type}):
Total test cases: {results['total_tests']}
Correct predictions: {results['correct_predictions']}
Incorrect predictions: {results['incorrect_predictions']}
Accuracy: {results['accuracy']:.2f}%
""")
    else:
        logger.info(f"""
Evaluation Summary:
Total test cases: {evaluation_results['total_tests']}
Correct predictions: {evaluation_results['correct_predictions']}
Incorrect predictions: {evaluation_results['incorrect_predictions']}
Accuracy: {evaluation_results['accuracy']:.2f}%
""")
    
    logger.info("Results saved to: %s", results_dir)

if __name__ == "__main__":
    asyncio.run(main())
