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
from utils import process_raw_responses
# import weave

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
    parser.add_argument('--skip-evaluation', 
                        action='store_true',
                        help='Skip evaluation after running tests')
    args = parser.parse_args()

    # Initialize Weave with metadata about the test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # weave.init(f"tool-selection-testing-{timestamp}")
    
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

    # Create results directory
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"test_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Step 1: Run tests and save raw results
    raw_results = {}
    if args.mode == 'no_function' and args.run_both_tool_modes:
        logger.info("Running tests in both tool modes...")
        # Run without tools
        logger.info("Running tests without tools...")
        raw_results['no_tools'] = await tester.run_tests(use_tools=False)
        # Run with tools
        logger.info("Running tests with tools...")
        raw_results['with_tools'] = await tester.run_tests(use_tools=True)
    else:
        # Regular single run
        # mode = 'with_tools' if args.mode == 'function_call' or args.run_both_tool_modes else 'no_tools'
        raw_results['with_tools'] = await tester.run_tests(use_tools=True)

    raw_results_file = os.path.join(results_dir, "raw_responses.json")
    with open(raw_results_file, 'w') as f:
        json.dump(raw_results, f, indent=2)

    # Step 2: Process raw results into standardized format
    processed_results = await process_raw_responses(raw_results_file, model)
    processed_results_file = os.path.join(results_dir, "processed_responses.json")
    with open(processed_results_file, 'w') as f:
        json.dump(processed_results, f, indent=2)

    # Save test parameters
    test_parameters = {
        "timestamp": timestamp,
        "test_mode": args.mode,
        "model_type": args.model_type,
        "dataset_path": args.dataset,
        "model_id": args.gemini_model_id if args.model_type == 'gemini' else args.openai_model_name,
        "semantic_judge_model": args.semantic_judge_model if not args.skip_evaluation else None,
        "generation_config": {
            "temperature": 0.0
        }
    }
    
    parameters_file = os.path.join(results_dir, "test_parameters.json")
    with open(parameters_file, 'w') as f:
        json.dump(test_parameters, f, indent=2)

    # Step 3: Evaluate results
    if not args.skip_evaluation:
        logger.info("Starting evaluation...")
        evaluator = Evaluator(
            test_mode=args.mode,
            semantic_judge_model_name=args.semantic_judge_model,
            semantic_judge_prompt=args.semantic_judge_prompt,
            run_both_tool_modes=args.run_both_tool_modes
        )
        
        await evaluator.evaluate_results(processed_results_file)
        evaluator.save_results(results_dir)
    else:
        logger.info("Skipping evaluation as per user request.")
    
    logger.info("Results saved to: %s", results_dir)

if __name__ == "__main__":
    asyncio.run(main())
