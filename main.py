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

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
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
    raw_results = await tester.run_tests(use_tools=(args.mode == 'function_call'))
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"test_run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save raw results
    raw_results_file = os.path.join(results_dir, "raw_responses.json")
    with open(raw_results_file, 'w') as f:
        json.dump(raw_results, f, indent=2)

    # Step 2: Evaluate results
    evaluator = Evaluator(
        test_mode=args.mode,
        semantic_judge_model_name=args.semantic_judge_model,
        semantic_judge_prompt=args.semantic_judge_prompt
    )
    evaluation_results = await evaluator.evaluate_results(raw_results)
    
    # Save evaluation results
    evaluator.save_results(results_dir)

    # Log summary
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
