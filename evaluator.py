import json
import csv
import os
import logging
from datetime import datetime
from vertexai.generative_models import GenerationConfig
import asyncio
from typing import List, Dict, Any
# import weave
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, semantic_judge_model_name=None):
        """Initialize the evaluator"""
        self.total_tests = 0
        self.correct_predictions = 0
        self.incorrect_predictions = 0
        self.detailed_results = []
        self.semantic_comparisons = []
        
        # Initialize semantic evaluation if model name is provided
        self.model = None
        self.prompt_templates = {}
        if semantic_judge_model_name:
            from vertexai.generative_models import GenerativeModel
            self.model = GenerativeModel(semantic_judge_model_name)
            
            # Load all prompt templates
            prompt_types = ['tool_selection', 'text_response', 'not_supported', 'error', 'clarifying']
            for type_name in prompt_types:
                prompt_path = f"prompts/semantic_judge_{type_name}.txt"
                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r') as f:
                        self.prompt_templates[type_name] = f.read()
                else:
                    logger.warning(f"Prompt template not found: {prompt_path}")

    def _are_values_equivalent(self, expected_val, model_val):
        """Compare two values, handling numeric equivalence"""
        # Try numeric comparison first
        try:
            # Convert both values to float
            expected_num = float(str(expected_val).strip())
            model_num = float(str(model_val).strip())
            return abs(expected_num - model_num) < 1e-10  # Using small epsilon for float comparison
        except (ValueError, TypeError):
            # If conversion fails, fall back to string comparison
            return str(expected_val).lower().strip() == str(model_val).lower().strip()

    def _are_function_calls_identical(self, ground_truth, model_call):
        """Check if two function calls are identical"""
        if not model_call or not ground_truth:
            return False
            
        if ground_truth['name'].lower() != model_call['name'].lower():
            return False
            
        expected_args = ground_truth['arguments']
        model_args = model_call['arguments']
        
        # Check if they have the same parameters
        if set(expected_args.keys()) != set(model_args.keys()):
            return False
            
        # Check if parameter values match
        for key in expected_args:
            if not self._are_values_equivalent(expected_args[key], model_args[key]):
                return False
                
        return True

    def _get_function_call_differences(self, ground_truth, model_call):
        """Get detailed differences between function calls"""
        differences = {
            'name_mismatch': False,
            'param_differences': [],
            'param_values': {},
            'needs_semantic_check': False
        }
        
        if not model_call:
            differences['name_mismatch'] = True
            return differences
            
        # Check function name
        if ground_truth['name'].lower() != model_call['name'].lower():
            differences['name_mismatch'] = True
            return differences
            
        expected_args = ground_truth['arguments']
        model_args = model_call['arguments']
        
        # Check parameters
        for key in expected_args:
            if key not in model_args:
                differences['param_differences'].append(f"Missing parameter: {key}")
                differences['needs_semantic_check'] = True
            elif not self._are_values_equivalent(expected_args[key], model_args[key]):
                differences['param_differences'].append(
                    f"{key}: expected '{expected_args[key]}', got '{model_args[key]}'"
                )
                # Only add to param_values if it's not a numeric mismatch
                try:
                    float(str(expected_args[key]))
                    float(str(model_args[key]))
                except (ValueError, TypeError):
                    differences['param_values'][key] = (expected_args[key], model_args[key])
                    differences['needs_semantic_check'] = True
                
        for key in model_args:
            if key not in expected_args:
                differences['param_differences'].append(f"Unexpected parameter: {key}")
                differences['needs_semantic_check'] = True
                
        return differences

    async def evaluate_results(self, results_file_path):
        """Evaluate results from the processed responses file"""
        logger.info("Starting evaluation of processed results...")
        
        with open(results_file_path, 'r') as f:
            results = json.load(f)

        evaluation_results = await self._evaluate_single_run(results['test_results'])
        self.detailed_results = evaluation_results['detailed_results']
        self.semantic_comparisons = evaluation_results.get('semantic_comparisons', [])
        self.total_tests = evaluation_results['total_tests']
        self.correct_predictions = evaluation_results['correct_predictions']
        self.incorrect_predictions = evaluation_results['incorrect_predictions']
        
        accuracy = (self.correct_predictions / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        logger.info("Evaluation completed.")
        
        return {
            'total_tests': self.total_tests,
            'correct_predictions': self.correct_predictions,
            'incorrect_predictions': self.incorrect_predictions,
            'accuracy': accuracy,
            'detailed_results': self.detailed_results,
            'semantic_comparisons': self.semantic_comparisons
        }

    async def _evaluate_single_run(self, test_results):
        """Evaluate a single run of test results"""
        evaluation_tasks = []
        total_tests = len(test_results)
        correct_predictions = 0
        incorrect_predictions = 0
        detailed_results = []
        semantic_comparisons = []
        
        for test_case in test_results:
            task = self._evaluate_test_case(test_case)
            evaluation_tasks.append(task)
        
        evaluated_results = await asyncio.gather(*evaluation_tasks)
        
        for result in evaluated_results:
            if result['is_correct']:
                correct_predictions += 1
            else:
                incorrect_predictions += 1
            detailed_results.append(result['detailed_result'])
            if 'semantic_comparisons' in result:
                semantic_comparisons.extend(result['semantic_comparisons'])
        
        return {
            'total_tests': total_tests,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': incorrect_predictions,
            'detailed_results': detailed_results,
            'semantic_comparisons': semantic_comparisons
        }

    async def _evaluate_test_case(self, test_case):
        """Evaluate a single test case"""
        test_id = test_case['id']
        logger.info(f"Evaluating test case {test_id}")
        user_query = test_case['user_query']
        ground_truth = test_case['ground_truth']
        model_function_call = test_case['model_function_call']
        model_text = test_case['model_text']
        record_type = test_case['type']
        expected_response_type = ground_truth['expected_response_type']
        
        if expected_response_type == 'function_call':
            # Function call evaluation
            are_identical = self._are_function_calls_identical(ground_truth['function_call'], model_function_call)
            
            result = {
                'test_case': test_id,
                'type': record_type,
                'user_query': user_query,
                'expected_function_call': ground_truth['function_call'],
                'model_function_call': model_function_call,
                'result': 'Correct' if are_identical else 'Incorrect',
                'model_response': model_text
            }
            
            if not are_identical:
                differences = self._get_function_call_differences(ground_truth['function_call'], model_function_call)
                if differences['name_mismatch']:
                    result['mismatch_type'] = 'Function'
                    result['reason'] = f"Expected function '{ground_truth['function_call']['name']}', got '{model_function_call['name'] if model_function_call else 'None'}'"
                else:
                    result['mismatch_type'] = 'Parameters'
                    result['reason'] = f"Value differences: {', '.join(differences['param_differences'])}"
                
                # Check for semantic equivalence if needed
                if self.model and differences['needs_semantic_check']:
                    logger.info(f"Running semantic evaluation for test case {test_id}")
                    semantic_tasks = []
                    for param, (expected_val, model_val) in differences['param_values'].items():
                        task = self._evaluate_semantic_equivalence(
                            f"Parameter '{param}' for query: {user_query}",
                            expected_val,
                            model_val,
                            test_id,
                            record_type
                        )
                        semantic_tasks.append(task)
                    
                    semantic_results = await asyncio.gather(*semantic_tasks)
                    if all(r['is_semantically_equivalent'] for r in semantic_results):
                        result['result'] = 'Correct'
                        result.pop('mismatch_type', None)
                        result.pop('reason', None)
                        return {
                            'is_correct': True,
                            'detailed_result': result,
                            'semantic_comparisons': semantic_results
                        }
            
            return {
                'is_correct': result['result'] == 'Correct',
                'detailed_result': result
            }
        
        elif expected_response_type == 'text':
            # Text response evaluation
            if model_function_call is not None:
                result = {
                    'test_case': test_id,
                    'type': record_type,
                    'user_query': user_query,
                    'expected_response': ground_truth['text'],
                    'model_response': f"Made function call: {model_function_call['name']}",
                    'result': 'Incorrect',
                    'reason': 'Model made a function call when text response was expected'
                }
                return {
                    'is_correct': False,
                    'detailed_result': result
                }
            
            # Proceed with text comparison
            result = {
                'test_case': test_id,
                'type': record_type,
                'user_query': user_query,
                'expected_response': ground_truth['text'],
                'model_response': model_text,
                'result': 'Incorrect',
                'reason': 'Responses are semantically different'
            }
            
            # Exact match check
            is_exact_match = (
                ground_truth['text'].strip().lower() ==
                (model_text or '').strip().lower()
            )
            
            if is_exact_match:
                result['result'] = 'Correct'
                result.pop('reason', None)
                return {
                    'is_correct': True,
                    'detailed_result': result
                }
            
            # Check for semantic equivalence if needed
            if self.model and model_text:
                logger.info(f"Running semantic evaluation for test case {test_id}")
                semantic_result = await self._evaluate_semantic_equivalence(
                    user_query,
                    ground_truth['text'],
                    model_text,
                    test_id,
                    record_type
                )
                
                if semantic_result['is_semantically_equivalent']:
                    result['result'] = 'Correct'
                    result.pop('reason', None)
                    return {
                        'is_correct': True,
                        'detailed_result': result,
                        'semantic_comparisons': [semantic_result]
                    }
            
            return {
                'is_correct': False,
                'detailed_result': result,
                'semantic_comparisons': [semantic_result] if 'semantic_result' in locals() else None
            }
        
        else:
            logger.error(f"Unknown expected_response_type: {expected_response_type}")
            result = {
                'test_case': test_id,
                'type': record_type,
                'user_query': user_query,
                'expected_response': str(ground_truth),
                'model_response': "Error: Unknown response type expected",
                'result': 'Incorrect',
                'reason': f'Unknown expected_response_type: {expected_response_type}'
            }
            return {
                'is_correct': False,
                'detailed_result': result
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _evaluate_semantic_equivalence(self, user_query, expected_text, model_text, test_case, record_type):
        """Evaluate semantic equivalence of two text responses"""
        logger.debug(f"Starting semantic evaluation for test case {test_case} (type: {record_type})")
        try:
            # Remove markdown formatting from model text if present
            model_text = model_text.replace('**', '').replace('*', '')
            
            # Get the appropriate prompt template for this record type
            prompt_template = self.prompt_templates.get(record_type)
            if not prompt_template:
                error_msg = f"No prompt template found for type {record_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            prompt = prompt_template.format(
                question=user_query,
                text1=expected_text,
                text2=model_text
            )
            logger.debug(f"Semantic evaluation prompt: {prompt}")

            response = await self.model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0,
                    candidate_count=1,
                    max_output_tokens=1000
                )
            )
            
            # get the text from the first line of the response
            judgment = response.text.strip().split('\n')[0]
            is_equivalent = judgment.lower() == 'equivalent'
            
            comparison_result = {
                'test_case': test_case,
                'type': record_type,
                'user_query': user_query,
                'expected_text': expected_text,
                'model_text': model_text,
                'is_semantically_equivalent': is_equivalent,
                'judge_explanation': response.text
            }
            
            logger.debug(f"Semantic evaluation result: {is_equivalent}")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Error in semantic evaluation for test case {test_case}: {str(e)}")
            return {
                'test_case': test_case,
                'type': record_type,
                'user_query': user_query,
                'expected_text': expected_text,
                'model_text': model_text,
                'is_semantically_equivalent': False,
                'judge_explanation': f"Error in semantic evaluation: {str(e)}"
            }

    def save_results(self, results_dir):
        """Save evaluation results to files"""
        os.makedirs(results_dir, exist_ok=True)
        
        # Calculate metrics for each type
        metrics = {
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'by_type': {
                'tool_selection': {'total': 0, 'correct': 0},
                'text_response': {'total': 0, 'correct': 0},
                'not_supported': {'total': 0, 'correct': 0},
                'error': {'total': 0, 'correct': 0},
                'clarifying': {'total': 0, 'correct': 0}
            }
        }
        
        # Process results
        for result in self.detailed_results:
            record_type = result['type']
            metrics['total'] += 1
            metrics['by_type'][record_type]['total'] += 1
            
            if result['result'] == 'Correct':
                metrics['correct'] += 1
                metrics['by_type'][record_type]['correct'] += 1
            else:
                metrics['incorrect'] += 1
        
        # Sort all results by test case
        sorted_results = sorted(self.detailed_results, key=lambda x: x['test_case'])
        
        # Save detailed results to CSV
        results_file = os.path.join(results_dir, "test_results.csv")
        fieldnames = [
            'test_case', 'type', 'user_query', 
            'expected_function_call', 'model_function_call',
            'expected_response', 'model_response', 
            'result', 'mismatch_type', 'reason'
        ]
        
        with open(results_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in sorted_results:
                cleaned_result = {k: v for k, v in result.items() if k in fieldnames}
                writer.writerow(cleaned_result)
        
        # Save semantic comparison logs
        if self.semantic_comparisons:
            comparisons_file = os.path.join(results_dir, "semantic_comparisons.jsonl")
            with open(comparisons_file, 'w', encoding='utf-8') as f:
                for comparison in self.semantic_comparisons:
                    comparison['timestamp'] = datetime.now().isoformat()
                    f.write(json.dumps(comparison) + '\n')
        
        # Print metrics
        accuracy = (metrics['correct'] / metrics['total']) * 100 if metrics['total'] > 0 else 0
        logger.info(f"""
Results:
Total test cases: {metrics['total']}
Correct predictions: {metrics['correct']}
Incorrect predictions: {metrics['incorrect']}
Overall Accuracy: {accuracy:.2f}%

Results by type:""")
        for type_name, type_metrics in metrics['by_type'].items():
            type_accuracy = (type_metrics['correct'] / type_metrics['total'] * 100) if type_metrics['total'] > 0 else 0
            logger.info(f"{type_name}: {type_accuracy:.2f}% ({type_metrics['correct']}/{type_metrics['total']})")
