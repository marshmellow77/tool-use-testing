import json
import csv
import os
import logging
from datetime import datetime
from vertexai.generative_models import GenerationConfig
import asyncio
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, test_mode, semantic_judge_model_name=None, semantic_judge_prompt=None, run_both_tool_modes=False):
        """Initialize the evaluator"""
        self.test_mode = test_mode
        self.run_both_tool_modes = run_both_tool_modes
        self.total_tests = 0
        self.correct_predictions = 0
        self.incorrect_predictions = 0
        self.detailed_results = []
        self.semantic_comparisons = []
        
        # Initialize semantic evaluation if model name is provided
        self.model = None
        self.prompt_template = None
        if semantic_judge_model_name and semantic_judge_prompt:
            from vertexai.generative_models import GenerativeModel
            self.model = GenerativeModel(semantic_judge_model_name)
            with open(semantic_judge_prompt, 'r') as f:
                self.prompt_template = f.read()

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
            if str(expected_args[key]).lower() != str(model_args[key]).lower():
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
            elif str(expected_args[key]).lower() != str(model_args[key]).lower():
                differences['param_differences'].append(
                    f"{key}: expected '{expected_args[key]}', got '{model_args[key]}'"
                )
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
        
        # Check if we have separate runs for with/without tools
        if isinstance(results, dict):
            self.detailed_results = {}
            
            # Process each mode's results
            for mode in results:
                mode_results = await self._evaluate_single_run(results[mode]['test_results'])
                self.detailed_results[mode] = mode_results['detailed_results']
                
                # Add semantic comparisons if they exist
                if 'semantic_comparisons' in mode_results:
                    self.semantic_comparisons.extend(mode_results['semantic_comparisons'])
            
            # Calculate overall metrics
            total_tests = sum(len(results[mode]['test_results']) for mode in results)
            correct_predictions = sum(
                sum(1 for r in self.detailed_results[mode] if r['result'] == 'Correct')
                for mode in self.detailed_results
            )
            incorrect_predictions = total_tests - correct_predictions
            
            self.total_tests = total_tests
            self.correct_predictions = correct_predictions
            self.incorrect_predictions = incorrect_predictions
        
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
        user_query = test_case['user_query']
        ground_truth = test_case['ground_truth']
        model_function_call = test_case['model_function_call']
        model_text = test_case['model_text']
        
        logger.debug(f"Evaluating test case {test_id}")
        logger.debug(f"Ground truth: {ground_truth['text']}")
        logger.debug(f"Model text: {model_text}")
        
        if self.test_mode == 'function_call':
            # Function call evaluation
            are_identical = self._are_function_calls_identical(ground_truth['function_call'], model_function_call)
            
            result = {
                'test_case': test_id,
                'user_query': user_query,
                'expected_function_call': ground_truth['function_call'],
                'model_function_call': model_function_call,
                'result': 'Correct' if are_identical else 'Incorrect',
                'model_response': model_text  # Record model's text response
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
                    semantic_tasks = []
                    for param, (expected_val, model_val) in differences['param_values'].items():
                        task = self._evaluate_semantic_equivalence(
                            f"Parameter '{param}' for query: {user_query}",
                            expected_val,
                            model_val,
                            test_id
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
        
        else:
            # Text response evaluation
            if model_function_call is not None:
                # Model made a function call when it shouldn't have
                result = {
                    'test_case': test_id,
                    'user_query': user_query,
                    'expected_response': ground_truth['text'],
                    'model_response': f"Made function call: {model_function_call['name']}",
                    'result': 'Incorrect',
                    'reason': 'Model made a function call when it should not have'
                }
                return {
                    'is_correct': False,
                    'detailed_result': result
                }
            else:
                # Proceed with text comparison
                result = {
                    'test_case': test_id,
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
                
                logger.debug(f"Exact match check: {is_exact_match}")
                
                if is_exact_match:
                    result['result'] = 'Correct'
                    result.pop('reason', None)
                    return {
                        'is_correct': True,
                        'detailed_result': result
                    }
                
                # Check for semantic equivalence if needed
                logger.debug(f"Semantic judge model initialized: {self.model is not None}")
                logger.debug(f"Model text present: {bool(model_text)}")
                
                if self.model and model_text:
                    logger.debug("Proceeding with semantic evaluation")
                    semantic_result = await self._evaluate_semantic_equivalence(
                        user_query,
                        ground_truth['text'],
                        model_text,
                        test_id
                    )
                    
                    if semantic_result['is_semantically_equivalent']:
                        result['result'] = 'Correct'
                        result.pop('reason', None)
                        return {
                            'is_correct': True,
                            'detailed_result': result,
                            'semantic_comparisons': [semantic_result]
                        }
                else:
                    logger.debug("Skipping semantic evaluation - conditions not met")
                
                return {
                    'is_correct': False,
                    'detailed_result': result
                }

    async def _evaluate_semantic_equivalence(self, user_query, expected_text, model_text, test_case):
        """Evaluate semantic equivalence of two text responses"""
        logger.debug(f"Starting semantic evaluation for test case {test_case}")
        try:
            # Remove markdown formatting from model text if present
            model_text = model_text.replace('**', '').replace('*', '')
            
            prompt = self.prompt_template.format(
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
            
            judgment = response.text.strip().lower()
            logger.debug(f"Semantic judge response: {judgment}")
            
            # Check for explicit "different" or "not equivalent" before checking for "equivalent"
            is_equivalent = (
                'equivalent' in judgment and 
                'not equivalent' not in judgment and 
                'not semantically equivalent' not in judgment and
                'different' not in judgment
            )
            
            comparison_result = {
                'test_case': test_case,
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
                'user_query': user_query,
                'expected_text': expected_text,
                'model_text': model_text,
                'is_semantically_equivalent': False,
                'judge_explanation': f"Error in semantic evaluation: {str(e)}"
            }

    def save_results(self, results_dir):
        """Save evaluation results to files"""
        os.makedirs(results_dir, exist_ok=True)
        
        # Calculate metrics for each mode
        no_tools_metrics = {'total': 0, 'correct': 0, 'incorrect': 0}
        with_tools_metrics = {'total': 0, 'correct': 0, 'incorrect': 0}
        
        # Process results based on modes present in the data
        all_results = []
        if 'no_tools' in self.detailed_results:
            for result in self.detailed_results['no_tools']:
                result['run_type'] = 'no_tools'
                no_tools_metrics['total'] += 1
                if result['result'] == 'Correct':
                    no_tools_metrics['correct'] += 1
                else:
                    no_tools_metrics['incorrect'] += 1
                all_results.append(result)
        
        if 'with_tools' in self.detailed_results:
            for result in self.detailed_results['with_tools']:
                result['run_type'] = 'with_tools'
                with_tools_metrics['total'] += 1
                if result['result'] == 'Correct':
                    with_tools_metrics['correct'] += 1
                else:
                    with_tools_metrics['incorrect'] += 1
                all_results.append(result)
        
        # Sort all results by test case
        sorted_results = sorted(all_results, key=lambda x: x['test_case'])
        
        # Save detailed results to CSV
        results_file = os.path.join(results_dir, "test_results.csv")
        fieldnames = (
            ['test_case', 'user_query', 'expected_function_call', 'model_function_call', 
             'result', 'mismatch_type', 'reason', 'model_response', 'run_type']
            if self.test_mode == 'function_call' else
            ['test_case', 'user_query', 'expected_response', 'model_response', 
             'result', 'reason', 'run_type']
        )
        
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
        
        # Print metrics based on run mode
        if no_tools_metrics['total'] > 0:
            no_tools_accuracy = (no_tools_metrics['correct'] / no_tools_metrics['total']) * 100
            logger.info(f"""
No Tools Mode:
Total test cases: {no_tools_metrics['total']}
Correct predictions: {no_tools_metrics['correct']}
Incorrect predictions: {no_tools_metrics['incorrect']}
Accuracy: {no_tools_accuracy:.2f}%
""")
        
        if with_tools_metrics['total'] > 0:
            with_tools_accuracy = (with_tools_metrics['correct'] / with_tools_metrics['total']) * 100
            logger.info(f"""
{'' if no_tools_metrics['total'] > 0 else 'Single Mode '}Results:
Total test cases: {with_tools_metrics['total']}
Correct predictions: {with_tools_metrics['correct']}
Incorrect predictions: {with_tools_metrics['incorrect']}
Accuracy: {with_tools_accuracy:.2f}%
""")
