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

    def _are_function_calls_identical(self, expected_call, model_call):
        """Check if two function calls are identical"""
        if expected_call['name'].lower() != model_call['name'].lower():
            return False
            
        expected_args = expected_call['arguments']
        model_args = model_call['arguments']
        
        # Check if they have the same parameters
        if set(expected_args.keys()) != set(model_args.keys()):
            return False
            
        # Check if parameter values match
        for key in expected_args:
            if str(expected_args[key]) != str(model_args[key]):
                return False
                
        return True

    def _are_values_equivalent(self, val1, val2):
        """Compare two values accounting for case and number format differences"""
        # Convert to strings for comparison
        str1 = str(val1).strip()
        str2 = str(val2).strip()
        
        # Try numeric comparison first
        try:
            num1 = float(str1)
            num2 = float(str2)
            return abs(num1 - num2) < 1e-10  # Using small epsilon for float comparison
        except (ValueError, TypeError):
            # If not numbers, compare strings case-insensitively
            return str1.lower() == str2.lower()

    def _get_function_call_differences(self, expected_call, model_call):
        """Get detailed differences between function calls"""
        differences = {
            'name_mismatch': False,
            'param_differences': [],
            'param_values': {},
            'needs_semantic_check': False
        }
        
        # Check function name
        if expected_call['name'].lower() != model_call['name'].lower():
            differences['name_mismatch'] = True
            return differences
            
        expected_args = expected_call['arguments']
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
                differences['param_values'][key] = (expected_args[key], model_args[key])
                differences['needs_semantic_check'] = True
                
        for key in model_args:
            if key not in expected_args:
                differences['param_differences'].append(f"Unexpected parameter: {key}")
                differences['needs_semantic_check'] = True
                
        return differences

    def _evaluate_function_call(self, record, model_response, test_case, user_query):
        """Evaluate a function call test case"""
        expected_function_call = record['assistant_response']['function_call']
        model_function_call = model_response.get('model_function_call')

        # Check if model made a function call
        if not model_function_call:
            model_text = model_response.get('text', 'No text response available')
            result = {
                'test_case': test_case,
                'user_query': user_query,
                'expected_function_call': expected_function_call,
                'model_function_call': None,
                'result': 'Incorrect',
                'mismatch_type': 'no_function_call',
                'reason': 'Model provided text response instead of function call',
                'model_response_text': model_text
            }
            return result, False

        # Compare function calls
        are_identical = self._are_function_calls_identical(expected_function_call, model_function_call)
        needs_semantic_check = not are_identical and self.model is not None

        if are_identical:
            result = {
                'test_case': test_case,
                'user_query': user_query,
                'expected_function_call': expected_function_call,
                'model_function_call': model_function_call,
                'result': 'Correct'
            }
        else:
            # Find what's different
            differences = self._get_function_call_differences(expected_function_call, model_function_call)
            
            if differences['name_mismatch']:
                mismatch_type = 'Function'
                reason = f"Expected function '{expected_function_call['name']}', got '{model_function_call['name']}'"
                is_correct = False
            else:
                mismatch_type = 'Parameters'
                reason = f"Value differences: {', '.join(differences['param_differences'])}"
                is_correct = False

            result = {
                'test_case': test_case,
                'user_query': user_query,
                'expected_function_call': expected_function_call,
                'model_function_call': model_function_call,
                'result': 'Incorrect' if not is_correct else 'Correct',
                'mismatch_type': mismatch_type,
                'reason': reason
            }

        return result, needs_semantic_check

    async def _evaluate_text_response(self, record, model_response, test_case, user_query):
        """Evaluate a text response test case"""
        expected_text = record['assistant_response']['content']
        
        # If it's a processed response (dict)
        if isinstance(model_response, dict):
            # Check for function call in model response first
            if model_response.get('model_function_call'):
                function_call = model_response['model_function_call']
                return {
                    'test_case': test_case,
                    'user_query': user_query,
                    'expected_response': expected_text,
                    'model_response': f"Made function call: {function_call['name']}",
                    'result': 'Incorrect',
                    'reason': 'Model made a function call when it should not have'
                }, False
            elif 'function_calls' in model_response:
                function_call = model_response['function_calls'][0]
                return {
                    'test_case': test_case,
                    'user_query': user_query,
                    'expected_response': expected_text,
                    'model_response': f"Made function call: {function_call['name']}",
                    'result': 'Incorrect',
                    'reason': 'Model made a function call when it should not have'
                }, False
            
            # Check for error message containing function call info
            if 'error' in model_response:
                error_msg = model_response['error']
                if '"name": "' in error_msg:
                    function_name = error_msg.split('"name": "')[1].split('"')[0]
                    return {
                        'test_case': test_case,
                        'user_query': user_query,
                        'expected_response': expected_text,
                        'model_response': f"Made function call: {function_name}",
                        'result': 'Incorrect',
                        'reason': 'Model made a function call when it should not have'
                    }, False
            
            model_text = (
                model_response.get('model_response') or
                model_response.get('text') or
                model_response.get('content', '')
            ).strip()
            
            if not model_text:  # If no text response, assume function call
                function_name = 'unknown'
                if model_response.get('model_function_call'):
                    function_name = model_response['model_function_call'].get('name', 'unknown')
                return {
                    'test_case': test_case,
                    'user_query': user_query,
                    'expected_response': expected_text,
                    'model_response': f"Made function call: {function_name}",
                    'result': 'Incorrect',
                    'reason': 'Model made a function call when it should not have'
                }, False
        
        # Fallback for other response types
        else:
            try:
                model_text = model_response.text if hasattr(model_response, 'text') else f"Made function call: unknown"
            except (AttributeError, TypeError):
                return {
                    'test_case': test_case,
                    'user_query': user_query,
                    'expected_response': expected_text,
                    'model_response': "Made function call: unknown",
                    'result': 'Incorrect',
                    'reason': 'Model made a function call when it should not have'
                }, False

        # Normal text response evaluation
        result = {
            'test_case': test_case,
            'user_query': user_query,
            'expected_response': expected_text,
            'model_response': model_text,
            'result': 'Incorrect',
            'reason': 'Responses are semantically different'
        }
        
        # Check for exact match first
        is_exact_match = expected_text.strip().lower() == model_text.strip().lower()
        needs_semantic_check = not is_exact_match and self.model is not None
        
        if is_exact_match:
            result['result'] = 'Correct'
            result.pop('reason', None)
        
        return result, needs_semantic_check

    async def _evaluate_semantic_equivalence(self, user_query, expected_text, model_text, test_case):
        """Evaluate semantic equivalence using LLM judge"""
        # Clean and format the texts for comparison
        expected_text = str(expected_text).strip()
        model_text = str(model_text).strip()
        
        # Remove markdown formatting from model text if present
        model_text = model_text.replace('**', '').replace('*', '')
        
        prompt = self.prompt_template.format(
            question=user_query,
            text1=expected_text,
            text2=model_text
        )

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    temperature=0,
                    candidate_count=1,
                    max_output_tokens=1000
                )
            )
            
            judgment = response.text.strip().lower()
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

    async def evaluate_results(self, raw_results):
        """Evaluate raw test results in parallel"""
        logger.info("Starting evaluation of raw results...")
        
        if self.test_mode == 'no_function' and self.run_both_tool_modes:
            # Handle both tool modes
            combined_results = {
                'no_tools': await self._evaluate_single_run(raw_results['no_tools']),
                'with_tools': await self._evaluate_single_run(raw_results['with_tools'])
            }
            return combined_results
        else:
            # Single run evaluation
            return await self._evaluate_single_run(raw_results)

    async def _evaluate_single_run(self, raw_results):
        """Evaluate a single run of test results"""
        test_dataset = raw_results['test_dataset']
        model_responses = raw_results['model_responses']
        run_type = raw_results.get('run_type', 'default')
        
        # Reset counters for this run
        self.total_tests = len(test_dataset)
        self.correct_predictions = 0
        self.incorrect_predictions = 0
        
        # Create tasks for parallel evaluation
        tasks = []
        for index, (record, model_response) in enumerate(zip(test_dataset, model_responses)):
            test_case = index + 1
            task = self._evaluate_test_case(record, model_response, test_case)
            tasks.append(task)
        
        # Run all evaluations in parallel
        results = await asyncio.gather(*tasks)
        
        # Process results
        run_results = []
        for result in results:
            if result['is_correct']:
                self.correct_predictions += 1
            else:
                self.incorrect_predictions += 1
            # Add run_type to the detailed result
            result['detailed_result']['run_type'] = run_type
            run_results.append(result['detailed_result'])
            if 'semantic_comparisons' in result:
                self.semantic_comparisons.extend(result['semantic_comparisons'])

        accuracy = (self.correct_predictions / self.total_tests) * 100 if self.total_tests > 0 else 0
        
        logger.info("Evaluation completed.")
        
        # Store results for this run
        if not hasattr(self, 'all_detailed_results'):
            self.all_detailed_results = []
        self.all_detailed_results.extend(run_results)
        
        return {
            'total_tests': self.total_tests,
            'correct_predictions': self.correct_predictions,
            'incorrect_predictions': self.incorrect_predictions,
            'accuracy': accuracy,
            'detailed_results': run_results,
            'semantic_comparisons': self.semantic_comparisons
        }

    async def _evaluate_test_case(self, record: Dict[str, Any], model_response: Dict[str, Any], test_case: int) -> Dict[str, Any]:
        """Evaluate a single test case"""
        user_query = record['user_query']
        run_type = model_response.get('run_type', 'default')
        logger.info(f"Evaluating test case {test_case}/{self.total_tests}...")
        
        if self.test_mode == 'function_call':
            result, needs_semantic_check = self._evaluate_function_call(record, model_response, test_case, user_query)
            result['run_type'] = run_type
            
            if needs_semantic_check:
                expected_call = record['assistant_response']['function_call']
                model_call = model_response.get('model_function_call')
                differences = self._get_function_call_differences(expected_call, model_call)
                
                semantic_tasks = []
                for param, (expected_val, model_val) in differences['param_values'].items():
                    task = self._evaluate_semantic_equivalence(
                        f"Parameter '{param}' for query: {user_query}",
                        expected_val,
                        model_val,
                        test_case
                    )
                    semantic_tasks.append(task)
                
                semantic_results = await asyncio.gather(*semantic_tasks)
                all_params_equivalent = all(r['is_semantically_equivalent'] for r in semantic_results)
                
                if all_params_equivalent:
                    result['result'] = 'Correct'
                    result.pop('mismatch_type', None)
                    result.pop('reason', None)
                    is_correct = True
                else:
                    is_correct = False
                
                return {
                    'is_correct': is_correct,
                    'detailed_result': result,
                    'semantic_comparisons': semantic_results
                }
            
            return {
                'is_correct': result['result'] == 'Correct',
                'detailed_result': result
            }
        else:
            # Get the expected and actual responses
            expected_response = record['assistant_response']['content']
            
            # Extract model response, handling different response formats
            if isinstance(model_response, dict):
                model_text = (
                    model_response.get('model_response') or
                    model_response.get('text') or
                    model_response.get('content', '')
                ).strip()
            else:
                model_text = str(model_response).strip()

            # Create the result dictionary
            result = {
                'test_case': test_case,
                'user_query': user_query,
                'expected_response': expected_response,
                'model_response': model_text,
                'result': 'Incorrect',  # Default to incorrect
                'reason': 'Responses are semantically different',  # Default reason
                'run_type': run_type
            }

            # Check for exact match first
            is_exact_match = expected_response.strip().lower() == model_text.strip().lower()
            needs_semantic_check = not is_exact_match and self.model is not None

            if is_exact_match:
                result['result'] = 'Correct'
                result.pop('reason', None)
                return {
                    'is_correct': True,
                    'detailed_result': result
                }

            if needs_semantic_check:
                semantic_result = await self._evaluate_semantic_equivalence(
                    user_query,
                    expected_response,
                    model_text,
                    test_case
                )
                
                if semantic_result['is_semantically_equivalent']:
                    result['result'] = 'Correct'
                    result.pop('reason', None)
                    is_correct = True
                else:
                    is_correct = False

                return {
                    'is_correct': is_correct,
                    'detailed_result': result,
                    'semantic_comparisons': [semantic_result]
                }

            return {
                'is_correct': False,
                'detailed_result': result
            }

    def save_results(self, results_dir):
        """Save evaluation results to files"""
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results to CSV
        results_file = os.path.join(results_dir, "test_results.csv")
        fieldnames = (
            ['test_case', 'user_query', 'expected_function_call', 'model_function_call', 
             'result', 'mismatch_type', 'reason', 'model_response', 'run_type']
            if self.test_mode == 'function_call' else
            ['test_case', 'user_query', 'expected_response', 'model_response', 
             'result', 'reason', 'run_type']
        )
        
        # If we have results from both modes, combine them
        all_results = []
        if isinstance(self.all_detailed_results, dict):
            # Add results from no_tools run
            if 'no_tools' in self.all_detailed_results:
                for result in self.all_detailed_results['no_tools']:
                    result['run_type'] = 'no_tools'
                    all_results.append(result)
            
            # Add results from with_tools run
            if 'with_tools' in self.all_detailed_results:
                for result in self.all_detailed_results['with_tools']:
                    result['run_type'] = 'with_tools'
                    all_results.append(result)
        else:
            # Single mode results
            all_results = self.all_detailed_results
        
        # Sort all results by test case
        sorted_results = sorted(all_results, key=lambda x: x['test_case'])
        
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
                    
        return results_dir