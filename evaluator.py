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
    def __init__(self, test_mode, semantic_judge_model_name=None, semantic_judge_prompt=None):
        """Initialize the evaluator"""
        self.test_mode = test_mode
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

    def _evaluate_text_response(self, record, model_response, test_case, user_query):
        """Evaluate a text response test case"""
        expected_text = record['assistant_response']['text']
        model_text = model_response.get('text', '')
        
        # Simple exact match for now
        is_correct = expected_text.strip().lower() == model_text.strip().lower()
        
        result = {
            'test_case': test_case,
            'user_query': user_query,
            'expected_text': expected_text,
            'model_text': model_text,
            'result': 'Correct' if is_correct else 'Incorrect'
        }
        
        return result

    async def _evaluate_semantic_equivalence(self, user_query, expected_call, model_call, test_case):
        """Evaluate semantic equivalence of function calls using LLM judge"""
        prompt = self.prompt_template.format(
            question=user_query,
            text1=json.dumps(expected_call, indent=2),
            text2=json.dumps(model_call, indent=2) if model_call else "No function call made"
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
            is_equivalent = 'equivalent' in judgment or 'yes' in judgment
            
            return {
                'test_case': test_case,
                'user_query': user_query,
                'expected_call': expected_call,
                'model_call': model_call,
                'is_semantically_equivalent': is_equivalent,
                'judge_explanation': response.text
            }
        except Exception as e:
            logger.error(f"Error in semantic evaluation for test case {test_case}: {str(e)}")
            return {
                'test_case': test_case,
                'user_query': user_query,
                'expected_call': expected_call,
                'model_call': model_call,
                'is_semantically_equivalent': False,
                'judge_explanation': f"Error in semantic evaluation: {str(e)}"
            }

    async def evaluate_results(self, raw_results):
        """Evaluate raw test results in parallel"""
        logger.info("Starting evaluation of raw results...")
        test_dataset = raw_results['test_dataset']
        model_responses = raw_results['model_responses']
        self.test_mode = raw_results['test_mode']
        
        self.total_tests = len(test_dataset)
        self.detailed_results = []  # Clear any existing results
        self.correct_predictions = 0
        self.incorrect_predictions = 0
        self.semantic_comparisons = []
        
        # Create tasks for parallel evaluation
        tasks = []
        for index, (record, model_response) in enumerate(zip(test_dataset, model_responses)):
            test_case = index + 1
            task = self._evaluate_test_case(record, model_response, test_case)
            tasks.append(task)
        
        # Run all evaluations in parallel
        results = await asyncio.gather(*tasks)
        
        # Process results once
        for result in results:
            if result['is_correct']:
                self.correct_predictions += 1
            else:
                self.incorrect_predictions += 1
            self.detailed_results.append(result['detailed_result'])
            if 'semantic_comparisons' in result:
                self.semantic_comparisons.extend(result['semantic_comparisons'])

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

    async def _evaluate_test_case(self, record: Dict[str, Any], model_response: Dict[str, Any], test_case: int) -> Dict[str, Any]:
        """Evaluate a single test case"""
        user_query = record['user_query']
        logger.info(f"Evaluating test case {test_case}/{self.total_tests}...")
        
        if self.test_mode == 'function_call':
            result, needs_semantic_check = self._evaluate_function_call(record, model_response, test_case, user_query)
            
            if needs_semantic_check:
                expected_call = record['assistant_response']['function_call']
                model_call = model_response.get('model_function_call')
                differences = self._get_function_call_differences(expected_call, model_call)
                
                # Run semantic evaluations in parallel
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
            result = self._evaluate_text_response(record, model_response, test_case, user_query)
            return {
                'is_correct': result['result'] == 'Correct',
                'detailed_result': result
            }

    def save_results(self, results_dir):
        """Save evaluation results to files"""
        # Save detailed results to CSV
        csv_file = os.path.join(results_dir, 'test_results.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'test_case', 'user_query', 'expected_function_call', 
                'model_function_call', 'result', 'mismatch_type', 
                'reason', 'model_response_text'
            ])
            writer.writeheader()
            writer.writerows(self.detailed_results)
            
        # Save semantic comparisons to JSONL if any exist
        if self.semantic_comparisons:
            jsonl_file = os.path.join(results_dir, 'semantic_comparisons.jsonl')
            with open(jsonl_file, 'w') as f:
                for comparison in self.semantic_comparisons:
                    comparison['timestamp'] = datetime.now().isoformat()
                    f.write(json.dumps(comparison) + '\n')