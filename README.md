# LLM Model Tester

This app allows you to test language models (LLMs) on their tool selection capabilities, such as via function calling. It supports both Gemini and OpenAI GPT-4o models, and provides evaluation logic to compute accuracies.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the App End-to-End](#running-the-app-end-to-end)
  - [Testing with Gemini](#testing-with-gemini)
  - [Testing with OpenAI GPT-4o](#testing-with-openai-gpt-4o)
- [Using the App with Custom Responses](#using-the-app-with-custom-responses)
  - [Generating Your Own Responses](#generating-your-own-responses)
  - [Using the Evaluation Logic](#using-the-evaluation-logic)
- [Customizing the App](#customizing-the-app)
- [License](#license)

## Repository Structure

```
your_project/
├── main.py
├── models.py
├── evaluator.py
├── model_tester.py
├── tools/
│   └── functions.py
├── datasets/
│   ├── function_call_dataset.json
│   └── text_response_dataset.json
├── prompts/
│   ├── semantic_judge_function_call.txt
│   └── semantic_judge_text.txt
├── results/
├── requirements.txt
├── README.md
```

## Requirements

- Python 3.7 or higher
- OpenAI API key (for OpenAI models and semantic judge)
- Google Cloud credentials (for Gemini model)
- Necessary Python packages (see `requirements.txt`)

## Setup

1. **Clone the repository**:

   ```
   git clone https://github.com/yourusername/your_project.git
   cd your_project
   ```

2. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Set up API keys**:

   - For **OpenAI models**, obtain your API key from your OpenAI account.
   - For **Gemini model**, ensure your Google Cloud credentials are set up properly.

## Running the App End-to-End

### Testing with Gemini

```
python main.py --model-type gemini --mode function_call --dataset datasets/function_call_dataset.json
```

### Testing with OpenAI GPT-4o

```
python main.py --model-type openai --mode function_call --dataset datasets/function_call_dataset.json --openai-api-key YOUR_OPENAI_API_KEY
```

Replace `YOUR_OPENAI_API_KEY` with your actual OpenAI API key.

## Using the App with Custom Responses

If you want to use the test dataset, generate your own responses, and then use the evaluation logic, follow these steps:

### Generating Your Own Responses

1. **Load the test dataset**:

   ```python
   import json

   def load_dataset(dataset_path):
       with open(dataset_path, 'r') as f:
           return json.load(f)

   test_dataset = load_dataset('datasets/function_call_dataset.json')
   ```

2. **Generate responses using your own model or method**:

   ```python
   model_responses = []

   for record in test_dataset:
       user_query = record['user_query']
       # Generate response using your own logic
       # For example:
       response = my_custom_model.generate_response(user_query)
       # For function_call mode, response should be a dict with 'name' and 'arguments'
       # For no_function mode, response should be a string
       model_responses.append(response)
   ```

### Using the Evaluation Logic

1. **Import the Evaluator**:

   ```python
   from evaluator import Evaluator
   ```

2. **Initialize the Evaluator**:

   ```python
   evaluator = Evaluator(
       test_mode='function_call',  # or 'no_function' for text responses
       semantic_judge_model_name='gpt-4',  # or your preferred model
       api_key='YOUR_OPENAI_API_KEY'
   )
   ```

3. **Run the Evaluation**:

   ```python
   import asyncio

   results = asyncio.run(evaluator.evaluate(test_dataset, model_responses))
   ```

4. **Review the Results**:

   ```python
   print(f"Total tests: {results['total_tests']}")
   print(f"Correct predictions: {results['correct_predictions']}")
   print(f"Incorrect predictions: {results['incorrect_predictions']}")
   print(f"Accuracy: {results['accuracy']:.2f}%")
   ```

## Customizing the App

- **Adding More Functions**: Add more function definitions to `tools/functions.py`, ensuring each function includes `"additionalProperties": False` in the parameters.
- **Modifying Datasets**: Update the datasets in the `datasets/` directory as needed.
- **Adjusting Prompts**: Modify the prompts in the `prompts/` directory if needed.

## License

This project is licensed under the MIT License.
