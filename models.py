import asyncio
from abc import ABC, abstractmethod
from openai import AsyncOpenAI
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part,
    Tool,
    ToolConfig,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from google.api_core import exceptions
import logging
import json
# import weave

logger = logging.getLogger(__name__)

class LLMModel(ABC):
    @abstractmethod
    async def generate_response(self, user_query, tool=None):
        pass

class OpenAIModel(LLMModel):
    def __init__(self, model_name, api_key, temperature=0, system_prompt=None):
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.client = AsyncOpenAI(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def generate_response(self, user_query, tool=None):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            if tool:
                tools = []
                for t in tool:
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["parameters"]
                        }
                    })
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.temperature,
                )
                
                message = response.choices[0].message
                
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    return {
                        "model_function_call": {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments)
                        },
                        "full_model_response": message.content,
                        "error": None
                    }
                
                return {
                    "model_function_call": None,
                    "full_model_response": message.content,
                    "error": None
                }
            else:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                )
                
                return {
                    "model_function_call": None,
                    "full_model_response": response.choices[0].message.content,
                    "error": None
                }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "model_function_call": None,
                "full_model_response": None,
                "error": str(e)
            }

class GeminiModel(LLMModel):
    def __init__(self, model_id, temperature=0):
        self.model = GenerativeModel(model_id)
        self.temperature = temperature
        self.generation_config = GenerationConfig(
            temperature=temperature,
            candidate_count=1
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((exceptions.ResourceExhausted, exceptions.ServiceUnavailable)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def generate_response(self, user_query, tool=None):
        try:
            prompt = Content(
                role="user",
                parts=[Part.from_text(user_query)]
            )
            if tool:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=self.generation_config,
                    tools=[tool],
                    tool_config=ToolConfig(
                        function_calling_config=ToolConfig.FunctionCallingConfig(
                            mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                        )
                    )
                )
            else:
                response = await self.model.generate_content_async(
                    prompt,
                    generation_config=self.generation_config
                )
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
