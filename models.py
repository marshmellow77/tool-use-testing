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

logger = logging.getLogger(__name__)

class LLMModel(ABC):
    @abstractmethod
    async def generate_response(self, user_query, use_tools=False, tool=None):
        pass

class OpenAIModel(LLMModel):
    def __init__(self, model_name, api_key, temperature=0, system_prompt=None):
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt or "You are a helpful assistant."
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_response(self, user_query, use_tools=False, tool=None):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        try:
            if use_tools and tool:
                tools = [{
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }]
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": tool["name"]}},
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

    async def generate_response(self, user_query, use_tools=False, tool=None):
        try:
            prompt = Content(
                role="user",
                parts=[Part.from_text(user_query)]
            )

            if use_tools and tool:
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
