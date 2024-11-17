import asyncio
from abc import ABC, abstractmethod
import openai
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
    async def generate_response(self, user_query, use_tools=False, functions=None):
        pass

    @abstractmethod
    async def get_full_prompt(self, user_query: str) -> str:
        """Get the full prompt including system message and user query"""
        pass

class OpenAIModel(LLMModel):
    def __init__(self, model_name, api_key, temperature=0, system_prompt=None):
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt or "You are a helpful assistant."
        openai.api_key = api_key

    async def get_full_prompt(self, user_query: str) -> str:
        """Get the full prompt including system message and user query"""
        return f"System: {self.system_prompt}\nUser: {user_query}"

    async def generate_response(self, user_query, use_tools=False, functions=None):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        if use_tools and functions:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=messages,
                functions=functions,
                temperature=self.temperature,
            )
        else:
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
        return response

class GeminiModel(LLMModel):
    def __init__(self, model_id, temperature=0):
        self.model = GenerativeModel(model_id)
        self.temperature = temperature
        self.generation_config = GenerationConfig(
            temperature=temperature,
            candidate_count=1
        )

    async def get_full_prompt(self, user_query: str, use_tools=False) -> str:
        """Get the full prompt including system message and user query"""
        return f"User: {user_query}"

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
