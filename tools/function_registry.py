from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vertexai.generative_models import FunctionDeclaration, Tool

@dataclass
class FunctionParameter:
    name: str
    type: str
    description: str
    required: bool = True

@dataclass
class Function:
    name: str
    description: str
    parameters: List[FunctionParameter]

    def to_openai_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    }
                    for param in self.parameters
                },
                "required": [
                    param.name for param in self.parameters 
                    if param.required
                ]
            }
        }

    def to_gemini_format(self) -> FunctionDeclaration:
        return FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self.to_openai_format()["parameters"]
        )

class FunctionRegistry:
    def __init__(self):
        self.functions: Dict[str, Function] = {}

    def register(self, function: Function):
        self.functions[function.name] = function

    def get_functions_for_model(self, model_type: str) -> Union[List[dict], Tool]:
        if model_type == "openai":
            return [f.to_openai_format() for f in self.functions.values()]
        elif model_type == "gemini":
            declarations = [f.to_gemini_format() for f in self.functions.values()]
            return Tool(function_declarations=declarations)
        else:
            raise ValueError(f"Unsupported model type: {model_type}") 