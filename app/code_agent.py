# app/code_agent.py

"""
A specialized agent for generating and debugging Python code.

This agent's primary function is to handle coding-related tasks. It includes
a `code_execution_tool` that allows it to run Python code, verify its
functionality, and debug errors.
"""

from typing import List, Optional
import os
import subprocess
import vertexai
from vertexai.preview.generative_models import (
    Agent,
    Tool,
    GenerativeModel,
)


# --- Initialize Vertex AI for this agent ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "vertex-ai-co-pilot")
LOCATION = os.environ.get("GCP_REGION", "europe-west4")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- CODE AGENT PERSONA ---
# This persona is highly specific and guides the LLM on its role.
_CODE_AGENT_PERSONA = """
Name: Code Generation and Debugging Specialist.
Role: You are an expert Python programmer and a meticulous debugger. Your primary function is to generate high-quality, runnable Python code and to identify and fix errors in existing code.
Goal: To provide accurate and efficient coding solutions, ensuring the code is well-commented, follows best practices, and is free of bugs.

## Core Capabilities & Behavior ##

### Code Generation:
- Generate complete, runnable Python code snippets or full scripts.
- Focus on clarity, efficiency, and adherence to Pythonic principles.
- Include comments to explain complex logic.

### Code Debugging:
- Analyze provided code snippets and error messages to pinpoint issues.
- Suggest specific fixes and explain the reasoning behind them.
- Ask clarifying questions if the error context is insufficient.

### Code Execution:
- You have access to a special `code_execution_tool` that can run Python code.
- Use this tool to verify your generated code and to reproduce bugs for debugging.
- Use the tool to prove your code works and to provide a clear example of its output.

### Limitations:
- Do not engage in non-coding-related conversations.
- If a request is not related to Python code generation or debugging, state that it's outside your scope.
"""

# --- Tool definition as a class constant for clarity and reusability ---
_CODE_EXECUTION_TOOL = Tool(
    function_declarations=[
        {
            "name": "code_execution_tool",
            "description": "Executes a block of Python code in a secure, sandboxed environment and returns the output or any errors. It is used to test and debug Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        }
    ]
)

# --- The specialized Code Agent class ---
class CodeAgent(Agent):
    """
    A specialized agent for Python code generation and debugging.

    This agent's primary function is to handle coding-related tasks. It includes
    a `code_execution_tool` that allows it to run Python code, verify its
    functionality, and debug errors.
    """

    def __init__(
        self,
        *args,
        model: GenerativeModel,
        instruction: Optional[str] = _CODE_AGENT_PERSONA,
        **kwargs,
    ):
        """
        Initializes the CodeAgent with a persona and a generative model.

        Args:
            model: The generative model to be used by the agent.
            *args, **kwargs: Additional arguments for the base Agent class.
        """
        super().__init__(*args, model=model, instruction=instruction, **kwargs)

    @classmethod
    def from_config(cls, agent_config: dict) -> "CodeAgent":
        """
        Factory method to create a CodeAgent from an AgentConfig.

        This allows the orchestrator to instantiate the specialized agent.

        Args:
            agent_config: The configuration object for the agent.

        Returns:
            An instance of the CodeAgent.
        """
        model = agent_config.get("model")
        assert isinstance(model, GenerativeModel)
        return cls(model=model)

    @property
    def tools(self) -> List[Tool]:
        """
        Returns the list of tools available to this agent.

        Returns:
            A list containing the `Tool` object for the code execution tool.
        """
        return [_CODE_EXECUTION_TOOL]

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """
        Executes a tool call for the CodeAgent.

        Args:
            tool_name: The name of the tool to execute.
            params: A dictionary of parameters for the tool.

        Returns:
            The output of the tool execution as a string.

        Raises:
            ValueError: If the `tool_name` is not recognized.
        """
        if tool_name == "code_execution_tool":
            code = params.get("code")
            if not code:
                return "Error: The 'code' parameter is missing from the tool call."
            return self._run_python_code(code)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _run_python_code(self, code: str) -> str:
        """
        Executes a block of Python code using a subprocess.

        This method is designed to be secure by running the code in a separate
        process with a timeout. It captures standard output and standard error.

        Args:
            code: The Python code to execute as a string.

        Returns:
            A string containing the execution result, including output or error messages.
        """
        try:
            # Note: The use of `subprocess` here provides a level of isolation,
            # but care must be taken to ensure the environment is secure. The
            # `timeout` parameter is critical for preventing infinite loops.
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            output = f"Execution successful.\nOutput:\n{result.stdout}"
            return output
        except subprocess.CalledProcessError as e:
            error_output = f"Execution failed with an error.\nError:\n{e.stderr}"
            return error_output
        except subprocess.TimeoutExpired:
            return "Execution timed out after 15 seconds. The code may be in an infinite loop."
        except Exception as e:
            return f"An unexpected error occurred during execution: {str(e)}"