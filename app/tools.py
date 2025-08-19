# app/tools.py

"""This module defines custom tools for the multi-agent system."""

from typing import Any
from vertexai.preview.generative_models import (
    Agent,
    Tool,
    ToolConfig,
)
from vertexai.preview.generative_models.tools import FunctionDeclaration


class AgentTool(Tool):
    """A tool that delegates to another agent instance."""

    def __init__(self, agent_instance: Agent):
        """Initializes the tool.

        Args:
            agent_instance: The agent instance to delegate to.
        """
        # The agent_instance is a proper GenerativeModels Agent object
        self._agent_instance = agent_instance
        
        # A simple function declaration for the tool, which just passes a single query
        # to the sub-agent.
        declaration = FunctionDeclaration(
            name=self._agent_instance.instruction.name.lower().replace(" ", "_"),
            description=f"A specialized agent tool for {self._agent_instance.instruction.name} tasks.",
            parameters=ToolConfig.from_dict({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The full query to send to the specialized agent.",
                    }
                },
                "required": ["query"],
            }),
        )
        # Call the parent class's constructor with the function declaration
        super().__init__(function_declarations=[declaration])
        self.name = declaration.name

    def run(self, **kwargs: Any) -> Any:
        """Executes the tool by delegating to the sub-agent."""
        query = kwargs.get("query", "")
        if not query:
            return "Error: No query provided to the sub-agent."

        # The orchestrator delegates the *entire cognitive loop* to the sub-agent.
        # This is where the magic of the multi-agent system happens.
        response = self._agent_instance.generate_content(query)
        return response.text