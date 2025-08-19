# app/ii_agent.py

"""
A specialized agent for interacting with the web.

This agent's primary function is to handle internet research tasks,
including fetching real-time data and answering questions that require
up-to-the-minute information.
"""

from typing import List, Optional
from vertexai.preview.generative_models import (
    Agent,
    Tool,
    GenerativeModel,
)


# --- Internet Information AGENT PERSONA ---
_II_AGENT_PERSONA = """
Name: Internet Information Agent
Role: You are a specialized agent designed to find real-time information and perform
internet research tasks. You are an expert at using external tools to access and
process up-to-the-minute data from the web. Your ultimate goal is to provide a
comprehensive and accurate answer by synthesizing information from multiple sources.

## Core Capabilities & Behavior ##
- **Multistep Reasoning**: You must use a Plan, Execute, Evaluate, Reflect (PEER) loop
  to break down complex research queries into smaller, manageable steps.
- **Tool Usage**: You have access to a suite of web research tools, including `Google Search`
  to find initial information and `browsing_tool` to extract detailed content from specific URLs.
- **Data Synthesis**: Your final response must synthesize information from all
  relevant search results and browsing data to provide a complete answer.
- **Clarity and Transparency**: When using information from the web, make it clear
  that your answer is based on a search, and provide citations or source URLs.
"""

# --- The specialized Internet Information Agent class ---
class ii_agent(Agent):
    """
    A specialized agent for internet information gathering with a PEER loop.
    """

    def __init__(self, *args, model: GenerativeModel, **kwargs):
        super().__init__(*args, model=model, instruction=_II_AGENT_PERSONA, **kwargs)

    @property
    def tools(self) -> List[Tool]:
        """Returns the list of tools available to this agent."""
        return [
            Tool(
                function_declarations=[
                    {
                        "name": "google_search",
                        "description": "Searches Google for information from the internet.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "queries": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "One or multiple queries to Google Search.",
                                },
                            },
                            "required": ["queries"],
                        },
                    },
                    {
                        "name": "browsing_tool",
                        "description": "Extracts information from a specific URL.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "The URL to browse."},
                                "query": {"type": "string", "description": "The query for the browsing tool."},
                            },
                            "required": ["url", "query"],
                        },
                    },
                ]
            )
        ]

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Executes a tool call for the Internet Information Agent."""
        if tool_name == "google_search":
            # This is a placeholder for the actual tool execution.
            queries = params.get("queries", [])
            results = [f"Simulated search result for: {q}" for q in queries]
            return "\n".join(results)
        elif tool_name == "browsing_tool":
            # This is a placeholder for the actual tool execution.
            return f"Simulated browsing result for URL: {params.get('url')} with query: {params.get('query')}"
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def on_message(
        self,
        message: dict,
        session: dict,
    ):
        """
        The core of the ii_agent's PEER loop. This is the entry point for
        the agent's reasoning process.
        """
        # Step 1: Plan
        # The LLM, using its instruction and tools, would formulate a plan here.
        plan = self.model.generate_content(f"Plan a research strategy for the query: {message['text']}")
        print(f"Plan: {plan.text}")

        # Step 2: Execute
        # The agent would execute the steps of the plan using its tools.
        # This is where the tool calls would be made and their results saved.
        
        # Step 3: Evaluate
        # The agent would evaluate the results from the execution phase.
        
        # Step 4: Reflect & Refine
        # The agent would refine the plan or synthesize a final answer.

        # For this blueprint, we simply use the agent's core model to
        # generate a final response after a conceptual "execution."
        response = self.model.generate_content(
            f"Given the plan and results, generate a final response to the user's query: {message['text']}"
        )
        return response.text