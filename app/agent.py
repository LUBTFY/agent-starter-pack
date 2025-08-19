# app/agent.py

import os
import vertexai
import google.auth
from vertexai.generative_models import GenerativeModel, Part
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.services.index_endpoint_service import IndexEndpointServiceClient
from google.cloud.aiplatform_v1.types import FindNeighborsRequest

# Import the Google Search API to directly ground the root agent
from google_search import search

# These imports are required to wrap the specialized agents as tools.
from .code_agent import CodeAgent
from .ii_agent import IIAgent

# --- UPDATED: AGENT PERSONA WITH PEER INSTRUCTIONS ---
AGENT_PERSONA = """
Name: Vertex AI Co-Pilot.
Role: You are a senior AI/ML engineer and a specialist in Google Cloud's Vertex AI platform. Your expertise is in the entire lifecycle of AI agents, from conceptual design and architecture to deployment, evaluation, and long-term maintenance.
Goal: To be a collaborative technical co-pilot and a strategic sounding board for a developer building AI agents for real-world client applications. Your purpose is to accelerate development, improve accuracy, and uphold best practices.

## Core Capabilities & Behavior ##

### PEER (Plan, Execute, Evaluate, Refine) Pattern:
- You must always follow the PEER pattern for complex, multi-step tasks.
- **Plan:** When given a complex query, first use the `planning_tool` to break the problem into a clear, logical sequence of steps.
- **Execute:** For each step in the plan, use the appropriate tool (e.g., `search_documentation`, `Google Search`, `code_agent`, `ii_agent`) to get a result.
- **Evaluate:** After execution, use the `reflection_tool` to analyze the result. Did it meet the goal for this step? Did it generate a new problem?
- **Refine:** Based on your evaluation, refine the plan or the next step.

### Strategic Planning & Problem-Solving:
- Proactive Planning: When a new project is proposed, you must first ask clarifying questions to understand the scope, constraints, and success metrics before providing a solution.
- Structured Troubleshooting: For debugging, you will propose a logical, step-by-step diagnostic plan. You will ask for specific error logs, context, or code snippets before giving a solution.
- Architectural Guidance: You will explain the trade-offs between different architectures (e.g., single-agent vs. multi-agent, RAG vs. fine-tuning) and justify your recommendations.

### Multimodal & Coding Expertise:
- Code Generation: Generate production-ready, runnable Python code for the Vertex AI SDK (google.cloud.aiplatform), complete with comments.
- Code Analysis & Debugging: When given code or an error, identify the flaw, fix it, and explain the 'why'.
- Visual Comprehension: Analyze images (screenshots, diagrams) and explicitly reference what you see in them.

### Ethical Guardrails & Safety:
- Data Privacy: You will not ask for or store any Personally Identifiable Information (PII).
- No Hallucinations: If you are unsure of an answer, state your uncertainty and suggest how a human can verify the information.
- Transparency: You must be able to explain how you arrived at an answer. You operate as an AI assistant.
"""

# --- Configuration section ---
# These are the IDs for your Vertex AI Vector Search index, sourced from .env.
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
INDEX_ID = os.environ.get("INDEX_ID")
ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID")
DEPLOYED_INDEX_ID = os.environ.get("DEPLOYED_INDEX_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME")
MODEL_NAME = os.environ.get("MODEL_NAME")

if not all([PROJECT_ID, INDEX_ID, ENDPOINT_ID, DEPLOYED_INDEX_ID, LOCATION, EMBEDDING_MODEL_NAME, MODEL_NAME]):
    raise ValueError("One or more required environment variables are not set. Please check your .env file.")

# --- Vector Search Tool ---
class VectorSearchTool(FunctionTool):
    """A tool to search the Vertex AI Vector Search index for documentation."""

    name: str = "search_documentation"
    description: str = "Searches the Vertex AI documentation for relevant information."
    parameters: list = [
        {
            "name": "query",
            "type": "string",
            "description": "The search query to find relevant documents.",
            "required": True,
        }
    ]

    def __init__(self):
        # Get credentials and explicitly set the quota project
        credentials, project = google.auth.default()
        credentials = credentials.with_quota_project(PROJECT_ID)

        # Initialize Vertex AI clients with the explicit credentials
        vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
        self.embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

        # Use the correct client and build the endpoint name manually
        self.client = IndexEndpointServiceClient(
            client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"},
            credentials=credentials
        )

        self.endpoint_name = (
            f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{ENDPOINT_ID}"
        )

    def execute(self, query: str):
        query_embedding = self.embedding_model.get_embeddings([query])[0].values
        
        request = FindNeighborsRequest(
            index_endpoint=self.endpoint_name,
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[
                FindNeighborsRequest.Query(
                    query_vector=query_embedding,
                    neighbor_count=3,
                    return_full_datums=True,
                )
            ],
        )
        response = self.client.find_neighbors(request)

        results = []
        if response.nearest_neighbors:
            for neighbor in response.nearest_neighbors[0].neighbors:
                text = next(item.string_value for item in neighbor.metadata.strings if item.name == 'text')
                source = next(item.string_value for item in neighbor.metadata.strings if item.name == 'source')
                results.append(f"Source: {source}\nContent: {text}")
        
        if not results:
            return "No relevant documents found in the knowledge base."
        
        return "\n\n".join(results)

# --- NEW: Google Search Tool to directly ground the agent ---
class GoogleSearchTool(FunctionTool):
    """A tool to perform Google searches to ground the agent's responses."""

    name: str = "google_search"
    description: str = "A tool for searching the web for information."
    parameters: list = [
        {
            "name": "queries",
            "type": "list[str]",
            "description": "A list of queries to search for.",
            "required": True,
        }
    ]
    
    def execute(self, queries: list[str]):
        """Executes the Google search."""
        results = search(queries=queries)
        output = []
        for result in results:
            if result.results:
                for item in result.results:
                    output.append(f"Source: {item.source_title}\nURL: {item.url}\nSnippet: {item.snippet}")
        return "\n\n".join(output)

# --- AgentTool (Wrapper for specialized agents) ---
class AgentTool(FunctionTool):
    """A tool that wraps another Agent, allowing it to be called by a parent agent."""

    def __init__(self, agent_instance: Agent):
        super().__init__(
            name=agent_instance.name,
            description=f"A specialized agent for {agent_instance.name}. "
                        f"Its instruction is: {agent_instance.instruction[:200]}...",
            parameters=[
                {
                    "name": "query",
                    "type": "string",
                    "description": "The query or task to pass to the specialized agent.",
                    "required": True,
                }
            ],
        )
        self.agent_instance = agent_instance

    def execute(self, query: str):
        response = self.agent_instance.generate_response(query)
        return response.text

# --- NEW: Planning and Reflection Tools for the PEER Pattern ---
class PlanningTool(FunctionTool):
    """A tool for breaking down complex tasks into a structured, step-by-step plan."""

    name: str = "planning_tool"
    description: str = "Breaks down a user's request into a logical sequence of steps. Input a query and get a plan as output."
    parameters: list = [
        {"name": "query", "type": "string", "description": "The complex task to plan for.", "required": True}
    ]

    # NOTE: The execute method here is a placeholder. The agent's LLM will interpret
    # the tool call itself as a signal to plan, so the returned string is just
    # a confirmation message. No complex logic is needed here.
    def execute(self, query: str):
        return f"A plan has been created for the task: {query}"


class ReflectionTool(FunctionTool):
    """A tool for evaluating the results of a previous action and deciding on the next step."""

    name: str = "reflection_tool"
    description: str = "Evaluates the result of a previous step to determine if it was successful and what the next action should be."
    parameters: list = [
        {"name": "last_result", "type": "string", "description": "The result of the last action executed.", "required": True}
    ]
    
    # NOTE: Similar to the PlanningTool, this execute method is a placeholder.
    # The agent's LLM will perform the evaluation and reflection based on the
    # prompt and the provided 'last_result'.
    def execute(self, last_result: str):
        return f"The agent has evaluated the result: {last_result}"


# --- Agent instantiation ---
vector_search_tool = VectorSearchTool()
google_search_tool = GoogleSearchTool()

# Wrap the specialized agents as tools for the root agent.
code_agent_for_root = AgentTool(agent_instance=CodeAgent(model=MODEL_NAME))
ii_agent_for_root = AgentTool(agent_instance=IIAgent(model=MODEL_NAME))

# Instantiate the PEER tools
planning_tool = PlanningTool()
reflection_tool = ReflectionTool()

# Use Gemini 2.5 Pro for the main orchestration agent
root_agent = Agent(
    name="root_agent",
    model=MODEL_NAME,
    instruction=AGENT_PERSONA,
    tools=[
        vector_search_tool,
        google_search_tool,
        code_agent_for_root,
        ii_agent_for_root,
        planning_tool,
        reflection_tool,
    ]
)