# filename: models.py
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Literal, Union
from pydantic import BaseModel, Field, ValidationError, model_validator # Import model_validator for Pydantic v2
from datetime import datetime # Import datetime for timestamping

# Module-level logger
logger = logging.getLogger(__name__)

# --- Graph Representation Models ---

# Define a model for input mapping instructions (useful for describing the plan)
class InputMapping(BaseModel):
    """Defines how to map data from previous results to a parameter of this node (as described in a plan)."""
    source_operation_id: str = Field(..., description="The operationId or effective_id of the previous node whose described result contains the source data.")
    source_data_path: str = Field(..., description="A path or expression (e.g., JSONPath like '$.id') to extract the data from the source node's described result.")
    target_parameter_name: str = Field(..., description="The name of the parameter/field in the current node's operation that this data maps to.")
    # Optional: Add parameter 'in' (path, query, header, cookie, body) for clarity/validation
    target_parameter_in: Optional[Literal["path", "query", "header", "cookie", "body"]] = Field(None, description="The location of the target parameter (path, query, header, cookie, body).")
    # Optional: Add transformation instructions if needed (e.g., format date)
    transformation: Optional[str] = Field(None, description="Optional instructions for transforming the data before mapping.")


class Node(BaseModel):
    """Represents a node (an API call description) in the execution graph."""
    operationId: str = Field(..., description="Unique identifier for the API operation (from OpenAPI spec).")
    display_name: Optional[str] = Field(None, description="A unique name for this specific node instance (e.g., 'createUser_step1'), required if using the same operationId multiple times in one graph.")
    summary: Optional[str] = Field(None, description="Short summary of the operation (from OpenAPI spec).")
    description: Optional[str] = Field(None, description="Detailed description of the operation.")
    payload_description: Optional[str] = Field(None, description="A string description of an example payload for this API call.")
    input_mappings: List[InputMapping] = Field(default_factory=list, description="Instructions on how data would be mapped from previous described results.")
    # Add fields to store actual parameters/request body descriptions if needed for description refinement
    parameters: Optional[List[Dict[str, Any]]] = Field(None, description="Description or example of parameters for this operation.")
    request_body: Optional[Dict[str, Any]] = Field(None, description="Description or example of the request body for this operation.")


    # Add a computed property or method to get the effective node ID for graph structure
    # This ensures edges/cycle checks use a unique identifier
    @property
    def effective_id(self) -> str:
        """Returns the unique identifier for this node instance in the graph."""
        return self.display_name if self.display_name else self.operationId

    # Pydantic v2 model_validator to ensure display_name is set if operationId is duplicated in a list of nodes
    # This validation would typically happen in the GraphOutput model, not Node itself,
    # as it requires context of other nodes in the list.
    # Keeping it simple for now, relying on LLM to provide display_name when needed.
    # @model_validator(mode='after')
    # def check_display_name_if_duplicated(self) -> 'Node':
    #     # This validation is tricky at the individual node level.
    #     # It's better done at the GraphOutput level after all nodes are parsed.
    #     return self


class Edge(BaseModel):
    """Represents a directed edge (dependency) in the execution graph description."""
    # Edges should now reference the effective_id (operationId or display_name)
    from_node: str = Field(..., description="The effective_id (operationId or display_name) of the source node.")
    to_node: str = Field(..., description="The effective_id (operationId or display_name) of the target node.")
    description: Optional[str] = Field(None, description="Optional description of why this dependency exists (e.g., data dependency).")
    # input_mapping is moved to Node in the new structure, but keep it here if edges also describe mappings
    # Decided to keep input_mappings only on the Node model based on the GraphOutput model prompt

    # Make Edge hashable for use in sets (use effective_id tuple)
    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self.from_node == other.from_node and self.to_node == other.to_node

class GraphOutput(BaseModel):
    """Represents the generated API execution graph description."""
    nodes: List[Node] = Field(default_factory=list, description="List of API operations (nodes) in the graph description.")
    edges: List[Edge] = Field(default_factory=list, description="List of dependencies (edges) between nodes in the graph description.")
    description: Optional[str] = Field(None, description="Natural language description of the overall workflow.")

    @model_validator(mode='after')
    def check_unique_node_ids(self) -> 'GraphOutput':
        """Validates that node effective_ids are unique within the graph."""
        node_ids = {}
        for node in self.nodes:
            if node.effective_id in node_ids:
                raise ValueError(f"Duplicate node effective_id found: '{node.effective_id}'. Use 'display_name' for duplicate operationIds.")
            node_ids[node.effective_id] = True
        return self

    @model_validator(mode='after')
    def check_edge_nodes_exist(self) -> 'GraphOutput':
        """Validates that edge source and target nodes exist in the nodes list."""
        node_effective_ids = {node.effective_id for node in self.nodes}
        for edge in self.edges:
            if edge.from_node not in node_effective_ids:
                raise ValueError(f"Edge source node '{edge.from_node}' not found in the list of nodes.")
            if edge.to_node not in node_effective_ids:
                raise ValueError(f"Edge target node '{edge.to_node}' not found in the list of nodes.")
        return self


# --- Tool Parameter Models (Keep for potential future use by planner/update_graph) ---
# These models could be used by an LLM to structure parameters for specific actions

class AddEdgeParams(BaseModel):
    """Parameters required for the add_edge tool (or planning step)."""
    from_node: str = Field(..., description="The operationId or display_name of the source node.")
    to_node: str = Field(..., description="The operationId or display_name of the target node.")
    description: Optional[str] = Field(None, description="Optional description for the new edge.")
    # Add input_mapping field if edges can define mappings
    input_mapping: List[InputMapping] = Field(default_factory=list, description="Instructions on how data would be mapped for this edge.")


class GeneratePayloadsParams(BaseModel):
    """Parameters/Instructions for generating payloads (descriptions)."""
    instructions: Optional[str] = Field(None, description="Specific user instructions for how payloads should be described.")
    target_apis: Optional[List[str]] = Field(None, description="Optional list of specific operationIds to describe payloads for.")

class GenerateGraphParams(BaseModel):
    """Parameters/Instructions for generating the execution graph description."""
    goal: Optional[str] = Field(None, description="The overall user goal or task to accomplish with the described API workflow.")
    instructions: Optional[str] = Field(None, description="Specific user instructions for how the graph should be structured.")

class PlanExecutionParams(BaseModel):
    """Parameters/Instructions for the planner."""
    goal: str = Field(..., description="The user's goal or task to create a plan for.")
    context: Optional[str] = Field(None, description="Additional context provided by the user or system.")

class ExecutePlanStepParams(BaseModel):
    """Parameters for executing a single step in the plan."""
    operation_id: str = Field(..., description="The operationId of the API call or action to simulate.")
    # Add fields for specific parameters or request body needed for the simulation step
    # These would be populated by the planner or previous execution steps
    parameters: Optional[Dict[str, Any]] = Field(None, description="Simulated parameters for the API call.")
    request_body: Optional[Dict[str, Any]] = Field(None, description="Simulated request body for the API call.")
    # Add a field to hold the simulated result of this step
    simulated_result: Optional[Dict[str, Any]] = Field(None, description="The simulated result (JSON-like) of executing this step.")


# --- State Model ---

class BotState(BaseModel):
    """Represents the full state of the conversation and processing."""
    session_id: str = Field(..., description="Unique identifier for the current session.")
    user_input: Optional[str] = Field(None, description="The latest input from the user.")

    # OpenAPI Specification related fields
    openapi_spec_string: Optional[str] = Field(None, description="Temporary storage for the raw OpenAPI specification text provided by the user in the current turn. Cleared after parsing attempt.") # Added temporary field
    openapi_spec_text: Optional[str] = Field(None, description="The raw OpenAPI specification text that was successfully parsed.") # Store successfully parsed text
    openapi_schema: Optional[Dict[str, Any]] = Field(None, description="The parsed and resolved OpenAPI schema as a dictionary.")
    schema_cache_key: Optional[str] = Field(None, description="Cache key used for the current schema.")
    schema_summary: Optional[str] = Field(None, description="LLM-generated text summary of the OpenAPI schema.")
    # Flag to indicate if the current input is likely a spec (set by router)
    input_is_spec: bool = Field(False, description="Flag indicating if the last user input was identified as an OpenAPI spec.")


    # API Identification and Payload Generation (Descriptions)
    # Store full identified API details including parameters/requestBody for context
    identified_apis: List[Dict[str, Any]] = Field(default_factory=list, description="List of APIs identified from the spec, including method, path, summary, parameters, requestBody.")
    payload_descriptions: Dict[str, str] = Field(default_factory=dict, description="Dictionary mapping operationId to generated example payload descriptions (string).")
    payload_generation_instructions: Optional[str] = Field(None, description="User instructions captured for payload description.")

    # Execution Graph Description (Represents a potential workflow structure)
    execution_graph: Optional[GraphOutput] = Field(None, description="The generated API execution graph description (represents a potential workflow structure).")
    graph_generation_instructions: Optional[str] = Field(None, description="User instructions captured for graph description.")
    # Field to hold reason for graph regeneration, if verification fails
    graph_regeneration_reason: Optional[str] = Field(None, description="Reason why the graph needs to be regenerated (e.g., cycle detected, missing APIs).")


    # Plan and Execute Fields
    # The plan is now a list of operationIds or step descriptions
    execution_plan: List[str] = Field(default_factory=list, description="Ordered list of operationIds or step descriptions for the planned execution.")
    current_plan_step: int = Field(0, description="Index of the current step in the execution_plan.")
    plan_execution_goal: Optional[str] = Field(None, description="The user's goal that initiated the current plan execution.")


    # Routing and Control Flow
    intent: Optional[str] = Field(None, description="The user's high-level intent as determined by the initial router LLM.")
    # Previous intent is now tracked implicitly by the graph history or explicitly in scratchpad if needed
    loop_counter: int = Field(0, description="Counter to detect potential loops in routing.")

    # Parameters extracted by the initial router or the planner (e.g., for update_graph or execute_plan_step)
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by the router or planner for the current action.")

    # --- Responder Fields ---
    final_response: str = Field("", description="The final, user-facing response generated by the responder.")

    # Output and Communication (Intermediate messages from core_logic nodes)
    response: Optional[str] = Field(None, description="Intermediate response message set by nodes (e.g., 'Schema parsed successfully'). Cleared by responder.")

    # LangGraph internal key for routing - exclude from serialization
    next_step: Optional[str] = Field(
        None,
        alias="__next__",
        exclude=True,
        description="Internal: the next LangGraph node to execute, set by router or nodes."
    )

    # Internal working memory - useful for storing intermediate results, simulated outputs, etc.
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, reasoning, history, planner decisions etc.")

    class Config:
        # Allow extra fields in scratchpad without validation errors
        extra = 'allow'
        # Enforce type validation on assignment to fields
        validate_assignment = True
        # Allow populating by field name as well as alias (__next__)
        populate_by_name = True


    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad."""
        # Ensure scratchpad is a dict
        if not isinstance(self.scratchpad, dict):
             self.scratchpad = {}
             logger.warning("Scratchpad was not a dictionary, re-initialized.")

        current_reason_log = self.scratchpad.get('reasoning_log', [])
        # Ensure reason_log is a list
        if not isinstance(current_reason_log, list):
             current_reason_log = []
             logger.warning("Scratchpad['reasoning_log'] was not a list, re-initialized.")

        timestamp = datetime.now().isoformat()
        new_entry = {"timestamp": timestamp, "tool": tool_name, "details": details}
        current_reason_log.append(new_entry)
        # Keep log size manageable, e.g., last 100 entries
        self.scratchpad['reasoning_log'] = current_reason_log[-100:]

        # Optionally also store a simple string log for easier viewing
        current_reason_string = self.scratchpad.get('reasoning_log_string', '')
         # Ensure reason_log_string is a string
        if not isinstance(current_reason_string, str):
             current_reason_string = ""
             logger.warning("Scratchpad['reasoning_log_string'] was not a string, re-initialized.")

        new_string_entry = f"\n---\n[{timestamp}] Tool: {tool_name}\nDetails: {details}\n---\n"
        # Keep string log size manageable, e.g., last 10000 characters
        self.scratchpad['reasoning_log_string'] = (current_reason_string + new_string_entry)[-10000:]
        logger.debug(f"Scratchpad Updated by {tool_name}: {details}")

