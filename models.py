# models.py
import logging
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ValidationError, model_validator
from datetime import datetime

logger = logging.getLogger(__name__)

# --- Graph Representation Models ---
class InputMapping(BaseModel):
    """Defines how to map data from a source node's output to a target node's input."""
    source_operation_id: str = Field(..., description="Effective_id of the source node providing data (can be a general key if data is in a shared pool).")
    source_data_path: str = Field(..., description="JSONPath-like string to extract data (e.g., '$.id', '$.data.items[0].name', '$.extracted_key_name'). Assumes data is in a shared pool accessible by this path.")
    target_parameter_name: str = Field(..., description="Name of the parameter/field in the current node's operation.")
    target_parameter_in: Literal["path", "query", "header", "cookie", "body", "body.fieldName"] = Field(..., description="Location of the target parameter.")
    transformation: Optional[str] = Field(None, description="Optional instruction for transforming data (e.g., 'format as date string'). Placeholder for future.")

class OutputMapping(BaseModel):
    """Defines how to extract data from a node's response and where to store it."""
    source_data_path: str = Field(..., description="JSONPath-like string to extract data from the node's JSON response body (e.g., '$.id', '$.data.token').")
    target_data_key: str = Field(..., description="The key under which the extracted data will be stored in the shared 'extracted_data' pool for subsequent nodes.")
    # common_model_field: Optional[str] = Field(None, description="Optional: If this output corresponds to a known field in a common data model (e.g., 'user_id', 'product_id').")


class Node(BaseModel):
    """Represents a node (an API call) in the execution graph."""
    operationId: str = Field(..., description="Original operationId from the OpenAPI spec.")
    display_name: Optional[str] = Field(None, description="Unique name for this node instance if operationId is reused (e.g., 'getUser_step1').")
    summary: Optional[str] = Field(None, description="Short summary of the API operation.")
    description: Optional[str] = Field(None, description="Detailed description of this step's purpose in the workflow.")
    
    # Fields required for execution
    method: Optional[str] = Field(None, description="HTTP method for the API call (e.g., GET, POST). Populated during graph generation or API identification.")
    path: Optional[str] = Field(None, description="API path template (e.g., /users/{userId}). Populated during graph generation or API identification.")
    payload_description: Optional[str] = Field(None, description="Natural language description of an example request payload and expected response structure. Can also be a JSON string template.")
    
    input_mappings: List[InputMapping] = Field(default_factory=list, description="How data from previous nodes or a shared pool maps to this node's inputs.")
    output_mappings: List[OutputMapping] = Field(default_factory=list, description="How to extract data from this node's response into a shared pool.")
    
    requires_confirmation: bool = Field(False, description="If true, workflow should interrupt for user confirmation before executing this node (e.g., for POST, PUT, DELETE).")

    # Optional: Store structured parameter/request body info if LLM generates it
    # parameters_schema: Optional[List[Dict[str, Any]]] = Field(None, description="Schema of parameters for this operation, if detailed by LLM.")
    # request_body_schema: Optional[Dict[str, Any]]] = Field(None, description="Schema of the request body for this operation, if detailed by LLM.")


    @property
    def effective_id(self) -> str:
        """Returns the unique identifier for this node instance in the graph."""
        return self.display_name if self.display_name else self.operationId

class Edge(BaseModel):
    """Represents a directed edge (dependency) in the execution graph."""
    from_node: str = Field(..., description="Effective_id of the source node.")
    to_node: str = Field(..., description="Effective_id of the target node.")
    description: Optional[str] = Field(None, description="Reason for the dependency.")

    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self.from_node == other.from_node and self.to_node == other.to_node

class GraphOutput(BaseModel):
    """Represents the generated API execution graph/plan."""
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)
    description: Optional[str] = Field(None, description="Overall natural language description of the workflow's purpose and flow.")
    refinement_summary: Optional[str] = Field(None, description="Summary of the last refinement made to this graph by an LLM.")

    @model_validator(mode='after')
    def check_graph_integrity(self) -> 'GraphOutput':
        if not self.nodes: # Allow empty graph if it's being built
            return self
            
        node_effective_ids = {node.effective_id for node in self.nodes}
        if len(node_effective_ids) != len(self.nodes):
            seen_ids = set()
            duplicates = [node.effective_id for node in self.nodes if node.effective_id in seen_ids or seen_ids.add(node.effective_id)] # type: ignore
            raise ValueError(f"Duplicate node effective_ids found: {list(set(duplicates))}. Use 'display_name' for duplicate operationIds.")

        for edge in self.edges:
            if edge.from_node not in node_effective_ids and edge.from_node.upper() != "START_NODE":
                raise ValueError(f"Edge source node '{edge.from_node}' not found in graph nodes (and not START_NODE).")
            if edge.to_node not in node_effective_ids and edge.to_node.upper() != "END_NODE":
                raise ValueError(f"Edge target node '{edge.to_node}' not found in graph nodes (and not END_NODE).")
        return self

# --- State Model ---
class BotState(BaseModel):
    """Represents the full state of the conversation and processing."""
    session_id: str = Field(..., description="Unique identifier for the current session.")
    user_input: Optional[str] = Field(None, description="The latest input from the user.")

    # OpenAPI Specification related fields
    openapi_spec_string: Optional[str] = Field(None, description="Temporary storage for raw OpenAPI spec text from user.")
    openapi_spec_text: Optional[str] = Field(None, description="Successfully parsed OpenAPI spec text.")
    openapi_schema: Optional[Dict[str, Any]] = Field(None, description="Parsed OpenAPI schema as a dictionary.")
    schema_cache_key: Optional[str] = Field(None, description="Cache key for the current schema.")
    schema_summary: Optional[str] = Field(None, description="LLM-generated summary of the OpenAPI schema.")
    input_is_spec: bool = Field(False, description="Flag indicating if last input was identified as an OpenAPI spec.")

    # API Identification and Payload Descriptions
    identified_apis: List[Dict[str, Any]] = Field(default_factory=list, description="List of APIs identified from spec (operationId, method, path, summary, params, requestBody).")
    payload_descriptions: Dict[str, str] = Field(default_factory=dict, description="Maps operationId to LLM-generated example payload and response descriptions.")

    # Execution Graph / Plan
    execution_graph: Optional[GraphOutput] = Field(None, description="The generated API execution graph/plan.")
    plan_generation_goal: Optional[str] = Field(None, description="User's goal for the current execution graph.")
    graph_refinement_iterations: int = Field(0, description="Counter for graph refinement attempts.")
    max_refinement_iterations: int = Field(3, description="Maximum refinement iterations.")
    graph_regeneration_reason: Optional[str] = Field(None, description="Feedback for why graph needs regeneration/refinement.")

    # Workflow Execution State Fields
    workflow_execution_status: Literal["idle", "running", "paused_for_confirmation", "completed", "failed"] = Field("idle", description="Status of the current workflow execution.")
    workflow_execution_results: Dict[str, Any] = Field(default_factory=dict, description="Stores results from executed nodes, keyed by node effective_id.")
    workflow_extracted_data: Dict[str, Any] = Field(default_factory=dict, description="Shared data pool extracted from node responses, used as input for subsequent nodes.")
    # current_workflow_executor_id: Optional[str] = Field(None, description="Identifier for the active workflow executor instance, if needed for managing multiple.")
    # The executor instance itself might be stored in scratchpad if needed across turns for a session

    # Routing and Control Flow
    intent: Optional[str] = Field(None, description="User's high-level intent from router.")
    loop_counter: int = Field(0, description="Counter for detecting routing loops.")
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by router for specific actions.")

    # Responder Fields
    final_response: str = Field("", description="Final, user-facing response from responder.")
    response: Optional[str] = Field(None, description="Intermediate response message from nodes (cleared by responder).")

    # LangGraph internal key for routing
    next_step: Optional[str] = Field(None, alias="__next__", exclude=True, description="Internal: next LangGraph node.")

    # General working memory
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, logs, workflow executor instances, etc.")

    class Config:
        extra = 'allow' # Allow extra fields for flexibility (like storing executor instance)
        validate_assignment = True
        populate_by_name = True

    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad's reasoning_log."""
        if not isinstance(self.scratchpad, dict): self.scratchpad = {}
        reason_log = self.scratchpad.get('reasoning_log', [])
        if not isinstance(reason_log, list): reason_log = []
        timestamp = datetime.now().isoformat()
        reason_log.append({"timestamp": timestamp, "tool": tool_name, "details": details})
        self.scratchpad['reasoning_log'] = reason_log[-100:] 
        logger.debug(f"Scratchpad Updated by {tool_name}: {details[:200]}...")

