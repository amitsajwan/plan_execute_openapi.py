# graph.py
import logging
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver # For type hinting checkpointer

# Assuming models.py is in the same directory or accessible in PYTHONPATH
from models import BotState
# Assuming core_logic.py and router.py are accessible
from core_logic import OpenAPICoreLogic, APIExecutor # Import APIExecutor for type hint
from router import OpenAPIRouter

logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """
    Sets final_response from state.response if available, clears intermediate response,
    and prepares the state for ending the current graph turn.
    """
    tool_name = "responder"
    logger.info(f"Responder ({tool_name}): Entered. Current state.response = '{state.response}', current state.final_response = '{state.final_response}'")
    state.update_scratchpad_reason(tool_name, f"Finalizing response. Initial state.response: '{state.response}'")
    
    if state.response: # If an intermediate response was set by the last node
        state.final_response = state.response
        logger.info(f"Responder ({tool_name}): Set final_response from state.response: '{state.final_response[:200]}...'")
    elif not state.final_response: # Only set a default if final_response isn't already set (e.g., by a direct error)
        state.final_response = "Processing complete. How can I help you further?"
        logger.warning(f"Responder ({tool_name}): state.response was empty/None. Using default final_response: '{state.final_response}'")
    else:
        # This case means state.response was falsey, but state.final_response already had a value (e.g. from a previous turn's error).
        logger.info(f"Responder ({tool_name}): state.response was falsey, but final_response was already set to: '{state.final_response[:200]}...'. No change to final_response.")

    # Clear fields for the next turn
    state.response = None # Clear intermediate response
    state.next_step = None # Clear routing directive from the previous node
    state.intent = None # Clear current intent
    # state.user_input is typically cleared or updated by the main loop receiving new input.

    state.update_scratchpad_reason(tool_name, f"Final response set in state: {state.final_response[:200]}...")
    logger.info(f"Responder ({tool_name}): Exiting. state.final_response = '{state.final_response}', state.response = '{state.response}'")
    return state


# Modified to accept api_executor_instance
def build_graph(
    router_llm: Any,
    worker_llm: Any,
    api_executor_instance: APIExecutor, # Added
    checkpointer: BaseCheckpointSaver
) -> StateGraph:
    """Builds and compiles the LangGraph StateGraph for the OpenAPI agent."""
    logger.info("Building LangGraph graph...")

    # Instantiate core logic and router, passing the api_executor
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm, api_executor_instance=api_executor_instance)
    router_instance = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # --- Add Nodes ---
    # Router node
    builder.add_node("router", router_instance.route)

    # Core logic nodes from OpenAPICoreLogic
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("process_schema_pipeline", core_logic.process_schema_pipeline)
    builder.add_node("_generate_execution_graph", core_logic._generate_execution_graph)
    builder.add_node("verify_graph", core_logic.verify_graph)
    builder.add_node("refine_api_graph", core_logic.refine_api_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("get_graph_json", core_logic.get_graph_json)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    
    # Interactive processing nodes
    builder.add_node("interactive_query_planner", core_logic.interactive_query_planner)
    builder.add_node("interactive_query_executor", core_logic.interactive_query_executor)

    # Workflow execution setup node (called by router or interactive_query_executor)
    # The actual execution happens via WorkflowExecutor, managed by main.py loop after this setup.
    builder.add_node("setup_workflow_execution", core_logic.setup_workflow_execution)
    # resume_workflow_with_payload is called by interactive_query_executor, not a direct graph node from router.

    # Handling nodes
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    
    # Responder node (finalizes output for the user)
    builder.add_node("responder", finalize_response)

    # --- Define Edges ---
    # Entry point
    builder.add_edge(START, "router")

    # Conditional edges from the router based on determined intent
    # The OpenAPIRouter.AVAILABLE_INTENTS Literal includes all valid node names
    # that the router can directly decide to go to.
    router_conditional_edges: Dict[str, str] = {
        # Maps intent value to the node name (usually the same)
        intent_val: intent_val for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
    }
    # Remove intents that are not direct graph nodes or are handled internally by other nodes
    # For example, interactive_query_executor is a node, but its sub-actions are internal to core_logic.
    # 'process_schema_pipeline' is a valid node.
    # '_generate_execution_graph' is a valid node.
    # 'resume_workflow_with_payload' is handled within interactive_query_executor.

    # Ensure all keys in router_conditional_edges are actual nodes added to the builder.
    # This is implicitly checked by LangGraph at compile time if an edge points to a non-existent node.

    builder.add_conditional_edges(
        "router", # Source node
        lambda state: state.intent, # Function that returns the key for the conditional edge
        router_conditional_edges # Mapping from key to target node name
    )

    # Define how nodes that set 'state.next_step' should be routed
    def route_from_internal_node_state(state: BotState) -> str:
        """Determines the next node based on state.next_step set by an internal node."""
        next_node_name = state.next_step
        if not next_node_name:
            # This should ideally not happen if nodes always set a next_step or default to responder
            logger.warning(f"Node (intent: {state.intent or 'Unknown'}) did not set state.next_step. Defaulting to 'responder'.")
            return "responder"
        
        logger.debug(f"Routing from internal node. Previous intent: '{state.intent}', Next step decided by node: '{next_node_name}'")
        return next_node_name

    # List all nodes that are expected to set 'state.next_step' to guide their own routing
    nodes_that_set_next_step = [
        "parse_openapi_spec", "process_schema_pipeline", "_generate_execution_graph",
        "verify_graph", "refine_api_graph", "describe_graph", "get_graph_json",
        "answer_openapi_query", "interactive_query_planner", "interactive_query_executor",
        "setup_workflow_execution", # This node will set next_step = "responder"
        "handle_unknown", "handle_loop"
        # 'router' is handled by its own conditional_edges based on 'intent'
        # 'responder' goes to END
    ]

    # Create a mapping for all possible target nodes from these internal nodes
    # This includes all nodes in the graph as potential targets.
    all_possible_internal_targets: Dict[str, str] = {
        node_name: node_name for node_name in builder.nodes # builder.nodes gives all added node names
    }
    if "router" not in all_possible_internal_targets: all_possible_internal_targets["router"] = "router" # Ensure router is a target
    if "responder" not in all_possible_internal_targets: all_possible_internal_targets["responder"] = "responder" # Ensure responder is a target


    for node_name in nodes_that_set_next_step:
        if node_name in builder.nodes: # Ensure the source node itself was added
            builder.add_conditional_edges(
                node_name,
                route_from_internal_node_state,
                all_possible_internal_targets # Any node can be a target if specified by state.next_step
            )
        else:
            logger.error(f"Configuration error: Node '{node_name}' listed in 'nodes_that_set_next_step' was not added to the graph builder.")

    # Terminal edge
    builder.add_edge("responder", END)

    # Compile the graph
    app = builder.compile(checkpointer=checkpointer)
    logger.info("LangGraph graph compiled successfully with checkpointer.")
    return app
