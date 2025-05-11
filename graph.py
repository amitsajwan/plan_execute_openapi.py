# graph.py
import logging
from typing import Any, Dict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver # For type hinting checkpointer

# Assuming models.py is in the same directory or accessible in PYTHONPATH
from models import BotState
# Assuming core_logic.py and router.py are accessible
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

# For type hinting the api_executor_instance parameter in build_graph
# Assuming APIExecutor is defined in workflow_executor.py as per recent updates
try:
    from workflow_executor import APIExecutor
except ImportError:
    logging.warning("graph.py: Could not import APIExecutor from workflow_executor.py for type hinting. Using Any.")
    APIExecutor = Any # Fallback type


logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """
    Sets final_response from state.response if available, clears intermediate response,
    and prepares the state for ending the current graph turn.
    """
    tool_name = "responder"
    # Log entry point with current response values for debugging
    logger.info(f"Responder ({tool_name}): Entered. state.response='{str(state.response)[:100]}...', state.final_response='{str(state.final_response)[:100]}...'")
    state.update_scratchpad_reason(tool_name, f"Finalizing response. Initial state.response: '{str(state.response)[:100]}...'")
    
    if state.response: # If an intermediate response was set by the last node
        state.final_response = state.response
        logger.info(f"Responder ({tool_name}): Set final_response from state.response: '{str(state.final_response)[:200]}...'")
    elif not state.final_response: # Only set a default if final_response isn't already set (e.g., by an error message directly in BotState)
        state.final_response = "Processing complete. How can I help you further?"
        logger.warning(f"Responder ({tool_name}): state.response was empty/None. Using default final_response: '{state.final_response}'")
    else:
        # This case means state.response was falsey, but state.final_response already had a value.
        logger.info(f"Responder ({tool_name}): state.response was falsey, but final_response was already set. No change to final_response: '{str(state.final_response)[:200]}...'")

    # Clear fields for the next turn to avoid carry-over issues
    state.response = None         # Clear intermediate response
    state.next_step = None      # Clear routing directive from the previous node
    state.intent = None         # Clear current intent, will be re-evaluated by router on new input
    # state.user_input is typically updated by the main loop receiving new input.
    # Scratchpad items like 'graph_to_send' are managed by the nodes that set them.

    state.update_scratchpad_reason(tool_name, f"Final response set in state: {str(state.final_response)[:200]}...")
    logger.info(f"Responder ({tool_name}): Exiting. state.final_response='{str(state.final_response)[:100]}...', state.response='{state.response}'")
    return state


def build_graph(
    router_llm: Any,
    worker_llm: Any,
    api_executor_instance: APIExecutor, # Modified: Added api_executor_instance
    checkpointer: BaseCheckpointSaver
) -> StateGraph:
    """Builds and compiles the LangGraph StateGraph for the OpenAPI agent."""
    logger.info("Building LangGraph graph with APIExecutor integration...")

    # Instantiate core logic, passing the api_executor_instance
    core_logic = OpenAPICoreLogic(worker_llm=worker_llm, api_executor_instance=api_executor_instance)
    router_instance = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # --- Add All Nodes ---
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
    
    # Interactive processing nodes (these call other core_logic methods internally)
    builder.add_node("interactive_query_planner", core_logic.interactive_query_planner)
    builder.add_node("interactive_query_executor", core_logic.interactive_query_executor)

    # New node for setting up workflow execution
    builder.add_node("setup_workflow_execution", core_logic.setup_workflow_execution)
    # Note: resume_workflow_with_payload is an internal action called by interactive_query_executor,
    # not a direct node from the main router.

    # Handling nodes
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    
    # Responder node (finalizes output for the user)
    builder.add_node("responder", finalize_response)

    # --- Define Edges ---
    # Entry point of the graph
    builder.add_edge(START, "router")

    # Conditional edges from the router based on determined intent
    # OpenAPIRouter.AVAILABLE_INTENTS should list all valid node names the router can target.
    router_targetable_intents = {
        intent_val: intent_val for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
        if intent_val in builder.nodes # Ensure the intent name corresponds to an added node
    }
    # Log if any intents defined in router are not actual graph nodes
    for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__: # type: ignore
        if intent_val not in router_targetable_intents:
            logger.warning(f"Router intent '{intent_val}' is defined in AVAILABLE_INTENTS but not added as a node in the graph. It cannot be routed to directly by the router.")

    builder.add_conditional_edges(
        "router", # Source node
        lambda state: state.intent, # Function that returns the key for the conditional edge
        router_targetable_intents # Mapping from intent key to target node name
    )

    # Define how nodes that set 'state.next_step' should be routed
    def route_from_internal_node_state(state: BotState) -> str:
        """Determines the next node based on state.next_step set by an internal node."""
        next_node_name = state.next_step
        current_node_info = state.intent or "Unknown (routing from internal node)" # Use intent as proxy for current node if available

        if not next_node_name:
            logger.warning(f"Node '{current_node_info}' did not set state.next_step. Defaulting to 'responder'.")
            return "responder"
        
        if next_node_name not in builder.nodes:
            logger.error(f"Node '{current_node_info}' tried to route to non-existent node '{next_node_name}'. Defaulting to 'handle_unknown'.")
            return "handle_unknown" # Or 'responder'
            
        logger.debug(f"Routing from internal node '{current_node_info}'. Next step decided by node: '{next_node_name}'")
        return next_node_name

    # List all nodes that are expected to set 'state.next_step' to guide their own routing
    # These are nodes that perform an action and then decide where to go next.
    nodes_that_set_next_step = [
        "parse_openapi_spec", "process_schema_pipeline", "_generate_execution_graph",
        "verify_graph", "refine_api_graph", "describe_graph", "get_graph_json",
        "answer_openapi_query", "interactive_query_planner", "interactive_query_executor",
        "setup_workflow_execution", # This node will set next_step = "responder"
        "handle_unknown", "handle_loop"
        # 'router' is handled by its own conditional_edges based on 'state.intent'
        # 'responder' always goes to END
    ]

    # Create a mapping for all possible target nodes from these internal nodes.
    # This ensures that any node name set in state.next_step can be routed to if it exists.
    all_graph_nodes_as_targets: Dict[str, str] = {
        node_name: node_name for node_name in builder.nodes # builder.nodes gives all added node names
    }
    # Ensure critical fallback nodes are in the target map, though builder.nodes should include them.
    if "responder" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["responder"] = "responder"
    if "handle_unknown" not in all_graph_nodes_as_targets: all_graph_nodes_as_targets["handle_unknown"] = "handle_unknown"


    for source_node_name in nodes_that_set_next_step:
        if source_node_name in builder.nodes: # Ensure the source node itself was added
            builder.add_conditional_edges(
                source_node_name,
                route_from_internal_node_state, # Function to get the next node key
                all_graph_nodes_as_targets # Mapping of possible next_step values to target nodes
            )
        else:
            # This would be a configuration error in the list 'nodes_that_set_next_step'
            logger.error(f"Configuration error: Node '{source_node_name}' listed in 'nodes_that_set_next_step' was not added to the graph builder.")

    # Terminal edge: all paths eventually lead to the responder, which then goes to END
    builder.add_edge("responder", END)

    # Compile the graph
    try:
        app = builder.compile(checkpointer=checkpointer)
        logger.info("LangGraph graph compiled successfully with checkpointer.")
        return app
    except Exception as e:
        logger.critical(f"LangGraph compilation failed: {e}", exc_info=True)
        raise # Re-raise the exception to prevent app startup with a broken graph
