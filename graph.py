# graph.py
import logging
from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, START, END
# Assuming MemorySaver or another checkpointer will be configured in main.py
# from langgraph.checkpoint.memory import MemorySaver

from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """Sets final_response, clears intermediate response, and prepares for END."""
    tool_name = "responder"
    state.update_scratchpad_reason(tool_name, "Finalizing response for user.")
    logger.debug("Executing responder node (finalize_response).")

    if state.response: # If an intermediate response was set by the last node
        state.final_response = state.response
    elif not state.final_response: # If no node set a response, and final_response is also empty
        state.final_response = "Processing complete. How can I help you further?"
        logger.warning("Responder: No intermediate 'response' or existing 'final_response'. Using default.")
    
    # Clear intermediate fields for the next turn
    state.response = None
    state.next_step = None 
    # state.user_input = None # Keep user_input for logging/context if needed, or clear
    # state.intent = None # Keep intent for logging, or clear
    state.update_scratchpad_reason(tool_name, f"Final response set: {state.final_response[:100]}...")
    return state

def build_graph(router_llm: Any, worker_llm: Any) -> StateGraph:
    """Builds and compiles the LangGraph StateGraph for the OpenAPI agent."""
    logger.info("Building LangGraph graph...")

    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router_instance = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # --- Add Nodes ---
    builder.add_node("router", router_instance.route) # Router returns string for next node

    # Core logic nodes (these modify state and return it)
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("process_schema_pipeline", core_logic.process_schema_pipeline)
    builder.add_node("_generate_execution_graph", core_logic._generate_execution_graph) # Internal, but router can target
    builder.add_node("verify_graph", core_logic.verify_graph)
    builder.add_node("refine_api_graph", core_logic.refine_api_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("get_graph_json", core_logic.get_graph_json)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    
    # Interactive phase nodes
    builder.add_node("interactive_query_planner", core_logic.interactive_query_planner)
    builder.add_node("interactive_query_executor", core_logic.interactive_query_executor)

    # Utility nodes
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("responder", finalize_response) # Final step before END

    # --- Define Edges ---
    builder.add_edge(START, "router")

    # Conditional routing FROM router (router.route returns the next node name string)
    # Keys here must match the string outputs of OpenAPIRouter.AVAILABLE_INTENTS
    router_conditional_edges: Dict[str, str] = {
        intent_val: intent_val for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
    }
    # Ensure all router outputs map to a valid node name
    # router_conditional_edges["interactive_query_executor"] is not directly routed by router, so remove if present
    router_conditional_edges.pop("interactive_query_executor", None) 


    builder.add_conditional_edges(
        "router",
        lambda state: state.intent, # Router sets state.intent with the next node name
        router_conditional_edges
    )

    # Conditional routing FROM other nodes (based on state.next_step)
    def route_from_internal_node_state(state: BotState) -> str:
        next_node_name = state.next_step
        if not next_node_name:
            logger.warning(f"Node {state.intent or 'Unknown previous'} did not set state.next_step. Defaulting to responder.")
            return "responder" # Default if a node forgets to set next_step
        logger.debug(f"Routing from internal node. state.next_step is '{next_node_name}'")
        return next_node_name

    # All possible nodes that an internal logic node might route to
    # This needs to cover all values that `state.next_step` might be set to by core_logic methods
    all_internal_target_nodes = {
        "process_schema_pipeline", "_generate_execution_graph", "verify_graph",
        "refine_api_graph", "describe_graph", "get_graph_json",
        "answer_openapi_query", "interactive_query_planner", "interactive_query_executor",
        "handle_unknown", "handle_loop", "responder", "router" # Allow routing back to router if needed
    }
    # Ensure parse_openapi_spec can also be a target if, for example, a reset is triggered
    all_internal_target_nodes.add("parse_openapi_spec")


    internal_node_routes: Dict[str, str] = {
        node_name: node_name for node_name in all_internal_target_nodes
    }
    # Add a fallback for safety, though ideally all next_steps are covered
    internal_node_routes["default_fallback_responder"] = "responder"


    # Nodes that set state.next_step for routing
    nodes_setting_next_step = [
        "parse_openapi_spec", "process_schema_pipeline", "_generate_execution_graph",
        "verify_graph", "refine_api_graph", "describe_graph", "get_graph_json",
        "answer_openapi_query", "interactive_query_planner", "interactive_query_executor",
        "handle_unknown", "handle_loop"
        # Responder goes to END, Router has its own conditional edge logic
    ]

    for node_name in nodes_setting_next_step:
        builder.add_conditional_edges(
            node_name,
            route_from_internal_node_state,
            internal_node_routes # Maps state.next_step value to the actual node name
        )

    # Final edge to END
    builder.add_edge("responder", END)

    # Compile the graph (checkpointer is added in main.py)
    app = builder.compile()
    logger.info("LangGraph graph compiled successfully.")
    return app
