# graph.py
import logging
from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver # Import base for type hinting

from models import BotState
from core_logic import OpenAPICoreLogic
from router import OpenAPIRouter

logger = logging.getLogger(__name__)

def finalize_response(state: BotState) -> BotState:
    """Sets final_response, clears intermediate response, and prepares for END."""
    tool_name = "responder"
    state.update_scratchpad_reason(tool_name, "Finalizing response for user.")
    logger.debug("Executing responder node (finalize_response).")

    if state.response:
        state.final_response = state.response
    elif not state.final_response:
        state.final_response = "Processing complete. How can I help you further?"
        logger.warning("Responder: No intermediate 'response' or existing 'final_response'. Using default.")
    
    state.response = None
    state.next_step = None 
    state.update_scratchpad_reason(tool_name, f"Final response set: {state.final_response[:100]}...")
    return state

def build_graph(router_llm: Any, worker_llm: Any, checkpointer: BaseCheckpointSaver) -> StateGraph: # Accept checkpointer
    """Builds and compiles the LangGraph StateGraph for the OpenAPI agent."""
    logger.info("Building LangGraph graph...")

    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router_instance = OpenAPIRouter(router_llm=router_llm)

    builder = StateGraph(BotState)

    # --- Add Nodes ---
    builder.add_node("router", router_instance.route)

    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("process_schema_pipeline", core_logic.process_schema_pipeline)
    builder.add_node("_generate_execution_graph", core_logic._generate_execution_graph)
    builder.add_node("verify_graph", core_logic.verify_graph)
    builder.add_node("refine_api_graph", core_logic.refine_api_graph)
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("get_graph_json", core_logic.get_graph_json)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    
    builder.add_node("interactive_query_planner", core_logic.interactive_query_planner)
    builder.add_node("interactive_query_executor", core_logic.interactive_query_executor)

    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("responder", finalize_response)

    # --- Define Edges ---
    builder.add_edge(START, "router")

    router_conditional_edges: Dict[str, str] = {
        intent_val: intent_val for intent_val in OpenAPIRouter.AVAILABLE_INTENTS.__args__ # type: ignore
    }
    router_conditional_edges.pop("interactive_query_executor", None) 

    builder.add_conditional_edges(
        "router",
        lambda state: state.intent,
        router_conditional_edges
    )

    def route_from_internal_node_state(state: BotState) -> str:
        next_node_name = state.next_step
        if not next_node_name:
            logger.warning(f"Node {state.intent or 'Unknown previous'} did not set state.next_step. Defaulting to responder.")
            return "responder"
        logger.debug(f"Routing from internal node. state.next_step is '{next_node_name}'")
        return next_node_name

    all_internal_target_nodes = {
        "process_schema_pipeline", "_generate_execution_graph", "verify_graph",
        "refine_api_graph", "describe_graph", "get_graph_json",
        "answer_openapi_query", "interactive_query_planner", "interactive_query_executor",
        "handle_unknown", "handle_loop", "responder", "router",
        "parse_openapi_spec" 
    }

    internal_node_routes: Dict[str, str] = {
        node_name: node_name for node_name in all_internal_target_nodes
    }
    # It's good practice to have a default fallback in conditional edges mapping
    # if the key from state.next_step might not exist in the map.
    # However, LangGraph's add_conditional_edges expects all possible keys returned by the callable
    # to be present in the path_map. If a key is returned that's not in path_map, it will error.
    # So, ensuring all nodes correctly set state.next_step to a valid target is crucial.
    # The 'default_fallback_responder' isn't used by LangGraph's mechanism directly unless the callable returns it.

    nodes_setting_next_step = [
        "parse_openapi_spec", "process_schema_pipeline", "_generate_execution_graph",
        "verify_graph", "refine_api_graph", "describe_graph", "get_graph_json",
        "answer_openapi_query", "interactive_query_planner", "interactive_query_executor",
        "handle_unknown", "handle_loop"
    ]

    for node_name in nodes_setting_next_step:
        builder.add_conditional_edges(
            node_name,
            route_from_internal_node_state,
            internal_node_routes
        )

    builder.add_edge("responder", END)

    # Compile the graph WITH the checkpointer here
    app = builder.compile(checkpointer=checkpointer)
    logger.info("LangGraph graph compiled successfully with checkpointer.")
    return app
