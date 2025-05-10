# filename: graph.py
import logging
from typing import Any, Dict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # Assuming MemorySaver is used

# Ensure INFO-level logs are visible
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import state model and core logic
from models import BotState # Import BotState
from core_logic import OpenAPICoreLogic # Import core logic class
from router import OpenAPIRouter # Import router class

# --- Responder Node Function ---
# This node should modify state in place and return it, like others
def finalize_response(state: BotState) -> BotState:
    """
    Sets the final_response based on state.response and clears intermediate response.
    Returns the updated state.
    """
    tool_name = "responder"
    state.update_scratchpad_reason(tool_name, "Finalizing response for user.")
    logger.debug("Executing responder node (finalize_response).")

    if state.response:
        state.final_response = state.response
    elif not state.final_response: # Only use default if no final_response was set previously
        state.final_response = (
            "I've completed the requested action, but there wasn't a specific message to display."
        )
        logger.warning("Responder: No intermediate 'response' found; using default final_response.")

    state.response = None # Clear intermediate response
    state.next_step = None # Clear the next_step state field
    state.update_scratchpad_reason(tool_name, "Final response set.")
    return state


# --- Graph Definition ---
def build_graph(router_llm: Any, worker_llm: Any) -> StateGraph:
    """
    Builds and compiles the LangGraph StateGraph for the OpenAPI agent.
    Configured for nodes that modify state in place and return BotState,
    with the next node name specified in state.next_step (for most nodes)
    or returned as a string from the router.
    """
    logger.info("Building LangGraph graph.py...")

    core_logic = OpenAPICoreLogic(worker_llm=worker_llm)
    router = OpenAPIRouter(router_llm=router_llm)

    # StateGraph expects nodes to return the state object they modified
    # The router is an exception as it explicitly returns the next node name string
    builder = StateGraph(BotState)

    # Add router node - its output determines the transition
    builder.add_node("router", router.route) # Router returns string

    # Add other tool nodes - they modify state and return it
    builder.add_node("parse_openapi_spec", core_logic.parse_openapi_spec)
    builder.add_node("process_schema_pipeline", core_logic.process_schema_pipeline) # Pipeline node
    builder.add_node("verify_graph", core_logic.verify_graph) # Verification node
    # Note: _identify_apis_from_schema and _generate_payload_descriptions are called internally by the pipeline,
    # they are not standalone nodes in this graph structure.
    builder.add_node("generate_execution_graph", core_logic._generate_execution_graph) # Graph generation node
    builder.add_node("describe_graph", core_logic.describe_graph)
    builder.add_node("get_graph_json", core_logic.get_graph_json)
    builder.add_node("answer_openapi_query", core_logic.answer_openapi_query)
    builder.add_node("handle_unknown", core_logic.handle_unknown)
    builder.add_node("handle_loop", core_logic.handle_loop)
    builder.add_node("update_graph", core_logic.update_graph) # Placeholder
    builder.add_node("plan_execution", core_logic.plan_execution) # Plan node
    builder.add_node("execute_plan_step", core_logic.execute_plan_step) # Execute node
    builder.add_node("responder", finalize_response) # Responder modifies and returns state

    # Entry point: Router
    builder.add_edge(START, "router")

    # Conditional routing FROM router - router returns a string (the name of the next node)
    # The keys in this dictionary must match the string outputs of the router.route method
    builder.add_conditional_edges(
        "router",
        # The router's output (string) is the next node key
        lambda x: x, # The router function directly returns the next node name string
        {
            "parse_openapi_spec": "parse_openapi_spec",
            "process_schema_pipeline": "process_schema_pipeline", # Router can route directly to pipeline if schema is cached but artifacts missing
            "verify_graph": "verify_graph",
            "plan_execution": "plan_execution", # Router routes to plan_execution for goals
            "execute_plan_step": "execute_plan_step", # Router could potentially route here if a plan is already active? (Less likely)
            "identify_apis": "identify_apis", # Router can route here for explicit command
            "generate_payloads": "generate_payloads", # Router can route here for explicit command
            "generate_execution_graph": "generate_execution_graph", # Router can route here for explicit command
            "describe_graph": "describe_graph",
            "get_graph_json": "get_graph_json",
            "answer_openapi_query": "answer_openapi_query",
            "handle_unknown": "handle_unknown",
            "handle_loop": "handle_loop",
            "responder": "responder", # Router can route directly to responder (e.g., empty input handled early)
            "update_graph": "update_graph", # Router routes here for graph update commands
        }
    )

    # Conditional routing FROM tool nodes (except router and responder)
    # These nodes modify state and return it. The transition depends on state.next_step.
    # If state.next_step is not set by the node, it defaults to 'responder'.
    # We need to explicitly map all possible state.next_step values each node might set.

    # Define a function to read state.next_step for routing
    def route_from_state(state: BotState) -> str:
        # Read the next_step from the state. If not set, default to 'responder'.
        nxt = state.next_step if state.next_step is not None else "responder"
        logger.debug(f"Routing from state.next_step: {nxt}")
        # Note: LangGraph's conditional edges mechanism reads this value
        # and handles the transition. We don't clear state.next_step here;
        # the nodes themselves should manage setting it for the *next* desired step.
        return nxt


    # Define the possible exit points from *any* tool node (except router and responder)
    # This mapping needs to cover all possible values a node can set state.next_step to.
    # Based on core_logic, nodes can set next_step to:
    # - "process_schema_pipeline" (from parse_openapi_spec after cache hit/parse success)
    # - "verify_graph" (from process_schema_pipeline or generate_execution_graph on success)
    # - "generate_execution_graph" (from verify_graph on failure, or itself on LLM format failure)
    # - "describe_graph" (from verify_graph on success if triggered by user, or after graph gen)
    # - "handle_unknown" (from various failures or loop detection)
    # - "handle_loop" (from router loop detection)
    # - "responder" (default or explicit end of a flow)
    # - "update_graph" (from update_graph itself if it needs more steps, or error)
    # - "plan_execution" (from router for goals)
    # - "execute_plan_step" (from plan_execution to start execution, or from execute_plan_step to continue)
    # - It might loop back to itself for retries (e.g. generate_execution_graph -> generate_execution_graph)
    # - Also need routes back to 'router' if a node finishes but needs the router to determine the *next* user input's intent

    # Let's list all possible nodes a tool node might route to
    all_possible_next_nodes = [
        "router", # Needed if a node finishes its task but the next step depends on the user's *new* input
        "parse_openapi_spec",
        "process_schema_pipeline",
        "verify_graph",
        "plan_execution",
        "execute_plan_step",
        "identify_apis",
        "generate_payloads",
        "generate_execution_graph",
        "describe_graph",
        "get_graph_json",
        "answer_openapi_query",
        "handle_unknown",
        "handle_loop",
        "responder",
        "update_graph",
    ]

    # Create the mapping dictionary
    common_routes: Dict[str, str] = {node_name: node_name for node_name in all_possible_next_nodes}


    # Apply conditional edges based on state.next_step for nodes that return state
    # This applies to all nodes EXCEPT the router (which returns string) and the responder (which goes to END)
    nodes_returning_state = [
        "parse_openapi_spec",
        "process_schema_pipeline",
        "verify_graph",
        "identify_apis", # Although called internally by pipeline, keep if ever used standalone
        "generate_payloads", # Although called internally by pipeline, keep if ever used standalone
        "generate_execution_graph",
        "describe_graph",
        "get_graph_json",
        "answer_openapi_query",
        "handle_unknown",
        "handle_loop",
        "update_graph",
        "plan_execution",
        "execute_plan_step",
    ]

    for node in nodes_returning_state:
         builder.add_conditional_edges(
             node,
             route_from_state, # Use the function that reads state.next_step
             common_routes # Map possible next_step values to nodes
         )


    # Edge from responder to END - This is the final state
    builder.add_edge("responder", END)

    app = builder.compile()
    logger.info("Graph compiled successfully.")
    return app
