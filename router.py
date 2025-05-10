# router.py
import logging
import re
import hashlib
from typing import Dict, Any, Literal, Optional

from models import BotState
from utils import llm_call_helper

logger = logging.getLogger(__name__)

class OpenAPIRouter:
    """
    Routes user input to appropriate nodes in the LangGraph agent.
    Handles initial spec detection and subsequent interactive query classification.
    """
    # Extended intents
    AVAILABLE_INTENTS = Literal[
        "parse_openapi_spec",          # User provides a new spec
        "process_schema_pipeline",     # Internal: After parsing, run full analysis
        "verify_graph",                # Internal or User: Verify current graph
        "_generate_execution_graph",   # Internal or User: Generate/regenerate graph with a goal
        "refine_api_graph",            # User or Internal: Refine current graph
        "describe_graph",              # User: Describe current graph
        "get_graph_json",              # User: Get JSON of current graph
        "answer_openapi_query",        # User: Ask a general question (simple lookup)
        "interactive_query_planner",   # User: Complex query needing multi-step internal processing
        # "interactive_query_executor",# Internal: Executes plan from interactive_query_planner
        "handle_unknown",              # Fallback for unhandled cases
        "handle_loop",                 # Detected routing loop
        "responder"                    # End of turn, send response
    ]

    SPEC_START_REGEX = re.compile(r"^\s*(\"openapi\":|\'openapi\':|openapi:|swagger:|{|-|\binfo:|\bpaths:|\bcomponents:)", re.IGNORECASE | re.MULTILINE)

    # Simple commands for when a schema is loaded
    SCHEMA_LOADED_COMMANDS: Dict[str, AVAILABLE_INTENTS] = {
        "describe graph": "describe_graph", "show graph": "describe_graph",
        "get graph json": "get_graph_json", "show graph json": "get_graph_json",
        "generate new graph for": "_generate_execution_graph", # Requires goal
        "create new plan for": "_generate_execution_graph",   # Requires goal
        "refine graph": "refine_api_graph", "improve plan": "refine_api_graph",
        "verify graph": "verify_graph",
        # "list apis": "answer_openapi_query", # Let LLM classify general questions
        # "what endpoints": "answer_openapi_query",
    }


    def __init__(self, router_llm: Any):
        if not hasattr(router_llm, 'invoke'):
            raise TypeError("router_llm must have an 'invoke' method.")
        self.router_llm = router_llm
        logger.info("OpenAPIRouter initialized.")

    def _is_new_spec(self, state: BotState, user_input: str) -> bool:
        """Heuristically checks if input is a new OpenAPI spec."""
        if not user_input or len(user_input) < 100: # Too short for a typical spec
            return False
        if not self.SPEC_START_REGEX.search(user_input):
            return False
        
        # If a spec is already loaded, check if the new one is different
        if state.openapi_spec_text:
            try:
                current_hash = hashlib.sha256(state.openapi_spec_text.strip().encode('utf-8')).hexdigest()
                new_hash = hashlib.sha256(user_input.strip().encode('utf-8')).hexdigest()
                if current_hash == new_hash:
                    logger.debug("Input looks like the currently loaded spec. Not treating as new.")
                    return False # It's the same spec
            except Exception as e:
                logger.warning(f"Error comparing spec hashes: {e}. Assuming different.", exc_info=True)
        
        # Try a quick parse check
        try:
            if user_input.strip().startswith("{"): # Likely JSON
                json.loads(user_input)
            else: # Likely YAML
                yaml.safe_load(user_input)
            logger.debug("Input is potentially a new, parsable spec.")
            return True
        except (json.JSONDecodeError, yaml.YAMLError):
            logger.debug("Input starts like a spec but failed basic parsing. Not treating as new spec via heuristic.")
            return False # Not easily parsable as JSON/YAML

    def route(self, state: BotState) -> str:
        """Determines the next node based on user input and current state."""
        user_input = state.user_input
        user_input_lower = user_input.lower() if user_input else ""
        previous_intent = state.intent # For loop detection
        
        # Reset flags for the new turn
        state.input_is_spec = False
        state.next_step = None # Should be set by the node this router routes to
        state.response = None # Clear previous intermediate response

        if not user_input:
            logger.warning("Router: Empty input. Routing to handle_unknown.")
            state.intent = "handle_unknown"
            return "handle_unknown"

        # 1. Heuristic: Is it a new OpenAPI spec?
        if self._is_new_spec(state, user_input):
            logger.info("Router: Detected potential new OpenAPI spec. Routing to parse_openapi_spec.")
            state.openapi_spec_string = user_input # Store for parser
            state.input_is_spec = True
            state.intent = "parse_openapi_spec"
            # Clear previous graph/plan related state if a new spec is provided
            state.execution_graph = None
            state.plan_generation_goal = None
            state.graph_refinement_iterations = 0
            state.payload_descriptions = {}
            state.identified_apis = []
            state.schema_summary = None
            return "parse_openapi_spec"

        # 2. If a spec is loaded, check for simple commands first
        determined_intent: Optional[OpenAPIRouter.AVAILABLE_INTENTS] = None
        if state.openapi_schema:
            for command_prefix, intent_val in self.SCHEMA_LOADED_COMMANDS.items():
                if user_input_lower.startswith(command_prefix):
                    logger.info(f"Router: Matched command '{command_prefix}'. Routing to {intent_val}.")
                    determined_intent = intent_val
                    if intent_val == "_generate_execution_graph":
                        # Extract goal from user input for these commands
                        state.plan_generation_goal = user_input[len(command_prefix):].strip()
                        if not state.plan_generation_goal:
                             state.plan_generation_goal = "Generate a relevant workflow based on the command."
                             logger.warning(f"Command '{command_prefix}' matched but no specific goal provided after prefix. Using generic goal.")
                        state.execution_graph = None # Force regeneration
                        state.graph_refinement_iterations = 0
                    elif intent_val == "refine_api_graph":
                        state.graph_regeneration_reason = user_input # Pass full command as refinement instruction
                    break
        
        # 3. If no simple command and spec is loaded, use LLM to classify complex query vs. simple query
        if determined_intent is None and state.openapi_schema:
            logger.debug("Router: No simple command match with loaded schema. Using LLM for intent classification.")
            # Provide more context to the LLM for better routing
            graph_exists_info = "An API execution graph/plan has been generated." if state.execution_graph else "No API execution graph/plan currently exists."
            
            classification_prompt = f"""
            An OpenAPI specification is loaded. {graph_exists_info}
            User input: "{user_input}"

            Classify the user's intent based on the input. Choose ONE of the following:
            - "answer_openapi_query": If the user is asking a straightforward question that can likely be answered by looking up existing information (API summary, API list, current graph description, specific payload examples). Examples: "List all GET APIs.", "What does the 'createUser' API do?", "Describe the current graph."
            - "interactive_query_planner": If the user's request implies a need to *modify context*, *regenerate parts of the plan/payloads with new information*, *set a new goal for the graph*, or requires multi-step reasoning beyond a simple lookup. Examples: "Focus the plan on 'Apple' products.", "What if I want to add a notification step after user creation?", "Regenerate the graph to show how to process a return.", "How would the 'createOrder' payload look for a VIP customer?"
            - "unknown": If the intent is unclear or doesn't fit the above.

            Chosen Classification:
            """
            try:
                llm_response = llm_call_helper(self.router_llm, classification_prompt).strip().lower()
                if "interactive_query_planner" in llm_response:
                    determined_intent = "interactive_query_planner"
                elif "answer_openapi_query" in llm_response:
                    determined_intent = "answer_openapi_query"
                else:
                    determined_intent = "handle_unknown"
                logger.info(f"Router LLM classified intent as: {determined_intent} (raw: '{llm_response}')")
            except Exception as e:
                logger.error(f"Router LLM classification failed: {e}", exc_info=True)
                determined_intent = "handle_unknown"

        # 4. Fallback if no intent determined yet (e.g., no spec loaded, and not a spec input)
        if determined_intent is None:
            logger.debug("Router: No spec loaded or intent unclear. Defaulting to handle_unknown.")
            determined_intent = "handle_unknown"
            if state.openapi_schema is None and len(user_input) > 50: # User might be trying to ask something complex without a spec
                 state.response = "Please provide an OpenAPI specification first so I can assist with that query."


        # Loop detection (simple version)
        final_intent_str = str(determined_intent) # Ensure it's a string
        if final_intent_str == previous_intent and \
           final_intent_str not in ["handle_unknown", "handle_loop", "parse_openapi_spec", "responder"]:
            state.loop_counter += 1
            if state.loop_counter >= 2: # Adjust threshold as needed
                logger.warning(f"Router: Potential loop detected with intent '{final_intent_str}'. Routing to handle_loop.")
                final_intent_str = "handle_loop"
                state.loop_counter = 0 # Reset
        else:
            state.loop_counter = 0

        state.intent = final_intent_str
        logger.info(f"Router final routing decision: {state.intent}")
        return state.intent
