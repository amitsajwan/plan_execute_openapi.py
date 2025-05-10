# filename: router.py
import logging
from typing import Dict, Any, List, Optional, Literal
import re # Import regex for spec detection
import json # Import json for potential spec structure checks
import hashlib # Import hashlib for spec comparison

# Assuming models.py defines BotState
from models import BotState

# Assuming utils.py has llm_call_helper
from utils import llm_call_helper # Removed unused import: parse_llm_json_output_with_model


# Module-level logger
logger = logging.getLogger(__name__)

class OpenAPIRouter:
    """
    Initial router for the LangGraph agent.
    Uses heuristics and an LLM to determine the high-level intent of the user's query
    and route to the appropriate starting node in the graph.
    Handles basic loop detection. Sets the input_is_spec flag in the state.
    This router is for an agent that DOES NOT perform API execution.
    Returns a string indicating the name of the next node.
    """
    # Added 'execute_plan_step' and 'update_graph' to available intents
    AVAILABLE_INTENTS = Literal[
        "parse_openapi_spec", # User provides a spec
        "process_schema_pipeline", # Internal route after parsing
        "verify_graph", # Internal route after graph generation
        "plan_execution", # User asks to plan a workflow description/task
        "execute_plan_step", # Internal route to execute a step in the plan
        "identify_apis", # User asks to identify relevant APIs
        "generate_payloads", # User asks to generate payload descriptions
        "generate_execution_graph", # User asks to generate the graph description
        "describe_graph", # User asks to describe the graph description
        "get_graph_json", # User asks for the graph description JSON
        "answer_openapi_query", # User asks a general question about spec/plan
        "handle_unknown", # Intent could not be determined
        "handle_loop", # Detected potential loop in routing
        "responder", # Router might route directly to responder
        "update_graph", # User asks to modify the graph (add/delete edges) - Placeholder
    ]

    # Regex to quickly check if input looks like JSON or YAML spec start
    # Added common YAML indicators like 'info:', 'paths:', 'components:'
    SPEC_START_REGEX = re.compile(r"^\s*(\"openapi\":|\'openapi\':|openapi:|swagger:|{|-|\binfo:|\bpaths:|\bcomponents:)", re.IGNORECASE)

    # Simple commands to check when schema exists
    # These are prioritized before the LLM classification for efficiency
    SCHEMA_LOADED_COMMANDS: Dict[str, AVAILABLE_INTENTS] = {
        "describe graph": "describe_graph",
        "show graph": "describe_graph",
        "get graph json": "get_graph_json",
        "show graph json": "get_graph_json",
        "generate graph": "generate_execution_graph", # Explicit command even if graph exists
        "generate plan": "plan_execution", # Treat 'generate plan' as a command
        "create plan": "plan_execution", # Treat 'create plan' as a command
        "list apis": "answer_openapi_query", # Treat common queries like commands
        "identify apis": "identify_apis", # Explicit command
        "what endpoints": "answer_openapi_query", # Treat common queries like commands
        "generate payloads": "generate_payloads", # Explicit command
        "describe payloads": "generate_payloads", # Treat 'describe payloads' as generate
        "add edge": "update_graph", # Route to a new node for graph modification (Placeholder)
        "delete edge": "update_graph", # Route to a new node for graph modification (Placeholder)
        "verify graph": "verify_graph", # Explicit command
    }

    def __init__(self, router_llm: Any):
        if not hasattr(router_llm, 'invoke'):
             raise TypeError("router_llm must have an 'invoke' method.")
        self.router_llm = router_llm
        logger.info("OpenAPIRouter initialized (without execution capabilities).")


    def route(self, state: BotState) -> str:
        """
        Determines the next node to execute based on user input and state.
        Returns the string name of the next node.
        Updates state in place with determined intent and potentially other flags.
        """
        user_input = state.user_input
        user_input_lower = user_input.lower() if user_input else ""
        # Use state.intent as the previous intent for loop detection
        previous_intent = state.intent
        loop_counter = state.loop_counter

        determined_intent: Optional[AVAILABLE_INTENTS] = None
        state.input_is_spec = False # Assume input is not a spec initially
        state.next_step = None # Clear next_step from previous turn

        if not user_input:
            logger.warning("Router received empty user input. Routing to handle_unknown.")
            determined_intent = "handle_unknown"
            state.update_scratchpad_reason("router", "Empty input, routing to handle_unknown.")
            state.intent = determined_intent # Update state intent immediately
            return determined_intent # Router returns string


        # --- Heuristic Check for NEW OpenAPI Spec ---
        # Prioritize spec parsing if the input looks like a spec AND it's substantially different from the current spec
        is_potential_spec = user_input and len(user_input) > 150 and self.SPEC_START_REGEX.search(user_input)

        # Simple check if the new potential spec is significantly different from the one already loaded
        # This avoids re-parsing the same spec if the user sends it multiple times
        is_different_spec = True
        if state.openapi_spec_text and user_input:
            # Crude check: compare hashes of non-whitespace content
            try:
                current_hash = hashlib.md5(state.openapi_spec_text.strip().encode('utf-8')).hexdigest()
                new_hash = hashlib.md5(user_input.strip().encode('utf-8')).hexdigest()
                if current_hash == new_hash:
                    is_different_spec = False
                    logger.debug("Input looks like the currently loaded spec. Skipping re-parsing.")
            except Exception as e:
                 logger.warning(f"Error comparing spec hashes: {e}. Assuming different spec.", exc_info=True)


        if is_potential_spec and is_different_spec:
             logger.info("Router heuristic detected potential NEW OpenAPI spec input. Routing to parse_openapi_spec.")
             state.update_scratchpad_reason("router", "Heuristic detected potential new spec. Routing to parse_openapi_spec.")
             determined_intent = "parse_openapi_spec"
             state.input_is_spec = True # Set the flag in state
             # Store the raw spec string in state for the parser node
             state.openapi_spec_string = user_input
             # Skip further checks if it looks like a spec
             state.intent = determined_intent # Update state intent
             return determined_intent # Router returns string


        # --- Handle state if schema is ALREADY loaded and input is NOT a new spec ---
        if determined_intent is None and state.openapi_schema:
            logger.debug("Schema exists and input is not a new spec. Checking specific commands or if input is a query/goal.")
            state.update_scratchpad_reason("router", "Schema exists. Checking for commands/query/goal.")

            # 1. Check for explicit commands first (case-insensitive)
            for command, intent in self.SCHEMA_LOADED_COMMANDS.items():
                # Use simple 'in' check for flexibility, could make this stricter
                if command in user_input_lower:
                    logger.info(f"Router detected specific command '{command}' with schema loaded. Routing to {intent}.")
                    determined_intent = intent
                    state.update_scratchpad_reason("router", f"Specific command '{command}' matched. Routing to {intent}.")
                    # If command implies graph update, capture parameters (basic placeholder)
                    if intent == "update_graph":
                         # This is a placeholder. A real implementation would parse the user input
                         # to extract edge details (from_node, to_node, type: add/delete).
                         # For now, just log and route.
                         logger.warning(f"Graph update command detected: '{user_input}'. Parameter parsing needed.")
                         state.extracted_params = {"command": command, "raw_input": user_input} # Store raw input for update_graph node
                    # If command is plan_execution, store the user input as the goal
                    if intent == "plan_execution":
                         state.plan_execution_goal = user_input # Store user input as the planning goal
                         state.update_scratchpad_reason("router", f"Command '{command}' matched. Setting planning goal: '{user_input}'")

                    break # Stop checking commands once one matches

            # 2. If no command matched, use LLM to classify if it's a general query or a goal
            if determined_intent is None:
                query_classification_prompt = f"""
                An OpenAPI specification is loaded. The user's input is:
                "{user_input}"

                Consider the input. Is it:
                1. A **Question**: Primarily asking for information *about* the loaded OpenAPI specification, the identified APIs, payload descriptions, the described execution graph, or the described plan? (e.g., "What endpoints are available?", "Explain the create user API.", "How does the graph flow?", "What parameters does getUser take?", "Describe the execution plan.")
                2. A **Goal**: Expressing a task or objective that could potentially be achieved by executing a sequence of API calls (even if simulated)? This often involves verbs like "create", "get", "update", "delete", "find", "process", "workflow for". (e.g., "Create a new user", "Get details for user 123", "Update the product quantity", "Workflow to onboard a customer", "Process an order").
                3. An **Unknown**: Something else that doesn't fit the above categories or explicit commands.

                Based ONLY on the user input, classify it as one of: "Question", "Goal", "Unknown".

                Current State Summary (for context only, base decision on user input):
                - OpenAPI spec loaded: Yes
                - Schema Summary: {state.schema_summary[:500] + '...' if state.schema_summary else 'None'}
                - Identified APIs: {'Yes' if state.identified_apis else 'No'} ({len(state.identified_apis) if state.identified_apis else 0} found)
                - Execution graph description exists: {'Yes' if state.execution_graph else 'No'}
                - Current Plan: {state.execution_plan or 'None'} (Step {state.current_plan_step}/{len(state.execution_plan) if state.execution_plan else 0})

                Classification: (Question/Goal/Unknown):
                """
                try:
                    llm_response = llm_call_helper(self.router_llm, query_classification_prompt)
                    classification = llm_response.strip().lower()
                    logger.debug(f"Router LLM query classification result: {classification}")

                    if "question" in classification:
                        logger.info("Router LLM classified input as a Question. Routing to answer_openapi_query.")
                        determined_intent = "answer_openapi_query"
                        state.update_scratchpad_reason("router", "LLM classified as Question -> answer_openapi_query.")
                    elif "goal" in classification:
                        logger.info("Router LLM classified input as a Goal. Routing to plan_execution.")
                        determined_intent = "plan_execution"
                        state.update_scratchpad_reason("router", "LLM classified as Goal -> plan_execution.")
                        state.plan_execution_goal = user_input # Store user input as the planning goal
                        state.update_scratchpad_reason("router", f"Setting planning goal based on LLM classification: '{user_input}'")
                    else:
                        logger.debug("Router LLM classified input as Unknown. Routing to handle_unknown.")
                        determined_intent = "handle_unknown"
                        state.update_scratchpad_reason("router", "LLM classified as Unknown -> handle_unknown.")

                except Exception as e:
                    logger.error(f"Error calling Router LLM for query classification: {e}", exc_info=True)
                    logger.warning("Router LLM query classification failed. Defaulting to handle_unknown.")
                    state.update_scratchpad_reason("router", f"LLM query classification failed: {e}. Defaulted to handle_unknown.")
                    determined_intent = "handle_unknown" # Default on error

        # --- General Intent Determination (fallback if still None - less likely with classification) ---
        # This block might be less necessary with explicit commands and classification,
        # but kept as a final safeguard.
        if determined_intent is None:
            logger.debug("Determining intent using general LLM prompt (fallback).")
            state.update_scratchpad_reason("router", "Using general LLM prompt for intent (fallback).")

            # Construct a more careful prompt, especially when schema is loaded
            schema_loaded_context = "No OpenAPI spec is currently loaded."
            if state.openapi_schema:
                schema_loaded_context = f"""An OpenAPI spec IS currently loaded.
                - Schema Summary (first 500 chars): {state.schema_summary[:500] + '...' if state.schema_summary else 'None'}
                - Identified APIs: {'Yes' if state.identified_apis else 'No'} ({len(state.identified_apis) if state.identified_apis else 0} found)
                - Graph description exists: {'Yes' if state.execution_graph else 'No'}
                - Current Plan: {state.execution_plan or 'None'} (Step {state.current_plan_step}/{len(state.execution_plan) if state.execution_plan else 0})
                IMPORTANT: Since a spec is loaded, DO NOT choose 'parse_openapi_spec' unless the input explicitly contains a new spec. Prioritize 'answer_openapi_query' for questions and 'plan_execution' for goals/tasks. If the input is a command not handled above (like 'generate payloads' or 'add edge'), choose the corresponding action from {list(self.AVAILABLE_INTENTS)}. If unsure, choose 'handle_unknown'."""

            prompt = f"""
            Determine the user's high-level intent from the list: {list(self.AVAILABLE_INTENTS)}.

            User Input: "{user_input}"

            Current State Context:
            {schema_loaded_context}
            - Previous Router Intent: {previous_intent or 'None'}

            Instructions:
            1. Analyze the User Input.
            2. Consider the Current State Context. If a spec is loaded, be very careful about choosing `parse_openapi_spec`. Prefer `answer_openapi_query` for informational requests about the loaded spec/artifacts and `plan_execution` for tasks/goals. If the input is a command (like 'generate graph', 'add edge'), choose the corresponding intent from the list.
            3. Choose the *single best matching intent* from the list: {list(self.AVAILABLE_INTENTS)}.
            4. If the intent is unclear or doesn't fit well, choose `handle_unknown`.
            5. Output ONLY the chosen intent string.

            Chosen Intent:
            """
            try:
                llm_response = llm_call_helper(self.router_llm, prompt)
                determined_intent_llm = llm_response.strip().lower() # Ensure lowercase

                # Validation against AVAILABLE_INTENTS Literal
                if determined_intent_llm not in self.AVAILABLE_INTENTS.__args__:
                    logger.warning(f"Router LLM returned invalid intent '{determined_intent_llm}'. Defaulting to handle_unknown.")
                    determined_intent = "handle_unknown"
                    state.update_scratchpad_reason("router", f"General LLM returned invalid intent '{llm_response.strip()}'. Defaulted to handle_unknown.")
                 # Prevent LLM from choosing parse_openapi_spec if schema exists (heuristic should catch specs)
                elif determined_intent_llm == "parse_openapi_spec" and state.openapi_schema:
                    # This case should ideally be caught by the heuristic, but as a safeguard
                     logger.warning(f"Router LLM chose '{determined_intent_llm}' when schema already exists and input wasn't heuristically a new spec. Overriding to handle_unknown.")
                     determined_intent = "handle_unknown"
                     state.update_scratchpad_reason("router", f"General LLM chose '{determined_intent_llm}' with existing schema. Defaulted to handle_unknown.")
                else:
                    determined_intent = determined_intent_llm
                    logger.debug(f"Router LLM determined general intent: {determined_intent}")
                    state.update_scratchpad_reason("router", f"General LLM determined intent: '{determined_intent}'.")
                    # If the fallback LLM chose plan_execution, set the goal
                    if determined_intent == "plan_execution":
                         state.plan_execution_goal = user_input
                         state.update_scratchpad_reason("router", f"Setting planning goal based on fallback LLM: '{user_input}'")


            except Exception as e:
                logger.error(f"Error calling Router LLM for general intent: {e}", exc_info=True)
                determined_intent = "handle_unknown"
                state.update_scratchpad_reason("router", f"General LLM call failed: {e}. Defaulted to handle_unknown.")

        # --- Final Fallback ---
        if determined_intent is None:
             logger.error("Router failed to determine intent after all checks. Defaulting to handle_unknown.")
             determined_intent = "handle_unknown"
             state.update_scratchpad_reason("router", "Failed to determine intent after all checks. Defaulted to handle_unknown.")

        # --- Apply Loop Detection ---
        final_intent = determined_intent
        # Check if the same valid, non-final intent is repeating
        # Also check if the user input has changed significantly, as repeated input should not trigger loop detection
        input_changed = (state.user_input != state.scratchpad.get("last_router_input"))
        state.scratchpad["last_router_input"] = state.user_input # Store current input

        # Loop detection applies if the determined intent is the same as the previous
        # AND it's not a terminal state (responder, handle_unknown, handle_loop)
        # AND it's not the start of a new spec parse
        # AND the input hasn't changed (to allow user to re-trigger if needed)
        if determined_intent == previous_intent and \
           determined_intent not in ["handle_unknown", "handle_loop", "parse_openapi_spec", "responder"] and \
           not input_changed:
             loop_counter += 1
             state.loop_counter = loop_counter # Update state in place
             logger.warning(f"Router detected repeated intent: {determined_intent}. Loop counter: {loop_counter}")
             if loop_counter >= 3: # Threshold for loop detection
                 logger.error(f"Router detected potential loop. Routing to handle_loop.")
                 final_intent = "handle_loop"
                 state.loop_counter = 0 # Reset counter on entering loop handler
             # else: proceed with determined_intent, counter updated
        else:
            # Reset loop counter if intent changes, input changes, or is a final/parse state
            state.loop_counter = 0 # Reset counter

        # --- Set Final State Fields ---
        state.intent = final_intent # Store the determined intent in state

        logger.info(f"Router routing to: {final_intent}")
        # Router returns the string name of the next node
        return final_intent
