# core_logic.py
import json
import logging
import asyncio # Added for asyncio.create_task
from typing import Any, Dict, List, Optional
import yaml

# Assuming models.py is in the same directory or accessible in PYTHONPATH
from models import BotState, GraphOutput, Node, InputMapping # OutputMapping is also in models.py now
from utils import (
    llm_call_helper, load_cached_schema, save_schema_to_cache,
    get_cache_key, check_for_cycles, parse_llm_json_output_with_model,
    SCHEMA_CACHE
)
from pydantic import ValidationError

# Assuming workflow_executor.py is in the same directory or accessible
# It should contain WorkflowExecutor and the (placeholder or real) APIExecutor
try:
    from workflow_executor import WorkflowExecutor, APIExecutor
except ImportError:
    logging.critical("Failed to import WorkflowExecutor or APIExecutor. Ensure workflow_executor.py is correct and accessible.")
    # Define placeholders if import fails to allow linting/partial functionality
    class APIExecutor: pass
    class WorkflowExecutor: pass


logger = logging.getLogger(__name__)

class OpenAPICoreLogic:
    def __init__(self, worker_llm: Any, api_executor_instance: APIExecutor): # Added api_executor_instance
        if not hasattr(worker_llm, 'invoke'):
            raise TypeError("worker_llm must have an 'invoke' method.")
        if not isinstance(api_executor_instance, APIExecutor):
            # This check might be too simple if APIExecutor is just a placeholder class
            # For a real implementation, you might check for specific methods.
            logger.warning("api_executor_instance does not seem to be a valid APIExecutor. Workflow execution might fail.")

        self.worker_llm = worker_llm
        self.api_executor = api_executor_instance # Store the APIExecutor instance
        logger.info("OpenAPICoreLogic initialized with worker_llm and api_executor.")

    def parse_openapi_spec(self, state: BotState) -> BotState:
        tool_name = "parse_openapi_spec"
        state.response = "Parsing OpenAPI specification..."
        state.update_scratchpad_reason(tool_name, "Attempting to parse OpenAPI spec.")

        spec_text = state.openapi_spec_string
        if not spec_text:
            state.response = "No OpenAPI specification text provided."
            state.update_scratchpad_reason(tool_name, "No spec text in state.")
            state.next_step = "responder"
            state.openapi_spec_string = None # Clear the raw spec string
            return state

        cache_key = get_cache_key(spec_text)
        # Try loading full state from cache if available (schema + derived artifacts)
        # This part would need more sophisticated caching if we cache BotState subsections
        cached_schema_artifacts = load_cached_schema(f"{cache_key}_full_analysis")

        if cached_schema_artifacts and isinstance(cached_schema_artifacts, dict):
            try:
                # Attempt to rehydrate parts of the state from cache
                # This is a simplified example; robust rehydration needs care
                state.openapi_schema = cached_schema_artifacts.get('openapi_schema')
                state.schema_summary = cached_schema_artifacts.get('schema_summary')
                state.identified_apis = cached_schema_artifacts.get('identified_apis', [])
                state.payload_descriptions = cached_schema_artifacts.get('payload_descriptions', {})
                graph_dict = cached_schema_artifacts.get('execution_graph')
                if graph_dict:
                    state.execution_graph = GraphOutput.model_validate(graph_dict)
                
                state.schema_cache_key = cache_key # Store the base key
                state.openapi_spec_text = spec_text # Store the original spec text
                state.openapi_spec_string = None # Clear the raw spec string

                logger.info(f"Loaded full analysis from cache for key: {cache_key}")
                state.response = "OpenAPI specification and derived analysis loaded from cache."
                if state.execution_graph:
                     state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                state.next_step = "responder"
                return state
            except Exception as e:
                logger.warning(f"Error rehydrating state from cached full analysis: {e}. Proceeding with parsing.")
                # Clear potentially partially loaded state
                state.openapi_schema = None
                state.schema_summary = None # etc.


        # Fallback to parsing if full analysis not cached or rehydration failed
        cached_schema = load_cached_schema(cache_key)
        if cached_schema:
            logger.info(f"Loaded schema (only) from cache for key: {cache_key}")
            state.openapi_schema = cached_schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text
            state.openapi_spec_string = None # Clear the raw spec string
            state.response = "OpenAPI specification (schema only) loaded from cache. Starting analysis pipeline..."
            state.next_step = "process_schema_pipeline" # Needs full pipeline run
            return state

        parsed_schema = None
        error_message = None
        try:
            # Try JSON first
            parsed_schema = json.loads(spec_text)
        except json.JSONDecodeError:
            try:
                # Try YAML if JSON fails
                parsed_schema = yaml.safe_load(spec_text)
            except yaml.YAMLError as yaml_e:
                error_message = f"YAML parsing failed: {yaml_e}"
            except Exception as e_yaml: # Catch any other unexpected error during YAML parsing
                error_message = f"Unexpected error during YAML parsing: {e_yaml}"
        except Exception as e_json: # Catch any other unexpected error during JSON parsing
            error_message = f"Unexpected error during JSON parsing: {e_json}"


        if parsed_schema and isinstance(parsed_schema, dict) and \
           ('openapi' in parsed_schema or 'swagger' in parsed_schema) and 'info' in parsed_schema:
            state.openapi_schema = parsed_schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text # Store the successfully parsed spec text
            state.openapi_spec_string = None # Clear the raw spec string from user input field

            if SCHEMA_CACHE:
                save_schema_to_cache(cache_key, parsed_schema) # Cache only the schema here
            else:
                logger.warning("SCHEMA_CACHE is None, schema not saved to disk cache.")

            logger.info("Successfully parsed OpenAPI spec.")
            state.response = "OpenAPI specification parsed. Starting analysis pipeline..."
            state.next_step = "process_schema_pipeline"
        else:
            state.openapi_schema = None
            state.openapi_spec_string = None # Clear the raw spec string
            final_error = error_message or "Parsed content is not a valid OpenAPI/Swagger spec (missing 'openapi'/'swagger' and 'info' fields, or not a dictionary)."
            state.response = f"Failed to parse specification: {final_error}"
            logger.error(f"Parsing failed: {final_error}. Input snippet: {spec_text[:200]}...")
            state.next_step = "responder"

        state.update_scratchpad_reason(tool_name, f"Parsing status: {'Success' if state.openapi_schema else 'Failed'}. Response: {state.response}")
        return state

    def _generate_llm_schema_summary(self, state: BotState):
        tool_name = "_generate_llm_schema_summary"
        state.response = "Generating API summary..."
        state.update_scratchpad_reason(tool_name, "Generating schema summary.")
        if not state.openapi_schema:
            state.schema_summary = "Could not generate summary: No schema loaded."
            logger.warning(state.schema_summary)
            state.response = state.schema_summary
            return

        # Basic info extraction
        spec_info = state.openapi_schema.get('info', {})
        title = spec_info.get('title', 'N/A')
        version = spec_info.get('version', 'N/A')
        description = spec_info.get('description', 'N/A')
        num_paths = len(state.openapi_schema.get('paths', {}))
        paths_preview_list = []
        for p, m_dict in list(state.openapi_schema.get('paths', {}).items())[:3]:
            methods = list(m_dict.keys()) if isinstance(m_dict, dict) else '[methods not parsable]'
            paths_preview_list.append(f"  {p}: {methods}")
        paths_preview = "\n".join(paths_preview_list)

        summary_prompt = (
            f"Summarize the following API specification. Focus on its main purpose, key resources/capabilities, "
            f"and any mentioned authentication schemes. Be concise (around 100-150 words).\n\n"
            f"Title: {title}\nVersion: {version}\nDescription: {description[:500]}...\n"
            f"Number of paths: {num_paths}\nExample Paths (first 3):\n{paths_preview}\n\n"
            f"Concise Summary:"
        )
        try:
            state.schema_summary = llm_call_helper(self.worker_llm, summary_prompt)
            logger.info("Schema summary generated.")
            state.response = "API summary created."
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}", exc_info=False) # exc_info=False to reduce log verbosity for common LLM errors
            state.schema_summary = f"Error generating summary: {str(e)[:150]}..." # Truncate error message
            state.response = state.schema_summary
        state.update_scratchpad_reason(tool_name, f"Summary status: {'Success' if state.schema_summary and not state.schema_summary.startswith('Error') else 'Failed'}")

    def _identify_apis_from_schema(self, state: BotState):
        tool_name = "_identify_apis_from_schema"
        state.response = "Identifying API operations..."
        state.update_scratchpad_reason(tool_name, "Identifying APIs.")
        if not state.openapi_schema:
            state.identified_apis = []
            logger.warning("No schema to identify APIs from.")
            state.response = "Cannot identify APIs: No schema loaded."
            return

        apis = []
        paths = state.openapi_schema.get('paths', {})
        for path_url, path_item in paths.items():
            if not isinstance(path_item, dict):
                logger.warning(f"Skipping non-dictionary path item at '{path_url}'")
                continue
            for method, operation_details in path_item.items():
                # Filter for valid HTTP methods and ensure operation_details is a dict
                if method.lower() not in {'get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'trace'} or \
                   not isinstance(operation_details, dict):
                    continue

                # Create a default operationId if not present
                op_id_suffix = path_url.replace('/', '_').replace('{', '').replace('}', '').strip('_')
                default_op_id = f"{method.lower()}_{op_id_suffix or 'root'}"
                
                api_info = {
                    'operationId': operation_details.get('operationId', default_op_id),
                    'path': path_url,
                    'method': method.upper(), # Standardize to uppercase
                    'summary': operation_details.get('summary', ''),
                    'description': operation_details.get('description', ''),
                    'parameters': operation_details.get('parameters', []), # List of param objects
                    'requestBody': operation_details.get('requestBody', {}), # Request body object
                    'responses': operation_details.get('responses', {}) # Responses object
                }
                apis.append(api_info)
        state.identified_apis = apis
        logger.info(f"Identified {len(apis)} API operations.")
        state.response = f"Identified {len(apis)} API operations."
        state.update_scratchpad_reason(tool_name, f"Identified {len(apis)} APIs.")

    def _generate_payload_descriptions(self, state: BotState, target_apis: Optional[List[str]] = None, context_override: Optional[str] = None):
        tool_name = "_generate_payload_descriptions"
        state.response = "Creating payload and response examples..."
        state.update_scratchpad_reason(tool_name, f"Generating payload descriptions. Targets: {target_apis or 'subset'}. Context: {bool(context_override)}")

        if not state.identified_apis:
            logger.warning("No APIs identified, cannot generate payload descriptions.")
            state.response = "Cannot create payload examples: No APIs identified."
            return

        payload_descs = state.payload_descriptions or {} # Initialize if None

        if target_apis: # User explicitly requested descriptions for these APIs
            apis_to_process = [api for api in state.identified_apis if api['operationId'] in target_apis]
        else: # Default: process a small subset of APIs that have request bodies or complex parameters
            apis_with_payload_info = [
                api for api in state.identified_apis
                if api.get('requestBody') or \
                   any(p.get('in') in ['body', 'formData'] for p in api.get('parameters', [])) # Legacy check
            ]
            # Prioritize APIs not yet described, then a small sample
            unprocessed_apis = [api for api in apis_with_payload_info if api['operationId'] not in payload_descs]
            if unprocessed_apis:
                apis_to_process = unprocessed_apis[:2] # Process up to 2 new ones
            else: # If all have descriptions, maybe re-process one for freshness if no specific targets
                apis_to_process = apis_with_payload_info[:1]


        logger.info(f"Attempting to generate payload descriptions for {len(apis_to_process)} APIs.")
        processed_count = 0
        for api_op in apis_to_process:
            op_id = api_op['operationId']
            # Skip if already processed in this run and no context override or specific target
            if op_id in payload_descs and not context_override and not target_apis and processed_count > 0:
                continue

            state.response = f"Generating payload example for '{op_id}'..." # Intermediate update

            # Simplified parameter and request body representation for the prompt
            params_summary = [f"{p.get('name')}({p.get('in')})" for p in api_op.get('parameters', [])[:5]]
            request_body_summary = "Yes" if api_op.get('requestBody') else "No"
            
            # Try to get a success response schema example
            success_response_schema_str = "N/A"
            responses = api_op.get('responses', {})
            for status_code, resp_details in responses.items():
                if status_code.startswith('2') and isinstance(resp_details, dict): # Check if it's a dict
                    content = resp_details.get('content', {})
                    json_content = content.get('application/json', {})
                    schema = json_content.get('schema', {})
                    if schema:
                        success_response_schema_str = json.dumps(schema, indent=2)[:300] # Limit length
                        break
            
            context_str = f" User Context: {context_override}." if context_override else ""

            prompt = (
                f"API Operation: {op_id} ({api_op['method']} {api_op['path']})\n"
                f"Summary: {api_op.get('summary', 'N/A')}\n{context_str}\n"
                f"Parameters (sample): {', '.join(params_summary) if params_summary else 'None'}\n"
                f"Request Body defined: {request_body_summary}\n"
                f"Success Response Schema (sample): {success_response_schema_str}...\n\n"
                f"Task: Provide a concise, typical JSON example for the request payload (if applicable) "
                f"and a brief description of the expected JSON response structure for a successful call. "
                f"Focus on key fields and realistic example values. If no request payload, state 'No request payload needed.' "
                f"Format clearly, e.g.:\n"
                f"Request Payload Example:\n```json\n{{\"key\": \"value\"}}\n```\n"
                f"Expected Response Structure:\nBrief description of response fields (e.g., 'Returns an object with id, name, and status.')."
            )

            try:
                description = llm_call_helper(self.worker_llm, prompt)
                payload_descs[op_id] = description
                processed_count += 1
            except Exception as e:
                logger.error(f"Error generating payload description for {op_id}: {e}", exc_info=False)
                payload_descs[op_id] = f"Error generating description: {str(e)[:100]}..."
                state.response = f"Error creating payload example for '{op_id}': {str(e)[:100]}..."
                if "quota" in str(e).lower() or "429" in str(e): # Check for API limit errors
                    logger.warning(f"Quota error during payload description for {op_id}. Stopping further payload generation for this turn.")
                    state.response += " Hit API limits during generation."
                    break # Stop processing more for this turn if limits are hit
        
        state.payload_descriptions = payload_descs
        if processed_count > 0:
            state.response = f"Generated payload examples for {processed_count} API operation(s)."
        elif not apis_to_process and not target_apis: # No specific targets and no APIs found needing descriptions
             state.response = "No relevant APIs found requiring new payload examples at this time."
        elif target_apis and not apis_to_process: # Specific targets requested but not found/matched
            state.response = f"Could not find the specified API(s) ({target_apis}) to generate payload examples."

        state.update_scratchpad_reason(tool_name, f"Payload descriptions updated for {processed_count} of {len(apis_to_process)} targeted APIs.")


    def _generate_execution_graph(self, state: BotState, goal: Optional[str] = None) -> BotState:
        tool_name = "_generate_execution_graph"
        current_goal = goal or state.plan_generation_goal or "General API workflow overview"
        state.response = f"Building API workflow graph for goal: '{current_goal[:70]}...'"
        state.update_scratchpad_reason(tool_name, f"Generating graph. Goal: {current_goal}")

        if not state.identified_apis:
            state.response = "Cannot generate graph: No API operations identified."
            state.execution_graph = None # Ensure it's cleared
            state.update_scratchpad_reason(tool_name, state.response)
            state.next_step = "responder" # Or handle_unknown if this is unexpected
            return state

        # Prepare a summary of available APIs for the prompt context
        # Include method and path for clarity, and if it requires confirmation (for POST/PUT/DELETE)
        api_summaries_for_prompt = []
        for idx, api in enumerate(state.identified_apis):
            if idx >= 15 and len(state.identified_apis) > 20: # Limit context size for very large specs
                api_summaries_for_prompt.append(f"- ...and {len(state.identified_apis) - 15} more operations.")
                break
            
            # Determine if confirmation is likely needed
            # This is a heuristic; the LLM might also decide based on the operation's nature
            likely_confirmation = api['method'].upper() in ["POST", "PUT", "DELETE", "PATCH"]
            
            api_summaries_for_prompt.append(
                f"- operationId: {api['operationId']} ({api['method']} {api['path']}), "
                f"summary: {api.get('summary', 'N/A')[:100]}, " # Limit summary length
                f"likely_requires_confirmation: {'yes' if likely_confirmation else 'no'}"
            )
        apis_str = "\n".join(api_summaries_for_prompt)

        feedback_str = f"Refinement Feedback: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else ""
        
        # Enhanced prompt for graph generation, including output_mappings and requires_confirmation
        prompt = f"""
        Goal: "{current_goal}". {feedback_str}
        Available API Operations (sample):\n{apis_str}

        Design an API execution graph as a JSON object. The graph must adhere to the Pydantic models:
        InputMapping: {{"source_operation_id": "str", "source_data_path": "str (e.g., '$.response_field_name')", "target_parameter_name": "str", "target_parameter_in": "Literal['path', 'query', 'body', 'body.fieldName']"}}
        OutputMapping: {{"source_data_path": "str (e.g., '$.id', '$.data.token')", "target_data_key": "str (key for shared data pool)"}}
        Node: {{"operationId": "str", "method": "str (GET/POST etc)", "path": "str (/users/{{id}})", "summary": "str", "description": "str (purpose in this workflow)", "input_mappings": [InputMapping], "output_mappings": [OutputMapping], "requires_confirmation": "bool (true for POST/PUT/DELETE usually)"}}
        Edge: {{"from_node": "str", "to_node": "str", "description": "str"}}
        GraphOutput: {{"nodes": [Node], "edges": [Edge], "description": "str (overall workflow)", "refinement_summary": "str (e.g., 'Initial graph with X, Y, Z steps.')"}}

        Instructions:
        1.  Create a "START_NODE" (`operationId: "START_NODE"`, `method: "SYSTEM"`, `path: "/start"`) and an "END_NODE" (`operationId: "END_NODE"`, `method: "SYSTEM"`, `path: "/end"`). These should not have input/output mappings or require confirmation.
        2.  Select 2-5 relevant API operations from the list to achieve the goal. For each, specify its `method` and `path` accurately from the API list.
        3.  Define `input_mappings` for each API node if it depends on data from a previous node's output (use `target_data_key` from the source node's `output_mappings`). The `source_data_path` in InputMapping should refer to a field in the source node's typical JSON response (e.g., '$.id', '$.data.items[0].name').
        4.  Define `output_mappings` for each API node, specifying how to extract key data from its response into a shared pool using `target_data_key` (e.g., 'user_auth_token', 'created_item_id'). The `source_data_path` in OutputMapping refers to a field in *this* node's JSON response.
        5.  Set `requires_confirmation: true` for nodes that modify data (POST, PUT, DELETE, PATCH), `false` otherwise.
        6.  Connect nodes with `edges`. START_NODE must connect to the first API operation(s). Last API operation(s) must connect to END_NODE. Ensure all `from_node` and `to_node` in edges match an `operationId` in the nodes list.
        7.  Provide an overall `description` of the workflow and a `refinement_summary`.

        Output ONLY the JSON object for GraphOutput. Ensure valid JSON.
        Example Node Snippet (Illustrative - adapt fields based on actual API):
        {{
            "operationId": "getUserDetails", "method": "GET", "path": "/users/{{userId}}",
            "summary": "Get details for a specific user", "description": "Retrieves user profile after login.",
            "input_mappings": [
                {{"source_operation_id": "loginUser", "source_data_path": "$.user_id_from_login", "target_parameter_name": "userId", "target_parameter_in": "path"}}
            ],
            "output_mappings": [
                {{"source_data_path": "$.email", "target_data_key": "user_email_address"}},
                {{"source_data_path": "$.profile.settings.theme", "target_data_key": "user_theme_preference"}}
            ],
            "requires_confirmation": false
        }}
        """
        
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            # Attempt to parse the LLM's JSON output using the GraphOutput model
            graph_output_candidate = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output_candidate:
                # Further validation specific to graph generation logic
                if not any(node.operationId == "START_NODE" for node in graph_output_candidate.nodes) or \
                   not any(node.operationId == "END_NODE" for node in graph_output_candidate.nodes):
                    logger.error("LLM generated graph is missing START_NODE or END_NODE.")
                    state.graph_regeneration_reason = "Generated graph missing START_NODE or END_NODE. Please ensure they are included."
                    # Fall through to error handling that might retry or go to handle_unknown
                else:
                    state.execution_graph = graph_output_candidate
                    state.response = "API workflow graph generated."
                    logger.info(f"Graph generated. Description: {graph_output_candidate.description or 'N/A'}")
                    if graph_output_candidate.refinement_summary:
                        logger.info(f"LLM summary for graph: {graph_output_candidate.refinement_summary}")
                    state.graph_regeneration_reason = None # Clear reason on success
                    state.graph_refinement_iterations = 0 # Reset iteration count
                    state.next_step = "verify_graph" # Proceed to verification
                    state.update_scratchpad_reason(tool_name, f"Graph gen success. Next: {state.next_step}")
                    return state # Return early on success

            # If graph_output_candidate is None or structural checks failed
            error_msg = "LLM failed to produce a valid GraphOutput JSON, or it was structurally incomplete (e.g., missing START/END nodes)."
            logger.error(error_msg + f" Raw LLM output snippet: {llm_response[:300]}...")
            state.response = "Failed to generate a valid execution graph (AI output format, structure, or missing critical nodes like START/END)."
            state.execution_graph = None # Ensure graph is None on failure
            state.graph_regeneration_reason = state.graph_regeneration_reason or "LLM output was not a valid GraphOutput object or missed key structural elements."
            
            # Retry logic for initial generation
            current_attempts = state.scratchpad.get('graph_gen_attempts', 0)
            if current_attempts < 1: # Allow one retry for initial generation
                state.scratchpad['graph_gen_attempts'] = current_attempts + 1
                logger.info("Retrying initial graph generation once due to validation/parsing failure.")
                state.next_step = "_generate_execution_graph" # Loop back to this function
            else:
                logger.error("Max initial graph generation attempts reached. Routing to handle_unknown.")
                state.next_step = "handle_unknown"
                state.scratchpad['graph_gen_attempts'] = 0 # Reset for future attempts if any

        except Exception as e:
            logger.error(f"Error during graph generation LLM call or processing: {e}", exc_info=False)
            state.response = f"Error generating graph: {str(e)[:150]}..."
            state.execution_graph = None
            state.graph_regeneration_reason = f"LLM call/processing error: {str(e)[:100]}..."
            state.next_step = "handle_unknown" # Or a more specific error handling node

        state.update_scratchpad_reason(tool_name, f"Graph gen status: {'Success' if state.execution_graph else 'Failed'}. Resp: {state.response}")
        return state

    def process_schema_pipeline(self, state: BotState) -> BotState:
        tool_name = "process_schema_pipeline"
        state.response = "Starting API analysis pipeline..."
        state.update_scratchpad_reason(tool_name, "Starting schema pipeline.")

        if not state.openapi_schema:
            state.response = "Cannot run pipeline: No schema loaded."
            state.next_step = "handle_unknown"
            return state

        # Reset relevant fields before starting the pipeline
        state.schema_summary = None
        state.identified_apis = []
        state.payload_descriptions = {}
        state.execution_graph = None
        state.graph_refinement_iterations = 0
        state.plan_generation_goal = state.plan_generation_goal or "Provide a general overview workflow." # Default goal
        state.scratchpad['graph_gen_attempts'] = 0 # Reset generation attempts
        state.scratchpad['refinement_validation_failures'] = 0 # Reset refinement failures

        # Step 1: Generate Schema Summary
        self._generate_llm_schema_summary(state)
        # Check for critical errors like API quota limits after the first LLM call
        if state.schema_summary and ("Error generating summary: 429" in state.schema_summary or "quota" in state.schema_summary.lower()):
            logger.warning("API limit hit during schema summary. Stopping pipeline.")
            state.response = state.schema_summary # Propagate the error message
            state.next_step = "responder" # Go to responder to show the error
            return state

        # Step 2: Identify APIs from Schema
        self._identify_apis_from_schema(state)
        if not state.identified_apis:
            state.response = (state.response or "") + " No API operations were identified from the schema. Cannot generate payload examples or an execution graph."
            state.next_step = "responder"
            return state

        # Step 3: Generate Payload Descriptions (for a subset)
        self._generate_payload_descriptions(state) # Default behavior: small subset
        # Check for critical errors again
        if any("Error generating description: 429" in desc for desc in state.payload_descriptions.values()) or \
           any("quota" in desc.lower() for desc in state.payload_descriptions.values()):
            logger.warning("API limit hit during payload description generation.")
            state.response = (state.response or "") + " Partial success: Hit API limits while generating some payload examples."
            # Decide if we should still try to generate a graph or stop.
            # For now, let's try to generate a graph if some APIs were identified.

        # Step 4: Generate Initial Execution Graph
        # This will use the default plan_generation_goal if not overridden
        self._generate_execution_graph(state, goal=state.plan_generation_goal)
        
        # After pipeline, save the full analysis to cache if successful
        if state.openapi_schema and state.schema_cache_key and SCHEMA_CACHE and state.next_step not in ["handle_unknown", "responder_with_error_from_pipeline"]: # Avoid caching failed states
            full_analysis_data = {
                'openapi_schema': state.openapi_schema,
                'schema_summary': state.schema_summary,
                'identified_apis': state.identified_apis,
                'payload_descriptions': state.payload_descriptions,
                'execution_graph': state.execution_graph.model_dump() if state.execution_graph else None,
                'plan_generation_goal': state.plan_generation_goal
            }
            save_schema_to_cache(f"{state.schema_cache_key}_full_analysis", full_analysis_data)
            logger.info(f"Saved full analysis to cache for key: {state.schema_cache_key}_full_analysis")


        state.update_scratchpad_reason(tool_name, f"Schema processing pipeline initiated. Next step determined by _generate_execution_graph: {state.next_step}")
        return state

    def verify_graph(self, state: BotState) -> BotState:
        tool_name = "verify_graph"
        state.response = "Verifying API workflow graph..."
        state.update_scratchpad_reason(tool_name, "Verifying graph structure and integrity.")

        if not state.execution_graph:
            state.response = state.response or "No execution graph to verify (possibly due to generation error)."
            state.graph_regeneration_reason = state.graph_regeneration_reason or "No graph was generated to verify."
            logger.warning(f"verify_graph: No graph found. Reason: {state.graph_regeneration_reason}. Routing to _generate_execution_graph for regeneration.")
            state.next_step = "_generate_execution_graph" # Try to regenerate
            return state

        # Perform validation checks
        issues = []
        try:
            # Pydantic model validation (including custom validators in GraphOutput)
            # This will raise ValueError if issues are found by the model_validator
            GraphOutput.model_validate(state.execution_graph.model_dump()) # Re-validate from dict form

            # Check for cycles
            is_dag, cycle_msg = check_for_cycles(state.execution_graph)
            if not is_dag:
                issues.append(cycle_msg or "Graph contains cycles.")

            # Check for START_NODE and END_NODE presence and basic linkage
            node_ids = {node.effective_id for node in state.execution_graph.nodes}
            if "START_NODE" not in node_ids: issues.append("START_NODE is missing.")
            if "END_NODE" not in node_ids: issues.append("END_NODE is missing.")

            if "START_NODE" in node_ids:
                start_outgoing = any(edge.from_node == "START_NODE" for edge in state.execution_graph.edges)
                start_incoming = any(edge.to_node == "START_NODE" for edge in state.execution_graph.edges)
                if not start_outgoing and len(state.execution_graph.nodes) > 2 : # More than just START -> END
                    issues.append("START_NODE has no outgoing edges to actual API operations.")
                if start_incoming:
                    issues.append("START_NODE should not have incoming edges.")
            
            if "END_NODE" in node_ids:
                end_incoming = any(edge.to_node == "END_NODE" for edge in state.execution_graph.edges)
                end_outgoing = any(edge.from_node == "END_NODE" for edge in state.execution_graph.edges)
                if not end_incoming and len(state.execution_graph.nodes) > 2:
                    issues.append("END_NODE has no incoming edges from actual API operations.")
                if end_outgoing:
                    issues.append("END_NODE should not have outgoing edges.")
            
            # Check if all nodes (except START/END) have method and path for execution
            for node in state.execution_graph.nodes:
                if node.effective_id.upper() not in ["START_NODE", "END_NODE"]:
                    if not node.method or not node.path:
                        issues.append(f"Node '{node.effective_id}' is missing 'method' or 'path', required for execution.")
                    # Check input/output mappings consistency (optional, can be complex)
                    # e.g., source_operation_id in input_mappings exists

        except ValidationError as ve: # Catch Pydantic validation errors
            logger.error(f"Graph Pydantic validation failed during verify_graph: {ve}")
            issues.append(f"Graph structure is invalid: {str(ve)[:200]}...") # Add Pydantic error summary
        except Exception as e: # Catch other unexpected errors during verification
            logger.error(f"Unexpected error during graph verification: {e}", exc_info=True)
            issues.append(f"An unexpected error occurred during verification: {str(e)[:100]}.")


        if not issues:
            state.response = "Graph verification successful (Structure, DAG, START/END nodes, basic execution fields)."
            state.update_scratchpad_reason(tool_name, "Graph verification successful.")
            logger.info("Graph verification successful.")
            state.graph_regeneration_reason = None # Clear reason as graph is good
            state.scratchpad['refinement_validation_failures'] = 0 # Reset refinement failures

            # Prepare graph for UI display
            try:
                state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                logger.info("Graph marked to be sent to UI after verification.")
            except Exception as e:
                logger.error(f"Error serializing graph for sending after verification: {e}")

            # Decide next step: refine further or finalize
            if state.graph_refinement_iterations < state.max_refinement_iterations:
                # Potentially refine even if "verified" if user asked for specific refinement goal
                # For now, if verified, assume initial refinements are done unless user explicitly asks more.
                # Let's assume for now that if it's verified, we describe it.
                # Refinement loop is more for fixing LLM errors.
                logger.info("Graph verified. Proceeding to describe graph.")
                state.next_step = "describe_graph"
            else: # Max refinements reached, or verification passed after some iterations
                logger.info("Max refinement iterations reached or graph verified. Proceeding to describe graph.")
                state.next_step = "describe_graph"
            
            # If this was the result of an initial spec processing, set a comprehensive response
            if state.input_is_spec: # Flag set when parse_openapi_spec handles a new spec
                api_title = state.openapi_schema.get('info', {}).get('title', 'the API') if state.openapi_schema else 'the API'
                state.response = (
                    f"Successfully processed the OpenAPI specification for '{api_title}'. "
                    f"Identified {len(state.identified_apis)} API operations, generated example payloads, "
                    f"and created an API workflow graph with {len(state.execution_graph.nodes)} steps. "
                    "The graph is verified. You can now ask questions, request specific plan refinements, or try to execute the workflow."
                )
                state.input_is_spec = False # Reset flag

        else: # Issues found
            error_details = " ".join(issues)
            state.response = f"Graph verification failed: {error_details}."
            state.graph_regeneration_reason = f"Verification failed: {error_details}." # Set reason for refinement/regeneration
            logger.warning(f"Graph verification failed: {error_details}.")

            # Attempt to refine or regenerate
            if state.graph_refinement_iterations < state.max_refinement_iterations:
                logger.info(f"Verification failed. Attempting graph refinement (iteration {state.graph_refinement_iterations + 1}).")
                state.next_step = "refine_api_graph"
            else:
                logger.warning("Max refinement iterations reached, but graph still has verification issues. Attempting full regeneration.")
                state.next_step = "_generate_execution_graph" # Reset and try full regeneration
                state.graph_refinement_iterations = 0 # Reset iterations for full regen
                state.scratchpad['graph_gen_attempts'] = 0 # Reset gen attempts for full regen

        state.update_scratchpad_reason(tool_name, f"Verification result: {state.response[:200]}...")
        return state

    def refine_api_graph(self, state: BotState) -> BotState:
        tool_name = "refine_api_graph"
        iteration = state.graph_refinement_iterations + 1
        state.response = f"Refining API workflow graph (Attempt {iteration}/{state.max_refinement_iterations})..."
        state.update_scratchpad_reason(tool_name, f"Refining graph. Iteration: {iteration}. Reason: {state.graph_regeneration_reason or 'General refinement request.'}")

        if not state.execution_graph:
            state.response = "No graph to refine. Please generate a graph first."
            logger.warning("refine_api_graph: No execution_graph found.")
            state.next_step = "_generate_execution_graph" # Try to generate one
            return state

        if iteration > state.max_refinement_iterations:
            state.response = (
                f"Max refinement iterations ({state.max_refinement_iterations}) reached. "
                f"Using current graph (description: {state.execution_graph.description or 'N/A'}). "
                "Please try a new goal or manually edit if needed."
            )
            logger.warning("Max refinement iterations reached. Proceeding with current graph.")
            state.next_step = "describe_graph" # Describe the current (possibly flawed) graph
            return state

        try:
            current_graph_json = state.execution_graph.model_dump_json(indent=2)
        except Exception as e:
            logger.error(f"Error serializing current graph for refinement prompt: {e}")
            state.response = "Error preparing current graph for refinement. Cannot proceed."
            state.next_step = "handle_unknown"
            return state

        # Prepare API context for the refinement prompt (similar to generation)
        api_summaries_for_prompt = []
        for idx, api in enumerate(state.identified_apis):
            if idx >= 10 and len(state.identified_apis) > 15: # Shorter list for refinement context
                api_summaries_for_prompt.append(f"- ...and {len(state.identified_apis) - 10} more operations.")
                break
            likely_confirmation = api['method'].upper() in ["POST", "PUT", "DELETE", "PATCH"]
            api_summaries_for_prompt.append(
                f"- opId: {api['operationId']} ({api['method']} {api['path']}), "
                f"summary: {api.get('summary', 'N/A')[:70]}, "
                f"confirm: {'yes' if likely_confirmation else 'no'}"
            )
        apis_ctx = "\n".join(api_summaries_for_prompt)
        
        # Refinement prompt - more targeted than initial generation
        prompt = f"""
        User's Overall Goal: "{state.plan_generation_goal or 'General workflow'}"
        Feedback for Refinement: "{state.graph_regeneration_reason or 'General request to improve the graph.'}"
        
        Current Graph (JSON to be refined):
        ```json
        {current_graph_json}
        ```
        Available API Operations (sample for context):\n{apis_ctx}

        Task: Refine the current graph based on the feedback. Ensure the refined graph:
        1.  Strictly adheres to the Pydantic model structure for GraphOutput, Node, Edge, InputMapping, OutputMapping (see previous generation prompt for details if needed).
        2.  Includes "START_NODE" and "END_NODE" correctly linked.
        3.  All node `operationId`s in edges must exist in the `nodes` list.
        4.  Nodes intended for execution have `method` and `path` attributes.
        5.  `input_mappings` and `output_mappings` are logical for data flow. `source_data_path` should be plausible JSON paths.
        6.  `requires_confirmation` is set appropriately (true for POST, PUT, DELETE, PATCH).
        7.  Addresses the specific feedback provided. If feedback mentions adding/removing/reordering specific operations, reflect that.
        8.  Provide a concise `refinement_summary` field in the JSON explaining what was changed or attempted.

        Output ONLY the refined GraphOutput JSON object.
        """
        
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            refined_graph_candidate = parse_llm_json_output_with_model(llm_response_str, expected_model=GraphOutput)

            if refined_graph_candidate:
                logger.info(f"Refinement attempt (iter {iteration}) produced a structurally valid GraphOutput.")
                state.execution_graph = refined_graph_candidate # Update with the refined graph
                refinement_summary = refined_graph_candidate.refinement_summary or "AI provided no specific summary for this refinement."
                state.update_scratchpad_reason(tool_name, f"LLM Refinement Summary (Iter {iteration}): {refinement_summary}")
                
                state.graph_refinement_iterations = iteration
                state.response = f"Graph refined (Iteration {iteration}). Summary: {refinement_summary}"
                state.graph_regeneration_reason = None # Clear reason after refinement attempt
                state.scratchpad['refinement_validation_failures'] = 0 # Reset validation failures
                state.next_step = "verify_graph" # Always verify after refinement
            else:
                # LLM failed to produce valid JSON or it didn't match GraphOutput model
                error_msg = "LLM refinement failed to produce a GraphOutput JSON that is valid or self-consistent."
                logger.error(error_msg + f" Raw LLM output snippet for refinement: {llm_response_str[:300]}...")
                state.response = f"Error during graph refinement (iteration {iteration}): AI output was invalid. Will retry refinement or regenerate graph."
                # Keep existing graph_regeneration_reason or set a new one
                state.graph_regeneration_reason = state.graph_regeneration_reason or "LLM output for refinement was not a valid GraphOutput object or had structural issues."
                
                state.scratchpad['refinement_validation_failures'] = state.scratchpad.get('refinement_validation_failures', 0) + 1
                
                if iteration < state.max_refinement_iterations:
                    if state.scratchpad['refinement_validation_failures'] >= 2: # Too many consecutive failures
                        logger.warning(f"Multiple consecutive refinement validation failures (iter {iteration}). Escalating to full graph regeneration.")
                        state.response += " Attempting full regeneration due to persistent refinement issues."
                        state.next_step = "_generate_execution_graph"
                        state.graph_refinement_iterations = 0 # Reset for full regeneration
                        state.scratchpad['refinement_validation_failures'] = 0
                        state.scratchpad['graph_gen_attempts'] = 0 # Reset gen attempts for full regeneration
                    else:
                        # Retry refinement
                        state.next_step = "refine_api_graph"
                else: # Max refinement iterations reached after LLM output error
                    logger.warning(f"Max refinement iterations reached after LLM output error during refinement. Describing last valid graph or failing.")
                    state.next_step = "describe_graph" # Describe whatever graph we have (or handle if None)
        
        except Exception as e:
            logger.error(f"Error during graph refinement LLM call or processing (iter {iteration}): {e}", exc_info=False)
            state.response = f"Error refining graph (iter {iteration}): {str(e)[:150]}..."
            state.graph_regeneration_reason = state.graph_regeneration_reason or f"Refinement LLM call/processing error (iter {iteration}): {str(e)[:100]}..."
            # Decide if we should retry refinement, regenerate, or give up
            if iteration < state.max_refinement_iterations:
                state.next_step = "refine_api_graph" # Retry refinement
            else:
                logger.warning(f"Max refinement iterations reached after exception. Describing graph or failing.")
                state.next_step = "describe_graph"
        return state

    def describe_graph(self, state: BotState) -> BotState:
        tool_name = "describe_graph"
        state.response = "Preparing graph description..."
        state.update_scratchpad_reason(tool_name, "Preparing to describe the current execution graph.")

        if not state.execution_graph:
            state.response = (state.response or "") + " No execution graph is currently available to describe."
            logger.warning("describe_graph: No execution_graph found in state.")
            # If no graph, and we came from a failed generation/refinement, this message might be superseded
            # by the error message from that process.
        else:
            # Use the description from the graph object if available and sufficiently detailed
            graph_desc = state.execution_graph.description
            if not graph_desc or len(graph_desc) < 20: # If description is too short or missing
                logger.info("Graph description is short or missing, generating a dynamic one.")
                # Prepare a concise summary of nodes for the LLM to describe
                node_summaries = []
                for node in state.execution_graph.nodes:
                    node_summaries.append(f"- {node.effective_id}: {node.summary or node.operationId[:50]}")
                
                nodes_str = "\n".join(node_summaries[:5]) # Show first 5 nodes
                if len(node_summaries) > 5:
                    nodes_str += f"\n- ... and {len(node_summaries) - 5} more nodes."

                prompt = (
                    f"The following API execution graph has been generated for the goal: '{state.plan_generation_goal or 'general use'}'.\n"
                    f"Nodes in the graph ({len(state.execution_graph.nodes)} total, sample):\n{nodes_str}\n\n"
                    f"Please provide a concise, user-friendly natural language description of this workflow. "
                    f"Explain its overall purpose and the general sequence of operations. "
                    f"Use Markdown for readability (e.g., a brief introductory sentence, then bullet points for key stages if appropriate)."
                )
                try:
                    dynamic_desc = llm_call_helper(self.worker_llm, prompt)
                    # If the original graph had a description, prepend the dynamic one or combine them
                    if graph_desc and graph_desc != dynamic_desc: # Avoid duplication if LLM just repeats
                        final_desc_for_user = f"**Overall Workflow Plan for: '{state.plan_generation_goal or 'General Use'}'**\n\n{dynamic_desc}\n\n*Original AI-generated graph description: {graph_desc}*"
                    else:
                        final_desc_for_user = f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{dynamic_desc}"
                except Exception as e:
                    logger.error(f"Error generating dynamic graph description: {e}")
                    final_desc_for_user = (
                        f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n"
                        f"{graph_desc or 'No detailed description available. The graph includes nodes like ' + ', '.join([n.effective_id for n in state.execution_graph.nodes[:3]]) + '...'}"
                    )
            else: # Sufficient description exists in the graph object
                 final_desc_for_user = f"**Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'**\n\n{graph_desc}"

            # Append refinement summary if available
            if state.execution_graph.refinement_summary:
                final_desc_for_user += f"\n\n**Last Refinement Note:** {state.execution_graph.refinement_summary}"
            
            state.response = final_desc_for_user
            
            # Also ensure the graph JSON is available for the UI if not already sent
            if 'graph_to_send' not in state.scratchpad and state.execution_graph:
                 try:
                    state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                 except Exception as e:
                    logger.error(f"Error serializing graph for sending during describe_graph: {e}")


        state.update_scratchpad_reason(tool_name, f"Graph description generated/retrieved. Response set: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        tool_name = "get_graph_json"
        state.response = "Fetching graph JSON..."
        state.update_scratchpad_reason(tool_name, "Attempting to provide graph JSON.")

        if not state.execution_graph:
            state.response = "No execution graph is currently available."
        else:
            try:
                graph_json_str = state.execution_graph.model_dump_json(indent=2)
                # Set the JSON string directly for the UI to render in a specific way
                state.scratchpad['graph_to_send'] = graph_json_str # For UI update
                state.response = f"The current API workflow graph is available in the graph view. You can also copy the JSON from there if needed."
                # If you want to send the JSON directly in chat:
                # state.response = f"Current API Workflow Graph JSON:\n```json\n{graph_json_str}\n```"
                logger.info("Provided graph JSON to scratchpad for UI.")
            except Exception as e:
                logger.error(f"Error serializing execution_graph to JSON: {e}")
                state.response = f"Error serializing graph to JSON: {str(e)}"
        
        state.next_step = "responder"
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        tool_name = "answer_openapi_query"
        state.response = "Thinking about your question..."
        state.update_scratchpad_reason(tool_name, f"Attempting to answer user query: {state.user_input[:100]}")

        if not state.openapi_schema and not state.execution_graph: # Check if any context exists
            state.response = "I don't have an OpenAPI specification loaded or a graph generated yet. Please provide one first."
            state.next_step = "responder"
            return state

        # Build context string for the LLM
        context_parts = []
        if state.user_input:
            context_parts.append(f"User Question: \"{state.user_input}\"")

        if state.schema_summary:
            context_parts.append(f"\n### API Specification Summary\n{state.schema_summary}")

        if state.identified_apis:
            api_list_md = "\n### Identified API Operations (Sample - first 10):\n"
            for i, api in enumerate(state.identified_apis[:10]):
                api_list_md += f"- **{api.get('operationId', 'N/A')}**: {api.get('method', '?')} {api.get('path', '?')} - _{api.get('summary', 'No summary')[:70]}..._\n"
            if len(state.identified_apis) > 10:
                api_list_md += f"- ... and {len(state.identified_apis) - 10} more.\n"
            context_parts.append(api_list_md)

        if state.execution_graph and state.execution_graph.description:
            graph_desc_md = f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') Description:\n{state.execution_graph.description}"
            if state.execution_graph.refinement_summary:
                graph_desc_md += f"\nLast Refinement: {state.execution_graph.refinement_summary}"
            context_parts.append(graph_desc_md)
        elif state.execution_graph:
             context_parts.append(f"\n### Current Workflow Graph ('{state.plan_generation_goal or 'General Purpose'}') exists but has no detailed description.")


        # Check if the question relates to a specific API's payload and include if available
        payload_info_md = ""
        if state.user_input and state.payload_descriptions:
            for op_id, desc_text in state.payload_descriptions.items():
                # Simple check if operationId is mentioned in the user input
                if op_id.lower() in state.user_input.lower():
                    payload_info_md = f"\n### Payload/Response Example for '{op_id}':\n{desc_text}\n"
                    context_parts.append(payload_info_md)
                    break # Add first match

        full_context = "\n".join(context_parts)
        if not full_context.strip(): # Ensure there's some context
            full_context = "No specific API context available, but an OpenAPI spec might be loaded."

        prompt = f"""
        You are an expert API assistant. Answer the user's question based on the provided context.
        Use Markdown for formatting (e.g., headings, lists, bolding, italics, and code blocks for JSON snippets).

        {full_context}

        Please provide a clear, concise, and helpful answer to the User Question.
        If the information is not available in the context, state that clearly.
        If listing multiple items (like API operations), use bullet points.
        If showing example JSON, ensure it is in a Markdown code block (e.g., ```json ... ```).
        Focus only on answering the question. Do not add conversational fluff beyond the answer.
        """
        try:
            state.response = llm_call_helper(self.worker_llm, prompt)
            logger.info("Successfully generated answer for OpenAPI query.")
        except Exception as e:
            logger.error(f"Error generating answer for OpenAPI query: {e}", exc_info=False)
            state.response = f"### Error Answering Query\nSorry, I encountered an error while trying to answer your question: {str(e)[:100]}..."
        
        state.update_scratchpad_reason(tool_name, f"Answered query. Response snippet: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        tool_name = "interactive_query_planner"
        state.response = "Planning how to address your interactive query..."
        state.update_scratchpad_reason(tool_name, f"Entering interactive query planner for input: {state.user_input[:100]}")

        # Clear previous interactive plan details from scratchpad
        state.scratchpad.pop('interactive_action_plan', None)
        state.scratchpad.pop('current_interactive_action_idx', None)
        state.scratchpad.pop('current_interactive_results', None)

        # Prepare context for the planner LLM
        graph_summary = state.execution_graph.description[:150] + "..." if state.execution_graph and state.execution_graph.description else "No graph currently generated."
        payload_keys_sample = list(state.payload_descriptions.keys())[:3]
        
        prompt = f"""
        User Query: "{state.user_input}"

        Current State Context:
        - API Spec Summary: {'Available' if state.schema_summary else 'Not available.'}
        - Identified APIs count: {len(state.identified_apis) if state.identified_apis else 0}. Example OpIDs: {", ".join([api['operationId'] for api in state.identified_apis[:3]])}...
        - Example Payload Descriptions available for OpIDs (sample): {payload_keys_sample}...
        - Current Execution Graph Goal: {state.plan_generation_goal or 'Not set.'}
        - Current Graph Description: {graph_summary}
        - Workflow Execution Status: {state.workflow_execution_status}

        Available Internal Actions (choose one or more in sequence, output as a JSON list):
        1.  `rerun_payload_generation`: Regenerate payload/response examples for specific APIs, possibly with new user-provided context.
            Params: {{ "operation_ids_to_update": ["opId1", "opId2"], "new_context": "User's new context string for generation" }}
        2.  `contextualize_graph_descriptions`: Rewrite descriptions within the *existing* graph (overall, nodes, edges) to reflect new user context or focus. This does NOT change graph structure.
            Params: {{ "new_context_for_graph": "User's new context/focus for descriptions" }}
        3.  `regenerate_graph_with_new_goal`: Create a *new* graph if the user states a completely different high-level goal OR requests a significant structural change (add/remove/reorder API steps).
            Params: {{ "new_goal_string": "User's new goal, incorporating the structural change (e.g., 'Workflow to X, then Y, and then Z as the last step')" }}
        4.  `refine_existing_graph_structure`: For minor structural adjustments to the existing graph (e.g., "add API Z after Y but before END_NODE", "remove API X"). This implies the overall goal is similar but the sequence/nodes need adjustment. The LLM will be asked to refine the current graph JSON.
            Params: {{ "refinement_instructions_for_structure": "User's specific feedback for structural refinement (e.g., 'Add operation Z after Y', 'Ensure X comes before Y')" }}
        5.  `answer_query_directly`: If the query can be answered using existing information (API summary, API list, current graph description, existing payload examples) without modifications to artifacts.
            Params: {{ "query_for_synthesizer": "The original user query or a rephrased one for direct answering." }}
        6.  `setup_workflow_execution_interactive`: If the user asks to run/execute the current graph. This action prepares the system for execution.
            Params: {{ "initial_parameters": {{ "param1": "value1" }} }} (Optional initial parameters for the workflow, if provided by user)
        7.  `resume_workflow_with_payload_interactive`: If the workflow is 'paused_for_confirmation' and the user provides the necessary payload/confirmation to continue.
            Params: {{ "confirmed_payload": {{...}} }} (The JSON payload confirmed or provided by the user)
        8.  `synthesize_final_answer`: (Usually the last step of a plan) Formulate a comprehensive answer to the user based on the outcomes of previous internal actions or if no other action is suitable.
            Params: {{ "synthesis_prompt_instructions": "Instructions for the LLM on what to include in the final answer, summarizing actions taken or information gathered." }}

        Task:
        1. Analyze the user's query in the context of the current system state.
        2. Create a short, logical "interactive_action_plan" (a list of action objects, max 3-4 steps).
           - For requests to run the graph, use `setup_workflow_execution_interactive`.
           - If the graph is paused and user provides data, use `resume_workflow_with_payload_interactive`.
           - For structural changes like "add X at the end", prefer `regenerate_graph_with_new_goal` or `refine_existing_graph_structure`.
        3. Provide a brief "user_query_understanding" (1-2 sentences).

        Output ONLY a JSON object with this structure:
        {{
          "user_query_understanding": "Brief interpretation of user's need.",
          "interactive_action_plan": [
            {{"action_name": "action_enum_value", "action_params": {{...}}, "description": "Briefly, why this action is chosen."}}
          ]
        }}
        If the query is very simple and can be answered directly, the plan might just be one "answer_query_directly" or "synthesize_final_answer" action.
        If the query is ambiguous or cannot be handled by available actions, use "synthesize_final_answer" with instructions to inform the user.
        """
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            parsed_plan_data = parse_llm_json_output_with_model(llm_response_str) # Expecting dict

            if parsed_plan_data and "interactive_action_plan" in parsed_plan_data and \
               isinstance(parsed_plan_data["interactive_action_plan"], list) and \
               "user_query_understanding" in parsed_plan_data:
                
                state.scratchpad['user_query_understanding'] = parsed_plan_data["user_query_understanding"]
                state.scratchpad['interactive_action_plan'] = parsed_plan_data["interactive_action_plan"]
                state.scratchpad['current_interactive_action_idx'] = 0
                state.scratchpad['current_interactive_results'] = []
                
                state.response = f"Understood query: {state.scratchpad['user_query_understanding']}. Starting internal actions..."
                logger.info(f"Interactive plan generated: {state.scratchpad['interactive_action_plan']}")
                state.next_step = "interactive_query_executor"
            else:
                logger.error(f"LLM failed to produce a valid interactive plan. Raw: {llm_response_str[:300]}")
                raise ValueError("LLM failed to produce a valid interactive plan JSON structure with required keys.")
        except Exception as e:
            logger.error(f"Error in interactive_query_planner: {e}", exc_info=False)
            state.response = f"Sorry, I encountered an error while planning how to address your request: {str(e)[:100]}..."
            # Fallback: try to answer directly or send to unknown
            state.next_step = "answer_openapi_query" # Or handle_unknown
        
        state.update_scratchpad_reason(tool_name, f"Interactive plan generated. Next: {state.next_step}. Response: {state.response[:100]}")
        return state

    def _internal_contextualize_graph_descriptions(self, state: BotState, new_context: str) -> str:
        """Helper to rewrite descriptions in the existing graph."""
        tool_name = "_internal_contextualize_graph_descriptions"
        if not state.execution_graph:
            return "No graph to contextualize."
        if not new_context:
            return "No new context provided for contextualization."

        logger.info(f"Attempting to contextualize graph descriptions with context: {new_context[:100]}...")
        
        # We'll try to update the main graph description and a few node descriptions.
        # This is a simplified approach. A more complex one might target specific nodes.
        
        # Contextualize overall graph description
        if state.execution_graph.description:
            prompt_overall = (
                f"Current overall graph description: \"{state.execution_graph.description}\"\n"
                f"New User Context/Focus: \"{new_context}\"\n\n"
                f"Rewrite the graph description to incorporate this new context/focus, keeping it concise. Output only the new description text."
            )
            try:
                state.execution_graph.description = llm_call_helper(self.worker_llm, prompt_overall)
                logger.info(f"Overall graph description contextualized: {state.execution_graph.description[:100]}...")
            except Exception as e:
                logger.error(f"Error contextualizing overall graph description: {e}")
                # Keep original if error

        # Contextualize a few node descriptions (e.g., first 2-3 non-START/END nodes)
        nodes_to_update = [n for n in state.execution_graph.nodes if n.operationId not in ["START_NODE", "END_NODE"]][:3]
        for node in nodes_to_update:
            if node.description:
                prompt_node = (
                    f"Current description for node '{node.effective_id}' ({node.summary}): \"{node.description}\"\n"
                    f"Overall User Context/Focus for the graph: \"{new_context}\"\n\n"
                    f"Rewrite this node's description to align with the new context, focusing on its role in the workflow under this context. Output only the new description text for this node."
                )
                try:
                    node.description = llm_call_helper(self.worker_llm, prompt_node)
                    logger.info(f"Node '{node.effective_id}' description contextualized: {node.description[:100]}...")
                except Exception as e:
                    logger.error(f"Error contextualizing node '{node.effective_id}' description: {e}")
                    # Keep original if error
        
        state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2) # Update UI
        state.update_scratchpad_reason(tool_name, f"Graph descriptions contextualized with context: {new_context[:70]}.")
        return f"Graph descriptions have been updated to reflect the context: '{new_context[:70]}...'."


    def interactive_query_executor(self, state: BotState) -> BotState:
        tool_name = "interactive_query_executor"
        plan = state.scratchpad.get('interactive_action_plan', [])
        idx = state.scratchpad.get('current_interactive_action_idx', 0)
        results = state.scratchpad.get('current_interactive_results', [])

        if not plan or idx >= len(plan):
            # No plan, or plan finished. If there were results, synthesize them.
            # Otherwise, just go to responder.
            if results: # If there are results from previous steps, synthesize a final answer
                logger.info("Interactive plan finished. Synthesizing final answer from results.")
                final_synthesis_instr = (
                    f"The user's query was: '{state.user_input}'. "
                    f"My understanding was: '{state.scratchpad.get('user_query_understanding', 'N/A')}'. "
                    f"The following internal actions were taken with these results: {'; '.join([str(r)[:100] + '...' for r in results])}. "
                    f"Please formulate a comprehensive final answer to the user based on these actions and results. "
                    f"If a graph was updated, mention it. If a workflow was started, inform about it."
                )
                # This re-uses the _internal_synthesize_final_answer logic
                # but we need to call it directly as we are at the end of the plan.
                # For simplicity, let's assume the last step of a plan is ALWAYS synthesize_final_answer
                # or the planner LLM adds it if needed.
                # If we reach here without a synthesize step, we default to responder.
                state.response = state.response or "Finished interactive processing. " + (results[-1] if results else "No specific result.")
            else:
                state.response = state.response or "Finished interactive processing with no specific actions taken or results."
            
            state.next_step = "responder"
            state.update_scratchpad_reason(tool_name, "Interactive plan execution completed or no plan.")
            return state

        action = plan[idx]
        action_name = action.get("action_name")
        action_params = action.get("action_params", {})
        action_description = action.get("description", "No description for action.") # Get action description

        state.response = f"Executing internal step ({idx + 1}/{len(plan)}): {action_description[:70]}..."
        state.update_scratchpad_reason(tool_name, f"Executing action ({idx + 1}/{len(plan)}): {action_name} - {action_description}")
        
        action_result_message = f"Action '{action_name}' completed." # Default success message

        try:
            if action_name == "rerun_payload_generation":
                op_ids = action_params.get("operation_ids_to_update", [])
                new_ctx = action_params.get("new_context", "")
                if op_ids and new_ctx:
                    self._generate_payload_descriptions(state, target_apis=op_ids, context_override=new_ctx)
                    action_result_message = f"Payload examples updated for {op_ids} with context '{new_ctx[:30]}...'."
                    results.append(action_result_message)
                else:
                    action_result_message = "Skipped rerun_payload_generation: Missing operation_ids or new_context."
                    results.append(action_result_message)
                state.next_step = "interactive_query_executor" # Continue to next action

            elif action_name == "contextualize_graph_descriptions":
                new_ctx_graph = action_params.get("new_context_for_graph", "")
                if new_ctx_graph:
                    action_result_message = self._internal_contextualize_graph_descriptions(state, new_ctx_graph)
                    results.append(action_result_message)
                else:
                    action_result_message = "Skipped contextualize_graph_descriptions: Missing new_context_for_graph."
                    results.append(action_result_message)
                state.next_step = "interactive_query_executor"

            elif action_name == "regenerate_graph_with_new_goal":
                new_goal = action_params.get("new_goal_string")
                if new_goal:
                    state.plan_generation_goal = new_goal
                    state.execution_graph = None # Clear old graph
                    state.graph_refinement_iterations = 0
                    state.scratchpad['graph_gen_attempts'] = 0
                    state.scratchpad['refinement_validation_failures'] = 0
                    # This call will set state.next_step to verify_graph or handle_unknown
                    self._generate_execution_graph(state, goal=new_goal)
                    action_result_message = f"Graph regeneration started for new goal: {new_goal[:50]}..."
                    results.append(action_result_message)
                    # The next step is determined by _generate_execution_graph
                else:
                    action_result_message = "Skipped regenerate_graph_with_new_goal: Missing new_goal_string."
                    results.append(action_result_message)
                    state.next_step = "interactive_query_executor"


            elif action_name == "refine_existing_graph_structure":
                refinement_instr = action_params.get("refinement_instructions_for_structure")
                if refinement_instr and state.execution_graph:
                    state.graph_regeneration_reason = refinement_instr # This feedback is used by refine_api_graph
                    state.scratchpad['refinement_validation_failures'] = 0 # Reset for this attempt
                    # This call will set state.next_step to verify_graph
                    self.refine_api_graph(state) # refine_api_graph handles iterations
                    action_result_message = f"Graph refinement (structure) started with instructions: {refinement_instr[:50]}..."
                    results.append(action_result_message)
                    # The next step is determined by refine_api_graph
                elif not state.execution_graph:
                    action_result_message = "Skipped refine_existing_graph_structure: No graph exists."
                    results.append(action_result_message)
                    state.next_step = "interactive_query_executor"
                else: # No instructions
                    action_result_message = "Skipped refine_existing_graph_structure: Missing refinement_instructions_for_structure."
                    results.append(action_result_message)
                    state.next_step = "interactive_query_executor"


            elif action_name == "answer_query_directly":
                query_to_answer = action_params.get("query_for_synthesizer", state.user_input or "")
                # Temporarily set user_input for answer_openapi_query
                original_user_input = state.user_input
                state.user_input = query_to_answer
                self.answer_openapi_query(state) # This sets state.response and state.next_step="responder"
                state.user_input = original_user_input # Restore original user input
                action_result_message = f"Direct answer generated for: {query_to_answer[:50]}..."
                results.append(action_result_message)
                # answer_openapi_query sets next_step to responder, which is fine if this is the last action.
                # If not last, we override next_step.
                if idx + 1 < len(plan):
                    state.next_step = "interactive_query_executor"
                else: # It was the last action
                    state.next_step = "responder"


            elif action_name == "setup_workflow_execution_interactive":
                # This action calls the setup method, which itself sets next_step="responder"
                # and prepares for main.py to start the streaming.
                self.setup_workflow_execution(state) # This sets state.response and state.next_step
                action_result_message = f"Workflow execution setup initiated. Status: {state.workflow_execution_status}."
                results.append(action_result_message)
                # setup_workflow_execution sets next_step to responder.
                # If this is not the last step in the plan, we might need to override.
                # However, typically, setting up execution is a final step for the planner.
                if idx + 1 < len(plan): # If more actions planned after setup (unusual for execution setup)
                    logger.warning("More actions planned after setup_workflow_execution_interactive. This might be unexpected.")
                    state.next_step = "interactive_query_executor"
                else:
                    state.next_step = "responder" # As set by setup_workflow_execution

            elif action_name == "resume_workflow_with_payload_interactive":
                confirmed_payload = action_params.get("confirmed_payload")
                if confirmed_payload and isinstance(confirmed_payload, dict):
                    self.resume_workflow_with_payload(state, confirmed_payload)
                    action_result_message = f"Workflow resumption with payload initiated. Status: {state.workflow_execution_status}."
                    results.append(action_result_message)
                else:
                    action_result_message = "Skipped resume_workflow: Missing or invalid confirmed_payload."
                    results.append(action_result_message)
                # resume_workflow_with_payload sets next_step to responder
                if idx + 1 < len(plan):
                     logger.warning("More actions planned after resume_workflow. This might be unexpected.")
                     state.next_step = "interactive_query_executor"
                else:
                    state.next_step = "responder"


            elif action_name == "synthesize_final_answer":
                synthesis_instr = action_params.get("synthesis_prompt_instructions", "Summarize actions and provide a final response.")
                # Gather all results so far for context
                all_prior_results_summary = "; ".join([str(r)[:150] for r in results])
                
                final_synthesis_prompt = (
                    f"User's original query: '{state.user_input}'.\n"
                    f"My understanding of the query: '{state.scratchpad.get('user_query_understanding', 'N/A')}'.\n"
                    f"Internal actions taken and their results (summary): {all_prior_results_summary if all_prior_results_summary else 'No specific actions taken or results to summarize.'}\n"
                    f"Additional instructions for synthesis: {synthesis_instr}\n\n"
                    f"Based on all the above, formulate a comprehensive and helpful final answer for the user in Markdown format."
                )
                try:
                    state.response = llm_call_helper(self.worker_llm, final_synthesis_prompt)
                    action_result_message = "Final answer synthesized."
                    # results.append(action_result_message) # The result *is* the final answer (state.response)
                except Exception as e:
                    logger.error(f"Error synthesizing final answer: {e}")
                    state.response = f"Sorry, I encountered an error while synthesizing the final answer: {str(e)[:100]}"
                    action_result_message = "Error during final answer synthesis."
                results.append(action_result_message) # Log that synthesis was attempted/done
                state.next_step = "responder" # Final answer, so go to responder

            else:
                action_result_message = f"Unknown or unhandled action: {action_name}."
                logger.warning(action_result_message)
                results.append(action_result_message)
                state.next_step = "interactive_query_executor" # Try next action if any

        except Exception as e_action:
            logger.error(f"Error executing action '{action_name}': {e_action}", exc_info=True)
            action_result_message = f"Error during action '{action_name}': {str(e_action)[:100]}..."
            results.append(action_result_message) # Log the error result
            state.response = action_result_message # Set response to the error
            # Decide if we should stop the plan or try to continue
            state.next_step = "interactive_query_executor" # Default to trying next action

        # Store result of current action (even if it's an error message)
        # state.scratchpad['current_interactive_results'] = results # results list is updated directly

        # Move to next action or finish
        state.scratchpad['current_interactive_action_idx'] = idx + 1
        if state.scratchpad['current_interactive_action_idx'] >= len(plan):
            # If the plan is finished, and the last action didn't set next_step to responder (e.g. answer_query_directly or synthesize_final_answer)
            # we need to ensure we go to the responder.
            if state.next_step != "responder" and action_name not in ["answer_query_directly", "synthesize_final_answer", "setup_workflow_execution_interactive", "resume_workflow_with_payload_interactive"]:
                 # If the last action was something like regenerate_graph, it sets its own next_step (e.g. verify_graph).
                 # We should honor that, UNLESS it was the very last action in the plan.
                 # This logic is getting complex. The planner should ensure the last step is usually a "finalizing" one.
                 # For now, if an action sets its own next_step (like _generate_execution_graph -> verify_graph), let it be.
                 # If it doesn't (like rerun_payload_generation), and it's the last action, then go to responder.
                if state.next_step == "interactive_query_executor": # Meaning the action itself didn't route elsewhere
                    logger.info(f"Interactive plan finished after action '{action_name}'. Finalizing with synthesis or responder.")
                    # Synthesize a final response if not already done by the last action
                    if action_name != "synthesize_final_answer":
                        final_synthesis_instr = (
                            f"The user's query was: '{state.user_input}'. "
                            f"My understanding was: '{state.scratchpad.get('user_query_understanding', 'N/A')}'. "
                            f"The following internal actions were taken with these results: {'; '.join([str(r)[:100] + '...' for r in results])}. "
                            f"Please formulate a comprehensive final answer to the user based on these actions and results."
                        )
                        try:
                            state.response = llm_call_helper(self.worker_llm, final_synthesis_instr)
                        except Exception as e:
                            state.response = "Processed your request. " + (results[-1] if results else "")
                    # If state.response is already set by the last action (e.g. synthesize_final_answer), it will be used.
                    state.next_step = "responder"

        # state.response might have been set by the action itself (e.g. answer_query_directly, synthesize_final_answer)
        # If not, the default "Executing internal step..." will be used for intermediate UI update.
        # The final response for the turn will be set by the responder node based on state.response.
        return state


    def handle_unknown(self, state: BotState) -> BotState:
        tool_name = "handle_unknown"
        # Check if there's already a more specific error message in state.response
        if not state.response or "error" not in str(state.response).lower():
            state.response = "I'm not sure how to process that request. Could you please rephrase it, or provide an OpenAPI specification if you haven't already?"
        
        state.update_scratchpad_reason(tool_name, f"Handling unknown input or situation. Final response to be: {state.response}")
        state.next_step = "responder"
        return state

    def handle_loop(self, state: BotState) -> BotState:
        tool_name = "handle_loop"
        state.response = "It seems we're stuck in a processing loop. Please try rephrasing your request or starting over with the OpenAPI specification."
        state.loop_counter = 0 # Reset loop counter
        state.update_scratchpad_reason(tool_name, "Loop detected, routing to responder with a loop message.")
        state.next_step = "responder"
        return state

    # Method to setup workflow execution (called by interactive_query_executor or router)
    def setup_workflow_execution(self, state: BotState) -> BotState:
        tool_name = "setup_workflow_execution"
        logger.info(f"[{state.session_id}] Setting up workflow execution based on current graph.")
        state.update_scratchpad_reason(tool_name, "Preparing for workflow execution.")

        if not state.execution_graph:
            state.response = "No execution graph is available to run. Please generate or load one first."
            state.workflow_execution_status = "failed"
            state.next_step = "responder"
            return state

        if state.workflow_execution_status == "running" or state.workflow_execution_status == "paused_for_confirmation":
            state.response = "A workflow is already running or paused. Please wait for it to complete or address the pause."
            state.next_step = "responder"
            return state

        try:
            # Initial data for the workflow might come from user input or be pre-configured.
            # For now, start with any data already in state.workflow_extracted_data
            initial_data_for_workflow = state.workflow_extracted_data.copy()
            # Potentially, user could provide initial parameters via user_input, parse them here.

            # The actual websocket_callback function will be supplied by main.py when it
            # retrieves this executor and calls run_workflow_streaming.
            workflow_executor = WorkflowExecutor(
                workflow_definition=state.execution_graph,
                api_executor_instance=self.api_executor, # Assumes self.api_executor is initialized
                websocket_callback=None, # Placeholder; main.py will set the actual callback
                session_id=state.session_id,
                initial_extracted_data=initial_data_for_workflow
            )
            # Store the executor instance in scratchpad so main.py can access it
            state.scratchpad['workflow_executor_instance'] = workflow_executor
            
            state.workflow_execution_status = "pending_start" # Signal to main.py to start streaming
            state.response = (
                "Workflow execution has been initialized. "
                "The system will now attempt to start running the defined API calls. "
                "You will receive updates on its progress."
            )
            logger.info(f"[{state.session_id}] WorkflowExecutor instance created and stored in scratchpad. Status set to 'pending_start'.")

        except Exception as e:
            logger.error(f"[{state.session_id}] Error initializing WorkflowExecutor: {e}", exc_info=True)
            state.response = f"Critical error setting up workflow execution: {str(e)[:150]}"
            state.workflow_execution_status = "failed"
            state.scratchpad.pop('workflow_executor_instance', None) # Clean up

        state.next_step = "responder" # The main graph responds; main.py will handle the actual start
        return state

    # Method to resume workflow (called by interactive_query_executor or router)
    def resume_workflow_with_payload(self, state: BotState, confirmed_payload: Dict[str, Any]) -> BotState:
        tool_name = "resume_workflow_with_payload"
        logger.info(f"[{state.session_id}] Attempting to resume workflow with confirmed_payload.")
        state.update_scratchpad_reason(tool_name, f"Received payload for workflow resumption: {str(confirmed_payload)[:100]}...")

        executor = state.scratchpad.get('workflow_executor_instance')
        if not executor or not isinstance(executor, WorkflowExecutor):
            state.response = "No active workflow executor found to resume. Please try setting up the workflow execution again."
            state.workflow_execution_status = "failed" # Or "idle"
            state.next_step = "responder"
            return state

        if state.workflow_execution_status != "paused_for_confirmation":
            state.response = (
                f"Workflow is not currently paused for confirmation (current status: {state.workflow_execution_status}). "
                "Cannot resume with payload at this time."
            )
            state.next_step = "responder"
            return state

        try:
            # The submit_interrupt_value method of WorkflowExecutor is async
            # It should be called by main.py, which has access to the asyncio loop
            # For now, we'll assume this method can be called and it queues the value.
            # The actual running task `run_workflow_streaming` will pick it up.
            
            # We need to ensure this is called in an async context if submit_interrupt_value is async.
            # Since this is a synchronous LangGraph node, we create a task.
            async def _submit_to_executor_task():
                await executor.submit_interrupt_value(confirmed_payload)

            asyncio.create_task(_submit_to_executor_task())
            
            state.workflow_execution_status = "running" # Set status to running as it will resume
            state.response = "Confirmation payload received. Workflow is attempting to resume execution."
            logger.info(f"[{state.session_id}] Confirmed payload submitted to WorkflowExecutor's queue. Workflow status set to 'running'.")

        except Exception as e:
            logger.error(f"[{state.session_id}] Error submitting confirmed payload to WorkflowExecutor: {e}", exc_info=True)
            state.response = f"Error resuming workflow with payload: {str(e)[:150]}"
            # Optionally, set workflow_execution_status to "failed" if submission fails critically

        state.next_step = "responder"
        return state
