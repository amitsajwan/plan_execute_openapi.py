# core_logic.py
import json
import logging
from typing import Any, Dict, List, Optional
import yaml

from models import BotState, GraphOutput, Node, InputMapping # Edge is part of GraphOutput
from utils import (
    llm_call_helper, load_cached_schema, save_schema_to_cache,
    get_cache_key, check_for_cycles, parse_llm_json_output_with_model,
    SCHEMA_CACHE 
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class OpenAPICoreLogic:
    """
    Handles core OpenAPI processing: parsing, summarization, API identification,
    payload description generation, and API execution graph/plan generation and refinement.
    Also handles interactive query planning and execution of internal actions.
    Designed for use as nodes in a LangGraph agent.
    """
    def __init__(self, worker_llm: Any):
        if not hasattr(worker_llm, 'invoke'):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("OpenAPICoreLogic initialized.")

    def parse_openapi_spec(self, state: BotState) -> BotState:
        """Parses raw OpenAPI spec (JSON/YAML), loads from cache if possible."""
        tool_name = "parse_openapi_spec"
        state.update_scratchpad_reason(tool_name, "Attempting to parse OpenAPI spec.")
        logger.info("Executing parse_openapi_spec node.")

        spec_text = state.openapi_spec_string
        if not spec_text:
            state.response = "No OpenAPI specification text provided."
            state.update_scratchpad_reason(tool_name, "No spec text in state.")
            state.next_step = "responder"
            state.openapi_spec_string = None
            return state

        cache_key = get_cache_key(spec_text)
        cached_schema = load_cached_schema(cache_key) 

        if cached_schema:
            logger.info(f"Loaded schema from cache for key: {cache_key}")
            state.openapi_schema = cached_schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text
            state.openapi_spec_string = None
            
            if not state.schema_summary or not state.identified_apis or not state.payload_descriptions or not state.execution_graph:
                logger.info("Cached schema loaded, but some derived artifacts missing. Proceeding to pipeline.")
                state.response = "OpenAPI specification loaded from cache. Completing analysis..."
                state.update_scratchpad_reason(tool_name, "Schema from cache, but derived artifacts missing. Running full pipeline.")
                state.next_step = "process_schema_pipeline"
            else:
                state.response = "OpenAPI specification and all derived artifacts loaded from cache."
                state.update_scratchpad_reason(tool_name, "Schema and all artifacts loaded from cache.")
                state.next_step = "responder"
            return state

        logger.info("Parsing new OpenAPI spec.")
        parsed_schema = None
        error_message = None
        try:
            parsed_schema = json.loads(spec_text)
            logger.debug("Successfully parsed spec as JSON.")
        except json.JSONDecodeError as json_e:
            logger.debug(f"JSON parsing failed: {json_e}. Attempting YAML.")
            try:
                parsed_schema = yaml.safe_load(spec_text)
                logger.debug("Successfully parsed spec as YAML.")
            except yaml.YAMLError as yaml_e:
                error_message = f"YAML parsing failed: {yaml_e}"
            except Exception as e_yaml: 
                error_message = f"Unexpected error during YAML parsing: {e_yaml}"
        except Exception as e_json: 
             error_message = f"Unexpected error during JSON parsing: {e_json}"


        if parsed_schema and isinstance(parsed_schema, dict) and \
           ('openapi' in parsed_schema or 'swagger' in parsed_schema) and 'info' in parsed_schema:
            state.openapi_schema = parsed_schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text
            state.openapi_spec_string = None
            if SCHEMA_CACHE: 
                save_schema_to_cache(cache_key, parsed_schema) 
            else:
                logger.warning("SCHEMA_CACHE is None in core_logic.py, schema not saved to disk cache.")

            logger.info("Successfully parsed and cached OpenAPI spec.")
            state.response = "OpenAPI specification parsed. Analyzing..."
            state.update_scratchpad_reason(tool_name, "Schema parsed. Proceeding with pipeline.")
            state.next_step = "process_schema_pipeline"
        else:
            state.openapi_schema = None
            state.openapi_spec_string = None
            final_error = error_message or "Parsed content is not a valid OpenAPI/Swagger spec (missing key fields like 'openapi'/'swagger' or 'info')."
            state.response = f"Failed to parse specification: {final_error}"
            state.update_scratchpad_reason(tool_name, f"Parsing failed: {final_error}")
            logger.error(f"Parsing failed: {final_error}")
            state.next_step = "responder"
        return state

    def _generate_llm_schema_summary(self, state: BotState):
        tool_name = "_generate_llm_schema_summary"
        state.update_scratchpad_reason(tool_name, "Generating schema summary.")
        if not state.openapi_schema:
            logger.warning("No schema to summarize.")
            state.schema_summary = "Could not generate summary: No schema loaded."
            return

        spec = state.openapi_schema
        info = spec.get('info', {})
        title = info.get('title', 'N/A')
        version = info.get('version', 'N/A')
        description = info.get('description', 'No description.')
        paths_count = len(spec.get('paths', {}))
        
        paths_preview = "\n".join([f"  {p}: {list(m.keys()) if isinstance(m, dict) else '[Invalid methods]'}" for p, m in list(spec.get('paths', {}).items())[:5]])

        summary_prompt = f"""
        Summarize this OpenAPI specification:
        Title: {title}, Version: {version}
        Description: {description[:1000]}{'...' if len(description) > 1000 else ''}
        Total Paths: {paths_count}
        Paths Preview (first 5):
        {paths_preview or "No paths defined."}

        Focus on:
        1. Overall purpose and main capabilities.
        2. Key resource categories and common operations.
        3. Authentication methods mentioned (if any in `components.securitySchemes` or descriptions).
        4. Notable features (e.g., pagination, rate limits if obvious).
        Keep the summary concise (around 200-300 words).
        """
        try:
            state.schema_summary = llm_call_helper(self.worker_llm, summary_prompt)
            logger.info("Schema summary generated.")
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}", exc_info=True)
            state.schema_summary = f"Error generating summary: {e}"
        state.update_scratchpad_reason(tool_name, f"Schema summary generation status: {'Success' if state.schema_summary and not state.schema_summary.startswith('Error') else 'Failed'}")


    def _identify_apis_from_schema(self, state: BotState):
        tool_name = "_identify_apis_from_schema"
        state.update_scratchpad_reason(tool_name, "Identifying APIs from schema.")
        if not state.openapi_schema:
            logger.warning("No schema to identify APIs from.")
            state.identified_apis = []
            return

        apis = []
        paths = state.openapi_schema.get('paths', {})
        for path, path_item in paths.items():
            if not isinstance(path_item, dict): continue
            for method, op_details in path_item.items():
                if method.lower() not in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'trace'] or not isinstance(op_details, dict):
                    continue
                # Ensure operationId is somewhat unique if missing, by including method and more path detail
                op_id_path_part = path.replace('/', '_').replace('{', '').replace('}', '').strip('_')
                default_op_id = f"{method.lower()}_{op_id_path_part if op_id_path_part else 'root'}"

                apis.append({
                    'operationId': op_details.get('operationId', default_op_id),
                    'path': path,
                    'method': method.upper(),
                    'summary': op_details.get('summary', ''),
                    'description': op_details.get('description', ''),
                    'parameters': op_details.get('parameters', []),
                    'requestBody': op_details.get('requestBody', {}),
                    'responses': op_details.get('responses', {}) 
                })
        state.identified_apis = apis
        logger.info(f"Identified {len(apis)} API operations.")
        state.update_scratchpad_reason(tool_name, f"Identified {len(apis)} APIs.")

    def _generate_payload_descriptions(self, state: BotState, target_apis: Optional[List[str]] = None, context_override: Optional[str] = None):
        tool_name = "_generate_payload_descriptions"
        state.update_scratchpad_reason(tool_name, f"Generating payload descriptions. Target APIs: {target_apis or 'All'}. Context Override: {bool(context_override)}")
        if not state.identified_apis:
            logger.warning("No identified APIs for payload descriptions.")
            return

        payload_descs = state.payload_descriptions or {}
        apis_to_process = [api for api in state.identified_apis if (target_apis is None or api['operationId'] in target_apis)]

        for api in apis_to_process[:30]: 
            op_id = api['operationId']
            if op_id in payload_descs and not context_override:
                continue

            param_details = json.dumps(api.get('parameters', []), indent=2)[:500]
            body_details = json.dumps(api.get('requestBody', {}), indent=2)[:500]
            success_response_schema = "No specific 2xx response schema example available."
            responses = api.get('responses', {})
            for code, resp_def in responses.items():
                if code.startswith('2') and isinstance(resp_def, dict) and 'content' in resp_def:
                    json_content = resp_def['content'].get('application/json', {})
                    if 'schema' in json_content:
                        success_response_schema = json.dumps(json_content['schema'], indent=2)[:500] + "..."
                        break
            
            context_instruction = f"\nIMPORTANT CONTEXT: {context_override}" if context_override else ""

            prompt = f"""
            For API operation '{op_id}' ({api['method']} {api['path']}), described as: "{api['summary']}".
            Parameters: {param_details}
            Request Body: {body_details}
            Example Successful Response Schema (if available): {success_response_schema}
            {context_instruction}

            Generate a concise, natural language description of:
            1. A typical request: Mention key parameters/body fields, their types, and example values. If {context_override}, tailor examples to it.
            2. A typical successful response: Describe its structure and key fields with example values.
            Focus on clarity for a developer. Be brief. If no parameters/body, state that.
            Output format:
            Request: [Description of request]
            Response: [Description of response]
            """
            try:
                desc = llm_call_helper(self.worker_llm, prompt)
                payload_descs[op_id] = desc
                logger.debug(f"Generated payload/response description for {op_id}")
            except Exception as e:
                logger.error(f"Error generating payload/response description for {op_id}: {e}", exc_info=True)
                payload_descs[op_id] = f"Error generating description: {e}"
        state.payload_descriptions = payload_descs
        state.update_scratchpad_reason(tool_name, f"Payload descriptions updated for {len(apis_to_process[:30])} targeted APIs.")


    def _generate_execution_graph(self, state: BotState, goal: Optional[str] = None) -> BotState:
        tool_name = "_generate_execution_graph"
        state.update_scratchpad_reason(tool_name, f"Generating execution graph. Goal: {goal or 'General overview'}")
        logger.info(f"Generating execution graph. Goal: {goal or 'General overview'}")

        if not state.identified_apis:
            state.response = "Cannot generate graph: No identified APIs."
            state.update_scratchpad_reason(tool_name, "No APIs to generate graph from.")
            state.next_step = "responder" # Route to responder to deliver this message
            return state

        api_details_for_prompt = "\n".join([
            f"- opId: {api['operationId']}, method: {api['method']}, path: {api['path']}, summary: {api['summary']}"
            for api in state.identified_apis[:50] 
        ])
        
        current_goal = goal or state.plan_generation_goal or "Provide a general, illustrative workflow using a few key APIs."
        regeneration_feedback = f"Previous attempt feedback: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else ""

        graph_prompt = f"""
        You are an AI API workflow planner. Based on the user's goal and available APIs, design an API execution graph.
        User Goal: "{current_goal}"
        {regeneration_feedback}

        Available APIs (partial list):
        {api_details_for_prompt}
        {'... (more APIs available)' if len(state.identified_apis) > 50 else ''}

        Output a JSON object matching the `GraphOutput` model (see structure below).
        - Select 3-7 relevant API operations for a plausible workflow.
        - Define `nodes` (API calls) and `edges` (dependencies).
        - For each node, attempt to define `input_mappings` if it depends on data from a previous node:
          - `source_operation_id`: `effective_id` of the source node.
          - `source_data_path`: Plausible JSONPath (e.g., `$.id`, `$.data.token`) from source's typical response.
          - `target_parameter_name`: Parameter name in the current node.
          - `target_parameter_in`: Location (e.g., 'path', 'query', 'body.fieldName').
        - Provide an overall `description` of the workflow.
        - Use `display_name` for nodes if an `operationId` is used multiple times.
        - Ensure the graph is a DAG.
        - CRITICAL: All `operationId` or `display_name` values used in `edges` (for `from_node` and `to_node`) MUST correspond to an `operationId` or `display_name` of a node defined in the `nodes` list of THIS SAME JSON output.

        Pydantic Model Structure (JSON):
        {{
          "nodes": [
            {{
              "operationId": "string", "display_name": "string" (optional), "summary": "string",
              "description": "string (purpose in this workflow)",
              "payload_description": "string (brief example request/response for this step)",
              "input_mappings": [
                {{"source_operation_id": "string", "source_data_path": "string", "target_parameter_name": "string", "target_parameter_in": "string"}}
              ]
            }}
          ],
          "edges": [ {{"from_node": "string", "to_node": "string", "description": "string"}} ],
          "description": "string (overall workflow description)"
        }}
        Output ONLY the JSON object.
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, graph_prompt)
            graph_output = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output:
                state.execution_graph = graph_output
                state.response = "API execution graph generated."
                state.graph_regeneration_reason = None 
                state.graph_refinement_iterations = 0 
                state.next_step = "verify_graph"
            else:
                logger.error("LLM failed to produce valid GraphOutput JSON for execution graph.")
                state.response = "Failed to generate a valid execution graph (AI format error)."
                state.graph_regeneration_reason = "LLM output format was incorrect."
                if state.scratchpad.get('graph_gen_attempts', 0) < 1:
                    state.scratchpad['graph_gen_attempts'] = state.scratchpad.get('graph_gen_attempts', 0) + 1
                    state.next_step = "_generate_execution_graph" 
                else:
                    state.next_step = "handle_unknown" 
                    state.scratchpad['graph_gen_attempts'] = 0 
        except Exception as e:
            logger.error(f"Error generating execution graph: {e}", exc_info=True)
            state.response = f"Error generating graph: {e}"
            state.graph_regeneration_reason = f"Error: {e}"
            state.next_step = "handle_unknown"
        state.update_scratchpad_reason(tool_name, f"Graph generation status: {'Success' if state.execution_graph else 'Failed'}. Response: {state.response}")
        return state

    def process_schema_pipeline(self, state: BotState) -> BotState:
        tool_name = "process_schema_pipeline"
        state.update_scratchpad_reason(tool_name, "Starting schema processing pipeline.")
        logger.info("Executing process_schema_pipeline.")

        if not state.openapi_schema:
            state.response = "Cannot run pipeline: No OpenAPI schema loaded."
            state.update_scratchpad_reason(tool_name, "No schema to process.")
            state.next_step = "handle_unknown"
            return state
        
        state.schema_summary = None
        state.identified_apis = []
        state.payload_descriptions = {}
        state.execution_graph = None
        state.plan_generation_goal = state.plan_generation_goal or "Provide a general overview workflow." 
        state.graph_refinement_iterations = 0
        state.scratchpad['graph_gen_attempts'] = 0

        self._generate_llm_schema_summary(state)
        self._identify_apis_from_schema(state)
        if state.identified_apis: 
            self._generate_payload_descriptions(state) 
        else: # No APIs identified, so can't generate graph or payloads
            state.response = state.response or "" + " No API operations were identified from the spec. Cannot generate payload descriptions or an execution graph."
            state.update_scratchpad_reason(tool_name, "No APIs identified. Skipping payload and graph generation.")
            state.next_step = "responder" # Go to responder to deliver this message
            return state


        self._generate_execution_graph(state, goal=state.plan_generation_goal) 

        state.update_scratchpad_reason(tool_name, f"Schema pipeline processing initiated. Next step set to: {state.next_step}")
        return state

    def verify_graph(self, state: BotState) -> BotState:
        tool_name = "verify_graph"
        state.update_scratchpad_reason(tool_name, "Verifying execution graph.")
        logger.info("Executing verify_graph node.")

        if not state.execution_graph:
            state.response = "No execution graph to verify."
            state.graph_regeneration_reason = "No graph was generated to verify."
            state.next_step = "_generate_execution_graph" 
            return state

        is_dag, cycle_message = check_for_cycles(state.execution_graph)

        if is_dag: 
            state.response = "Graph verification successful: No cycles found and structure seems valid."
            state.update_scratchpad_reason(tool_name, "Graph verification successful.")
            logger.info("Graph verification successful.")
            state.graph_regeneration_reason = None 

            if state.graph_refinement_iterations < state.max_refinement_iterations:
                state.next_step = "refine_api_graph"
            else: 
                if state.input_is_spec: 
                    api_title = state.openapi_schema.get('info', {}).get('title', 'the API') # type: ignore
                    state.response = (
                        f"Successfully processed '{api_title}'. Identified {len(state.identified_apis)} APIs, "
                        f"generated payload examples, and created an API workflow graph with {len(state.execution_graph.nodes)} steps. "
                        "You can now ask questions or request specific plan refinements."
                    )
                    state.input_is_spec = False 
                    state.next_step = "responder"
                else: 
                    state.response = "Graph verification successful. " + (state.execution_graph.refinement_summary or "")
                    state.next_step = "describe_graph" 
        else:
            state.response = f"Graph verification failed: {cycle_message}. "
            state.graph_regeneration_reason = f"Verification failed: {cycle_message}."
            state.update_scratchpad_reason(tool_name, f"Graph verification failed: {cycle_message}.")
            logger.warning(f"Graph verification failed: {cycle_message}. Routing to refine/regenerate.")
            if state.graph_refinement_iterations < state.max_refinement_iterations :
                 state.next_step = "refine_api_graph" 
            else:
                 state.next_step = "_generate_execution_graph" 
        return state

    def refine_api_graph(self, state: BotState) -> BotState:
        tool_name = "refine_api_graph"
        iteration = state.graph_refinement_iterations + 1
        state.update_scratchpad_reason(tool_name, f"Refining API graph. Iteration: {iteration}")
        logger.info(f"Executing refine_api_graph node. Iteration: {iteration}")

        if not state.execution_graph:
            state.response = "No graph to refine. Please generate a graph first."
            state.next_step = "_generate_execution_graph"
            return state

        if iteration > state.max_refinement_iterations:
            state.response = f"Max refinement iterations ({state.max_refinement_iterations}) reached. Using current graph: {state.execution_graph.description if state.execution_graph else 'No graph available.'}"
            state.next_step = "describe_graph" 
            return state

        current_graph_json = state.execution_graph.model_dump_json(indent=2)
        api_context_for_refinement = "\n".join([
            f"- opId: {api['operationId']}, summary: {api['summary']}"
            for api in state.identified_apis[:30] 
        ])

        refinement_prompt = f"""
        You are an AI API workflow design critic and optimizer.
        Your task is to critique and **output a fully valid, self-contained, and improved JSON representation** of an API execution graph.

        User's Goal: "{state.plan_generation_goal or 'General workflow overview.'}"
        Previous Attempt Feedback (if any): {state.graph_regeneration_reason or "N/A. This might be the first refinement attempt or previous attempts were structurally okay."}

        Current Graph (JSON to be refined):
        ```json
        {current_graph_json}
        ```
        Available API Operations (for context and potential additions/substitutions during your refinement):
        {api_context_for_refinement}

        **CRITICAL Output Requirements for the REFINED Graph JSON:**
        1.  **Self-Contained Node Definitions:** ALL nodes referenced in the `edges` (in `from_node` or `to_node`) MUST be explicitly defined with their `operationId` (and optional `display_name`) in the `nodes` list OF THE SAME JSON OUTPUT YOU ARE GENERATING. Do not reference nodes that you haven't included in your `nodes` list.
        2.  **Valid Edges:** Ensure `from_node` and `to_node` in each edge correctly use the `operationId` (or `display_name` if `operationId` is reused for different steps) of nodes present in YOUR `nodes` list.
        3.  **DAG Structure:** The graph must be a Directed Acyclic Graph (no circular dependencies).
        4.  **Pydantic Model Adherence:** The entire output must be a single JSON object strictly matching the `GraphOutput` model structure provided below.

        Refinement Focus Areas:
        A.  **Goal Alignment:** Does the graph effectively achieve the user's goal? Are there any redundant or missing steps (nodes)?
        B.  **Logical Flow & Correctness:** Is the sequence of API calls logical? Are dependencies (edges) correct and meaningful?
        C.  **Data Flow & `input_mappings` (VERY IMPORTANT):**
            * Review each `input_mappings` entry.
            * Is `source_data_path` plausible for extracting data from the typical response of `source_operation_id` (e.g., `$.id`, `$.data.token`, `$.items[0].userId`)?
            * Are `target_parameter_name` and `target_parameter_in` correct for the target node?
            * **Suggest more precise `source_data_path` if they are generic or missing. Add mappings if essential data flow is missing.**
        D.  **Clarity & Detail:** Improve overall graph `description`, node `summary`/`description`, and edge `description` for clarity.
        E.  **Completeness:** Ensure node `payload_description` fields are adequate.

        Output Instructions:
        -   Return a SINGLE REVISED JSON object representing the `GraphOutput` model.
        -   Include a "refinement_summary" field (string) at the root of your JSON, explaining your key changes or why no changes were needed.
        -   If the current graph is already optimal and requires no changes based on the feedback and your critique, state this in "refinement_summary" and return the original graph structure within the required JSON format.

        Pydantic Model Structure (for your output JSON):
        {{
          "nodes": [
            {{
              "operationId": "string", "display_name": "string" (optional, use if same operationId is used for multiple steps), 
              "summary": "string", "description": "string (purpose of this node in THIS workflow)",
              "payload_description": "string (brief example request/response for this specific step)",
              "input_mappings": [
                {{"source_operation_id": "string" /* effective_id of source */, "source_data_path": "string", "target_parameter_name": "string", "target_parameter_in": "string" /* path, query, body.fieldName etc. */}}
              ]
            }}
          ],
          "edges": [ 
            {{"from_node": "string" /* effective_id of an existing node in YOUR nodes list */, "to_node": "string" /* effective_id of an existing node in YOUR nodes list */, "description": "string"}} 
          ],
          "description": "string (overall workflow description)",
          "refinement_summary": "string" 
        }}
        Ensure your entire response is ONLY this JSON object.
        """
        try:
            llm_response_str = llm_call_helper(self.worker_llm, refinement_prompt)
            raw_output_dict = parse_llm_json_output_with_model(llm_response_str)

            if not raw_output_dict or not isinstance(raw_output_dict, dict):
                raise ValueError("LLM refinement output was not valid JSON dictionary.")

            refinement_summary = raw_output_dict.pop("refinement_summary", "No specific summary from AI.")
            state.update_scratchpad_reason(tool_name, f"LLM Refinement Summary: {refinement_summary}")

            refined_graph = GraphOutput.model_validate(raw_output_dict) 
            
            state.execution_graph = refined_graph # Update with the new, validated graph
            state.graph_refinement_iterations = iteration
            state.response = f"Graph refined (Iteration {iteration}). Summary: {refinement_summary}"
            state.graph_regeneration_reason = None 
            state.next_step = "verify_graph" 

        except (ValueError, ValidationError) as e: 
            logger.error(f"Error processing refined graph (iteration {iteration}): {e}", exc_info=True)
            # Keep the PREVIOUS valid graph if refinement fails validation.
            # state.execution_graph remains unchanged from before this failed refinement attempt.
            state.response = f"Error during graph refinement (iteration {iteration}): {str(e)[:200]}. Using previous graph version if available."
            state.graph_regeneration_reason = f"Refinement error (iteration {iteration}): {str(e)[:200]}" 
            
            if iteration < state.max_refinement_iterations: # Check against iteration, not state.graph_refinement_iterations
                 state.next_step = "refine_api_graph" # Retry refinement with feedback
            else:
                 logger.warning(f"Max refinement iterations reached after error. Describing last valid graph (if any).")
                 state.next_step = "describe_graph" 
        except Exception as e: 
            logger.error(f"Unexpected error during graph refinement (iteration {iteration}): {e}", exc_info=True)
            state.response = f"Unexpected error refining graph (iteration {iteration}): {e}. Using previous graph version."
            state.graph_regeneration_reason = f"Unexpected refinement error (iteration {iteration}): {e}"
            state.next_step = "describe_graph" # Fallback to describing whatever graph state we have
        return state

    def describe_graph(self, state: BotState) -> BotState:
        tool_name = "describe_graph"
        state.update_scratchpad_reason(tool_name, "Describing execution graph.")
        logger.info("Executing describe_graph node.")
        if not state.execution_graph:
            state.response = "No execution graph is currently available to describe."
            logger.warning("describe_graph: No execution_graph found in state.")
        else:
            graph_desc = state.execution_graph.description
            if not graph_desc or len(graph_desc) < 20: # If too short or missing, try to generate
                nodes_summary = "\n".join([f"- {node.effective_id}: {node.summary or node.operationId}" for node in state.execution_graph.nodes[:5]])
                prompt = f"""
                Describe the following API execution graph workflow in natural language.
                Overall Goal: {state.plan_generation_goal or "General API workflow"}
                Nodes (first few):
                {nodes_summary}
                Number of nodes: {len(state.execution_graph.nodes)}, Number of edges: {len(state.execution_graph.edges)}
                Explain its purpose and how the main steps connect. Be concise.
                """
                try:
                    generated_desc = llm_call_helper(self.worker_llm, prompt)
                    # If graph_desc was empty, use generated. If it had something, append.
                    graph_desc = generated_desc if not graph_desc else f"{graph_desc}\nFurther details: {generated_desc}"
                except Exception as e:
                    logger.error(f"Error generating dynamic graph description: {e}")
                    # Fallback to ensure state.response is set
                    graph_desc = state.execution_graph.description or f"Could not generate dynamic description. Stored graph description: Not available or too brief."
            
            state.response = f"Current API Workflow Graph for '{state.plan_generation_goal or 'general use'}':\n{graph_desc}"
            if state.execution_graph.refinement_summary: # Add refinement summary if it exists
                state.response += f"\nLast Refinement: {state.execution_graph.refinement_summary}"
            logger.info(f"describe_graph: Set state.response to: {state.response[:200]}...")

        state.next_step = "responder"
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        tool_name = "get_graph_json"
        state.update_scratchpad_reason(tool_name, "Getting graph JSON.")
        if not state.execution_graph:
            state.response = "No execution graph found."
        else:
            try:
                graph_json = state.execution_graph.model_dump_json(indent=2)
                state.response = f"Current Execution Graph JSON:\n```json\n{graph_json}\n```"
            except Exception as e:
                state.response = f"Error serializing graph to JSON: {e}"
        state.next_step = "responder"
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        tool_name = "answer_openapi_query"
        state.update_scratchpad_reason(tool_name, "Answering general OpenAPI query.")
        logger.info("Executing answer_openapi_query node.")

        user_input = state.user_input
        if not state.openapi_schema:
            state.response = "No OpenAPI spec loaded. Please provide one first."
            state.next_step = "responder"
            return state

        context_parts = [
            f"User Question: \"{user_input}\"",
            f"API Summary: {state.schema_summary or 'Not available.'}",
            f"Identified APIs ({len(state.identified_apis)} total): " + ", ".join([api['operationId'] for api in state.identified_apis[:10]]) + "...",
            f"Current Graph Goal: {state.plan_generation_goal or 'General overview'}",
            f"Graph Description: {state.execution_graph.description if state.execution_graph else 'Not available.'}",
        ]
        if user_input and any(op_id in user_input for op_id in state.payload_descriptions): # type: ignore
            for op_id, desc in state.payload_descriptions.items():
                if op_id in user_input: # type: ignore
                    context_parts.append(f"Payload/Response for '{op_id}': {desc[:300]}...")
                    break
        
        prompt = "\n\n".join(context_parts) + "\n\nAnswer the user's question based on the provided API information and graph context. Be concise and factual. If information is missing, state that."

        try:
            state.response = llm_call_helper(self.worker_llm, prompt)
        except Exception as e:
            state.response = f"Error answering query: {e}"
        state.next_step = "responder"
        return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        tool_name = "interactive_query_planner"
        state.update_scratchpad_reason(tool_name, f"Planning internal actions for user query: {state.user_input}")
        logger.info(f"Executing {tool_name} for query: {state.user_input}")

        state.scratchpad.pop('interactive_action_plan', None)
        state.scratchpad.pop('current_interactive_action_idx', None)
        state.scratchpad.pop('current_interactive_results', None)

        graph_summary = state.execution_graph.description if state.execution_graph else "No graph loaded."
        payload_keys = list(state.payload_descriptions.keys())[:5]

        planner_prompt = f"""
        You are an AI assistant that plans internal actions to respond to a user's query about a loaded OpenAPI spec and its derived artifacts (summary, API list, payload examples, execution graph).

        User Query: "{state.user_input}"

        Current State Context:
        - API Spec Summary: {state.schema_summary[:300] if state.schema_summary else 'N/A'}...
        - Identified APIs (first 5 opIds): {", ".join([api['operationId'] for api in state.identified_apis[:5]])}...
        - Example Payload Descriptions available for (first 5 opIds): {payload_keys}...
        - Current Execution Graph Goal: {state.plan_generation_goal or 'N/A'}
        - Current Graph Description: {graph_summary[:300]}...

        Available Internal Actions (choose one or more in sequence):
        1.  `rerun_payload_generation`: Regenerate payload/response examples for specific APIs with new context.
            Params: {{ "operation_ids_to_update": ["opId1", "opId2"], "new_context": "User's new context string" }}
        2.  `contextualize_graph_descriptions`: Rewrite descriptions within the *existing* graph (overall, nodes, edges) to reflect new user context.
            Params: {{ "target_node_operation_ids": ["opId1"], "new_context": "User's new context string" }}
        3.  `regenerate_graph_with_new_goal`: Create a *new* graph if the user states a completely different high-level goal.
            Params: {{ "new_goal_string": "User's new goal for the graph" }}
        4.  `refine_existing_graph_further`: Trigger another LLM refinement cycle on the current graph based on user feedback.
            Params: {{ "refinement_instructions": "User's specific feedback for refinement" }}
        5.  `answer_query_directly`: If the query can be answered using existing information without modifications.
            Params: {{ "query_for_synthesizer": "The original user query or a rephrased one for direct answering." }}
        6.  `synthesize_final_answer`: (Usually the last step) Formulate a comprehensive answer to the user based on the outcomes of previous internal actions.
            Params: {{ "synthesis_prompt_instructions": "Instructions for the LLM on what to include in the final answer." }}

        Task:
        1. Understand the user's query.
        2. Create a short, logical "interactive_action_plan" (a list of action objects) to address it.
        3. If the query is simple, the plan might just be one "answer_query_directly" or "synthesize_final_answer" action.
        4. Provide a brief "user_query_understanding".

        Output ONLY a JSON object with this structure:
        {{
          "user_query_understanding": "Brief interpretation of user's need.",
          "interactive_action_plan": [
            {{"action_name": "action_enum_value", "action_params": {{...}}, "description": "Why this action."}}
          ]
        }}
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, planner_prompt)
            parsed_plan_data = parse_llm_json_output_with_model(llm_response)

            if parsed_plan_data and "interactive_action_plan" in parsed_plan_data and \
               isinstance(parsed_plan_data["interactive_action_plan"], list):
                state.scratchpad['interactive_action_plan'] = parsed_plan_data["interactive_action_plan"]
                state.scratchpad['current_interactive_action_idx'] = 0
                state.scratchpad['current_interactive_results'] = []
                state.scratchpad['user_query_understanding'] = parsed_plan_data.get("user_query_understanding", "N/A")
                state.response = f"Understood query: {state.scratchpad['user_query_understanding']}. Processing..."
                state.next_step = "interactive_query_executor"
                logger.info(f"Interactive plan generated with {len(parsed_plan_data['interactive_action_plan'])} steps.")
            else:
                raise ValueError("LLM failed to produce a valid interactive action plan structure.")
        except Exception as e:
            logger.error(f"Error in interactive_query_planner: {e}", exc_info=True)
            state.response = f"Sorry, I had trouble planning how to address your request: {e}. Try rephrasing?"
            state.next_step = "responder" 
        return state

    def _internal_rerun_payload_generation(self, state: BotState, operation_ids_to_update: List[str], new_context: str) -> str:
        logger.info(f"Internal: Rerunning payload generation for {operation_ids_to_update} with context: {new_context}")
        self._generate_payload_descriptions(state, target_apis=operation_ids_to_update, context_override=new_context)
        updated_payloads = {op_id: state.payload_descriptions.get(op_id) for op_id in operation_ids_to_update if op_id in state.payload_descriptions}
        return f"Payload descriptions updated for {len(updated_payloads)} APIs with new context. Examples: {str(list(updated_payloads.values())[:1])[:100]}..."


    def _internal_contextualize_graph(self, state: BotState, target_node_op_ids: Optional[List[str]], new_context: str) -> str:
        if not state.execution_graph: return "No graph to contextualize."
        logger.info(f"Internal: Contextualizing graph descriptions for nodes {target_node_op_ids or 'all'} with context: {new_context}")
        
        changed_nodes_count = 0
        
        prompt_overall = f"""
        Given the new context: "{new_context}"
        And the current graph description: "{state.execution_graph.description}"
        Rewrite the graph description to incorporate this new context. Keep it concise.
        New Description:
        """
        try:
            if state.execution_graph: 
                 state.execution_graph.description = llm_call_helper(self.worker_llm, prompt_overall)
                 changed_nodes_count +=1 
        except Exception as e:
            logger.error(f"Failed to update overall graph description with context: {e}")

        if target_node_op_ids and state.execution_graph: 
            for node in state.execution_graph.nodes:
                if node.effective_id in target_node_op_ids:
                    prompt_node = f"""
                    Given the new context: "{new_context}"
                    And the current node '{node.effective_id}' description: "{node.description}"
                    And its summary: "{node.summary}"
                    Rewrite the node's description and summary to incorporate this new context. Be concise.
                    Output JSON: {{"new_description": "...", "new_summary": "..."}}
                    """
                    try:
                        resp = llm_call_helper(self.worker_llm, prompt_node)
                        parsed_resp = parse_llm_json_output_with_model(resp)
                        if parsed_resp and isinstance(parsed_resp, dict):
                            node.description = parsed_resp.get("new_description", node.description)
                            node.summary = parsed_resp.get("new_summary", node.summary)
                            changed_nodes_count +=1
                    except Exception as e:
                        logger.error(f"Failed to update node {node.effective_id} description with context: {e}")
        
        return f"Graph descriptions contextualized. {changed_nodes_count} parts updated."

    def _internal_answer_query_directly(self, state: BotState, query_for_synthesizer: str) -> str:
        logger.info(f"Internal: Answering query directly: {query_for_synthesizer}")
        original_user_input = state.user_input
        state.user_input = query_for_synthesizer
        self.answer_openapi_query(state) 
        direct_answer = state.response
        state.user_input = original_user_input 
        state.response = None 
        return direct_answer or "Could not formulate a direct answer."

    def _internal_synthesize_final_answer(self, state: BotState, synthesis_prompt_instructions: str) -> str:
        logger.info(f"Internal: Synthesizing final answer. Instructions: {synthesis_prompt_instructions}")
        
        interactive_results_summary = "\n".join([str(res)[:300]+"..." for res in state.scratchpad.get('current_interactive_results', [])])
        
        prompt = f"""
        Based on the following internal actions and their results, synthesize a final answer for the user.
        User's Original Query Understanding: {state.scratchpad.get('user_query_understanding', 'N/A')}
        Synthesis Instructions: {synthesis_prompt_instructions}

        Summary of Internal Actions Taken and Results:
        {interactive_results_summary or "No specific internal actions taken or results logged."}

        Current Graph Description (if relevant): {state.execution_graph.description if state.execution_graph else "N/A"}
        
        Formulate a concise, helpful, and direct answer to the user.
        """
        try:
            final_answer = llm_call_helper(self.worker_llm, prompt)
            state.response = final_answer 
            return final_answer
        except Exception as e:
            logger.error(f"Error synthesizing final answer: {e}")
            state.response = f"Error synthesizing final answer: {e}"
            return state.response


    def interactive_query_executor(self, state: BotState) -> BotState:
        tool_name = "interactive_query_executor"
        
        action_plan = state.scratchpad.get('interactive_action_plan', [])
        action_idx = state.scratchpad.get('current_interactive_action_idx', 0)
        interactive_results = state.scratchpad.get('current_interactive_results', [])

        if not action_plan or action_idx >= len(action_plan):
            logger.info("Interactive action plan finished or empty.")
            state.update_scratchpad_reason(tool_name, "Interactive plan finished.")
            if not state.response: 
                 state.response = "Finished processing your interactive request. What's next?"
            state.next_step = "responder"
            return state

        action = action_plan[action_idx]
        action_name = action.get("action_name")
        action_params = action.get("action_params", {})
        action_description = action.get("description", "No description")

        state.update_scratchpad_reason(tool_name, f"Executing interactive action ({action_idx+1}/{len(action_plan)}): {action_name} - {action_description}")
        logger.info(f"Executing interactive action: {action_name} with params: {action_params}")
        
        action_result_message = f"Action '{action_name}' executed."
        current_step_response = "" 

        try:
            if action_name == "rerun_payload_generation":
                op_ids = action_params.get("operation_ids_to_update", [])
                context = action_params.get("new_context", "")
                if op_ids and context:
                    action_result_message = self._internal_rerun_payload_generation(state, op_ids, context)
                    current_step_response = f"Updated payload examples for {op_ids} based on '{context[:50]}...'."
                else: action_result_message = "Skipped rerun_payload_generation: missing op_ids or context."

            elif action_name == "contextualize_graph_descriptions":
                target_op_ids = action_params.get("target_node_operation_ids")
                context = action_params.get("new_context", "")
                if context:
                    action_result_message = self._internal_contextualize_graph(state, target_op_ids, context)
                    current_step_response = f"Updated graph descriptions based on '{context[:50]}...'."
                else: action_result_message = "Skipped contextualize_graph_descriptions: missing context."
            
            elif action_name == "regenerate_graph_with_new_goal":
                new_goal = action_params.get("new_goal_string", "")
                if new_goal:
                    state.plan_generation_goal = new_goal
                    state.graph_refinement_iterations = 0 
                    state.execution_graph = None 
                    self._generate_execution_graph(state, goal=new_goal) 
                    action_result_message = f"Started regenerating graph for new goal: {new_goal}"
                    current_step_response = action_result_message
                    interactive_results.append(action_result_message)
                    state.scratchpad['current_interactive_results'] = interactive_results
                    logger.info(f"Interactive action '{action_name}' triggered main graph regeneration. Interactive plan execution will not continue to next step immediately.")
                    return state 
                else: action_result_message = "Skipped regenerate_graph_with_new_goal: missing new_goal_string."

            elif action_name == "refine_existing_graph_further":
                instructions = action_params.get("refinement_instructions", "")
                if state.execution_graph:
                    state.graph_regeneration_reason = instructions 
                    self.refine_api_graph(state) 
                    action_result_message = f"Started further refinement of graph based on: {instructions[:50]}..."
                    current_step_response = action_result_message
                    interactive_results.append(action_result_message)
                    state.scratchpad['current_interactive_results'] = interactive_results
                    logger.info(f"Interactive action '{action_name}' triggered main graph refinement. Interactive plan execution will not continue to next step immediately.")
                    return state 
                else: action_result_message = "Skipped refine_existing_graph_further: no graph to refine."

            elif action_name == "answer_query_directly":
                query = action_params.get("query_for_synthesizer", state.user_input) 
                direct_answer = self._internal_answer_query_directly(state, query or "") 
                action_result_message = f"Direct answer attempt: {direct_answer[:100]}..."
                state.response = direct_answer 

            elif action_name == "synthesize_final_answer":
                instructions = action_params.get("synthesis_prompt_instructions", "Summarize findings for the user.")
                final_answer = self._internal_synthesize_final_answer(state, instructions)
                action_result_message = f"Final answer synthesized: {final_answer[:100]}..."
                state.response = final_answer 
            
            else:
                action_result_message = f"Unknown interactive action: {action_name}. Skipped."
                logger.warning(action_result_message)

        except Exception as e:
            logger.error(f"Error executing interactive action {action_name}: {e}", exc_info=True)
            action_result_message = f"Error in action '{action_name}': {e}"
        
        interactive_results.append(action_result_message)
        state.scratchpad['current_interactive_results'] = interactive_results
        state.scratchpad['current_interactive_action_idx'] = action_idx + 1
        
        if current_step_response and not state.response: 
            state.response = current_step_response
        elif not state.response: 
             state.response = f"Completed internal step: {action_description[:50]}..."


        if state.scratchpad['current_interactive_action_idx'] < len(action_plan):
            state.next_step = "interactive_query_executor" 
        else:
            logger.info("All interactive actions executed.")
            if not state.response: 
                 state.response = "Finished processing your request based on the interactive plan."
            state.next_step = "responder" 

        return state


    def handle_unknown(self, state: BotState) -> BotState:
        tool_name = "handle_unknown"
        state.update_scratchpad_reason(tool_name, "Handling unknown intent or error state.")
        logger.warning("Executing handle_unknown node.")
        current_response = state.response # Preserve any existing error message
        if state.openapi_schema:
            state.response = current_response or "I'm not sure how to process that. You can ask about the API, its operations, or the current workflow graph."
        else:
            state.response = current_response or "I couldn't understand your request. Please provide an OpenAPI spec (JSON/YAML) or ask for help."
        state.next_step = "responder"
        return state

    def handle_loop(self, state: BotState) -> BotState:
        tool_name = "handle_loop"
        state.update_scratchpad_reason(tool_name, "Handling potential loop.")
        logger.warning("Executing handle_loop node. Loop detected.")
        state.response = "It seems we're in a processing loop. Could you please rephrase or try a different request?"
        state.loop_counter = 0 
        state.next_step = "responder"
        return state
