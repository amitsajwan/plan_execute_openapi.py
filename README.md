# filename: core_logic.py
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml # Import yaml for robust parsing
import traceback # Import traceback for detailed error logging
import re # Import regex for parameter extraction (basic example)

# Assuming models.py defines BotState, GraphOutput, Node, Edge, InputMapping etc.
from models import (
    BotState, GraphOutput, Node, Edge, AddEdgeParams,
    GeneratePayloadsParams, GenerateGraphParams, InputMapping,
    PlanExecutionParams, ExecutePlanStepParams # Import new models
)
# Assuming utils.py defines helpers
from utils import (
    llm_call_helper, load_cached_schema, save_schema_to_cache,
    get_cache_key, check_for_cycles, parse_llm_json_output_with_model
)
from pydantic import ValidationError # Import ValidationError for parsing issues

# Module-level logger
logger = logging.getLogger(__name__)

class OpenAPICoreLogic:
    """
    Handles the core tasks of parsing OpenAPI specs, generating payload
    descriptions, creating execution graph descriptions, and managing
    graph descriptions using a worker LLM.
    Includes 'Plan and Execute' logic for subsequent interactions.
    These methods are designed to be used as nodes in the LangGraph.
    They modify the BotState object in place and return the updated state.
    Actual API execution logic is NOT included - it is simulated.
    """
    def __init__(self, worker_llm: Any):
        """
        Initializes the core logic component.

        Args:
            worker_llm: The language model instance dedicated to performing tasks.
        """
        if not hasattr(worker_llm, 'invoke'):
             raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("OpenAPICoreLogic initialized (simulated execution).")

    def parse_openapi_spec(self, state: BotState) -> BotState:
        """
        Parses the raw OpenAPI specification string stored in state.openapi_spec_string.
        Handles both JSON and YAML. Attempts to load from cache first.
        If successful, updates state.openapi_schema, state.schema_summary,
        state.openapi_spec_text, and clears state.openapi_spec_string.
        Then triggers the rest of the initial processing pipeline.
        Returns the updated state.
        """
        tool_name = "parse_openapi_spec"
        state.update_scratchpad_reason(tool_name, "Attempting to parse OpenAPI spec.")
        logger.info("Executing parse_openapi_spec node.")

        spec_text = state.openapi_spec_string
        if not spec_text:
            state.response = "No OpenAPI specification text found in the state to parse."
            state.update_scratchpad_reason(tool_name, "No spec text in state.")
            # No spec to parse, route to responder or handle_unknown
            state.next_step = "responder" if state.openapi_schema else "handle_unknown"
            state.openapi_spec_string = None # Clear the temporary field
            return state

        cache_key = get_cache_key(spec_text)
        cached_schema = load_cached_schema(cache_key)

        if cached_schema:
            logger.info(f"Loaded schema from cache for key: {cache_key}")
            state.openapi_schema = cached_schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text # Store the raw text that led to this schema
            state.openapi_spec_string = None # Clear the temporary field

            # If summary or identified_apis are missing from cache/state, generate them
            # This might happen if the state was saved before these steps completed
            if not state.schema_summary or not state.identified_apis:
                 logger.info("Cached schema loaded, but summary or APIs missing. Generating...")
                 # Generate summary and identify APIs immediately after loading from cache
                 self._generate_llm_schema_summary(state) # Updates state.schema_summary
                 self._identify_apis_from_schema(state) # Updates state.identified_apis
                 # Also generate payloads and graph if they are missing
                 if not state.payload_descriptions or not state.execution_graph:
                      logger.info("Cached schema loaded, but payloads or graph missing. Generating...")
                      # Route to the full pipeline to ensure all steps run
                      state.response = "OpenAPI specification loaded from cache. Completing analysis..."
                      state.update_scratchpad_reason(tool_name, "Schema loaded from cache, missing artifacts. Proceeding with full pipeline.")
                      state.next_step = "process_schema_pipeline"
                      return state


            state.response = "OpenAPI specification loaded from cache."
            state.update_scratchpad_reason(tool_name, "Schema loaded from cache. All artifacts seem present.")
            logger.info("Schema loaded from cache with existing artifacts.")
            # If everything was already in cache, just inform the user and go to responder
            state.next_step = "responder" # Go to responder with success message
            return state


        logger.info("Parsing new OpenAPI spec.")
        parsed_schema = None
        error_message = None

        try:
            # Attempt JSON parsing first
            parsed_schema = json.loads(spec_text)
            logger.debug("Successfully parsed spec as JSON.")
        except json.JSONDecodeError as json_e:
            logger.debug(f"JSON parsing failed: {json_e}. Attempting YAML.")
            try:
                # Attempt YAML parsing
                parsed_schema = yaml.safe_load(spec_text)
                logger.debug("Successfully parsed spec as YAML.")
            except yaml.YAMLError as yaml_e:
                logger.error(f"YAML parsing failed: {yaml_e}")
                error_message = f"Failed to parse specification: Invalid JSON or YAML format. JSON error: {json_e}. YAML error: {yaml_e}"
            except Exception as e:
                 logger.error(f"Unexpected error during YAML parsing: {e}", exc_info=True)
                 error_message = f"Failed to parse specification: Unexpected error during YAML parsing: {e}"
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing: {e}", exc_info=True)
            error_message = f"Failed to parse specification: Unexpected error during JSON parsing: {e}"


        if parsed_schema:
            # Basic validation: Check for 'openapi' or 'swagger' and 'info'
            if not isinstance(parsed_schema, dict) or ('openapi' not in parsed_schema and 'swagger' not in parsed_schema) or 'info' not in parsed_schema:
                 error_message = "Parsed content does not appear to be a valid OpenAPI/Swagger specification (missing 'openapi'/'swagger' or 'info' fields)."
                 logger.error(error_message)
                 parsed_schema = None # Invalidate parsed schema if basic structure is wrong

        if parsed_schema:
            state.openapi_schema = parsed_schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text # Store the raw text
            state.openapi_spec_string = None # Clear the temporary field
            save_schema_to_cache(cache_key, parsed_schema)
            logger.info("Successfully parsed and cached OpenAPI spec.")

            state.response = "OpenAPI specification parsed successfully. Analyzing the API..."
            state.update_scratchpad_reason(tool_name, "Schema parsed successfully. Proceeding with full processing pipeline.")
            # Trigger the rest of the processing pipeline
            state.next_step = "process_schema_pipeline"

        else:
            # Parsing failed
            state.openapi_schema = None
            state.schema_cache_key = None
            state.openapi_spec_text = None
            state.openapi_spec_string = None # Clear the temporary field
            state.response = error_message or "Failed to parse the provided OpenAPI specification."
            state.update_scratchpad_reason(tool_name, f"Parsing failed: {state.response}")
            logger.error(f"Parsing failed: {state.response}")
            state.next_step = "responder" # Go to responder to report parsing error

        return state

    def _generate_llm_schema_summary(self, state: BotState):
         """Internal helper to generate LLM schema summary and update state."""
         tool_name = "_generate_llm_schema_summary"
         state.update_scratchpad_reason(tool_name, "Generating schema summary.")
         logger.debug("Generating schema summary.")

         if not state.openapi_schema:
             logger.warning("No schema available to summarize.")
             state.schema_summary = "Could not generate summary: No OpenAPI schema loaded."
             state.update_scratchpad_reason(tool_name, "No schema available.")
             return

         spec = state.openapi_schema
         info = spec.get('info', {})
         title = info.get('title', 'Untitled API')
         version = info.get('version', 'Unknown')
         description = info.get('description', 'No description provided')

         # Prepare a string representation of paths/operations for the LLM
         paths_summary = ""
         paths = spec.get('paths', {})
         # Limit path details for prompt size, focus on path and methods
         for path, path_item in list(paths.items())[:30]:
             paths_summary += f"\n  {path}:"
             if isinstance(path_item, dict):
                 methods = [method.upper() for method in path_item.keys() if method in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options', 'trace']]
                 paths_summary += f" [{', '.join(methods)}]"
             else:
                 paths_summary += " [Invalid path item]"


         summary_prompt = f"""
         You are summarizing an OpenAPI specification for later reference. Create a concise but comprehensive summary of the API. Focus on the purpose, main features, and types of operations available.

         API Title: {title}
         API Version: {version}

         Description: {description[:1500] + '...' if len(description) > 1500 else description}

         Key Paths/Operations (partial list):
         {paths_summary or 'No paths defined.'}

         Include in your summary:
         1. The overall purpose of this API.
         2. Major resource categories/endpoints and the types of operations they support (e.g., "/users" supports GET to list users and POST to create users).
         3. Any authentication or security requirements mentioned (e.g., API key, OAuth2).
         4. Notable features or patterns in the API design (e.g., uses pagination, supports webhooks).
         5. The total number of paths and operations if available in the spec (calculate if possible, or state based on partial list).

         Keep your summary informative and under 700 words.
         """

         try:
             schema_summary = llm_call_helper(self.worker_llm, summary_prompt)
             state.schema_summary = schema_summary
             state.update_scratchpad_reason(tool_name, "Generated schema summary successfully.")
             logger.info("Schema summary generated.")
         except Exception as e:
             logger.error(f"Error generating schema summary: {e}", exc_info=True)
             state.schema_summary = f"Error generating summary: {str(e)}"
             state.update_scratchpad_reason(tool_name, f"Failed to generate schema summary: {e}")


    def _identify_apis_from_schema(self, state: BotState):
        """Internal helper to identify all APIs in the schema and update state."""
        tool_name = "_identify_apis_from_schema"
        state.update_scratchpad_reason(tool_name, "Identifying APIs.")
        logger.debug("Identifying all APIs in the schema.")

        if not state.openapi_schema:
            logger.warning("No schema available to identify APIs from.")
            state.identified_apis = []
            state.update_scratchpad_reason(tool_name, "No schema available.")
            return

        spec = state.openapi_schema
        paths = spec.get('paths', {})

        all_apis = []
        for path, path_item in paths.items():
            # Skip parameters at path level and other keywords
            if not isinstance(path_item, dict):
                logger.warning(f"Skipping invalid path item for {path}: {path_item}")
                continue

            for method, operation in path_item.items():
                if method not in ['get', 'post', 'put', 'delete', 'patch', 'head', 'options', 'trace']:
                    continue # Skip non-HTTP methods like 'parameters' or 'summary' at path level

                if not isinstance(operation, dict):
                     logger.warning(f"Skipping invalid operation object for {method} {path}: {operation}")
                     continue

                operation_id = operation.get('operationId', f"{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '')}")
                summary = operation.get('summary', 'No summary')
                description = operation.get('description', 'No description')

                # Basic parameter and request body info
                parameters = operation.get('parameters', [])
                request_body = operation.get('requestBody', {})

                api_item = {
                    'operationId': operation_id,
                    'path': path,
                    'method': method.upper(), # Store method as uppercase convention
                    'summary': summary,
                    'description': description,
                    'parameters': parameters, # Include parameters details
                    'requestBody': request_body # Include requestBody details
                }
                all_apis.append(api_item)

        state.identified_apis = all_apis
        state.update_scratchpad_reason(tool_name, f"Identified {len(all_apis)} APIs.")
        logger.info(f"Identified {len(all_apis)} APIs in spec.")

    def _generate_payload_descriptions(self, state: BotState, target_apis: Optional[List[str]] = None):
        """Internal helper to generate payload descriptions and update state."""
        tool_name = "_generate_payload_descriptions"
        state.update_scratchpad_reason(tool_name, "Generating payload descriptions.")
        logger.debug("Generating payload descriptions.")

        if not state.identified_apis:
            logger.warning("No identified APIs to generate payloads for.")
            state.payload_descriptions = {}
            state.update_scratchpad_reason(tool_name, "No identified APIs.")
            return

        # Use existing descriptions if available, but overwrite if target_apis is specified
        payload_descriptions = state.payload_descriptions or {}

        apis_to_process = state.identified_apis
        if target_apis:
             # Filter to only target APIs if specified
             apis_to_process = [api for api in state.identified_apis if api['operationId'] in target_apis]
             logger.debug(f"Generating payloads for targeted APIs: {target_apis}")
             state.update_scratchpad_reason(tool_name, f"Targeted APIs: {target_apis}")
             # If targeting specific APIs, remove existing descriptions for those APIs to regenerate
             for op_id in target_apis:
                 if op_id in payload_descriptions:
                     del payload_descriptions[op_id]


        # Only process APIs that require body or parameters and haven't been described
        apis_requiring_payload = [
             api for api in apis_to_process
             if (api.get('requestBody') or api.get('parameters')) and
                api['operationId'] not in payload_descriptions
        ]

        # Limit the number of LLM calls for performance
        apis_for_llm = apis_requiring_payload[:20] # Process up to 20 requiring description

        if not apis_for_llm:
             logger.info("No APIs requiring new payload descriptions.")
             state.update_scratchpad_reason(tool_name, "No APIs require new descriptions.")
             # state.payload_descriptions remains as is
             return


        for api in apis_for_llm:
            operation_id = api['operationId']
            path = api['path']
            method = api['method']
            summary = api.get('summary', 'No summary')
            parameters = api.get('parameters', [])
            request_body = api.get('requestBody', {})

            # Build a detailed description of parameters and request body schema for the prompt
            payload_details = ""
            if parameters:
                 payload_details += "\nParameters:\n"
                 for param in parameters:
                     param_name = param.get('name', 'N/A')
                     param_in = param.get('in', 'N/A')
                     param_required = param.get('required', False)
                     param_schema_type = param.get('schema', {}).get('type', 'any')
                     param_description = param.get('description', 'No description')
                     payload_details += f"- name: {param_name}, in: {param_in}, required: {param_required}, type: {param_schema_type}, description: {param_description}\n"

            if request_body:
                 payload_details += "\nRequest Body Schema:\n"
                 # Get the schema for the request body, preferring 'application/json' if available
                 content = request_body.get('content', {})
                 json_content = content.get('application/json', {})
                 schema = json_content.get('schema', request_body) # Use request_body if json schema not found

                 # Attempt to dump a simplified JSON schema or just indicate presence
                 try:
                      # Limit schema dump size for prompt
                      schema_dump = json.dumps(schema, indent=2)
                      payload_details += schema_dump[:1000] + '...' if len(schema_dump) > 1000 else schema_dump
                 except:
                      payload_details += "Schema details unavailable or complex."


            payload_prompt = f"""
            Generate a clear, natural language description of an EXAMPLE payload or set of parameters for the following API operation.
            This description is for a human user to understand *how* to use the API, NOT for machine execution. Focus on clarity and realistic examples.

            Operation ID: {operation_id}
            Path: {path}
            Method: {method}
            Summary: {summary}

            Detailed Schema/Parameter Info:
            {payload_details or "No parameters or request body defined."}

            Provide a description that explains:
            1. What parameters (path, query, header, cookie) are needed, their type, and if required.
            2. What request body fields are required, if any, their type, and structure.
            3. A realistic example of values for the key fields that would work for a typical use case.
            4. Any important constraints or validation rules mentioned in the schema.
            5. If there's no request body or parameters, state that explicitly.

            Keep your description concise and easy to understand. Format your response as a clear paragraph or bullet points, NOT actual JSON or code unless explicitly showing an example *within* the description.
            """

            try:
                payload_description = llm_call_helper(self.worker_llm, payload_prompt)
                payload_descriptions[operation_id] = payload_description
                logger.debug(f"Generated payload description for {operation_id}")
                state.update_scratchpad_reason(tool_name, f"Generated payload description for {operation_id}.")
            except Exception as e:
                logger.error(f"Error generating payload description for {operation_id}: {e}", exc_info=True)
                payload_descriptions[operation_id] = f"Error generating description: {str(e)}"
                state.update_scratchpad_reason(tool_name, f"Failed to generate payload description for {operation_id}: {e}")

        state.payload_descriptions = payload_descriptions
        logger.info(f"Generated payload descriptions for {len(apis_for_llm)} APIs.")


    def _generate_execution_graph(self, state: BotState, goal: Optional[str] = None, instructions: Optional[str] = None) -> BotState:
        """
        Internal helper to generate a description of an execution graph.
        Updates state.execution_graph.
        Returns the updated state.
        """
        tool_name = "_generate_execution_graph"
        state.update_scratchpad_reason(tool_name, "Generating execution graph description.")
        logger.info("Generating execution graph description.")

        if not state.identified_apis:
            state.response = "Cannot generate graph description: No identified APIs."
            state.update_scratchpad_reason(tool_name, "No identified APIs for graph generation.")
            state.next_step = "responder" # Go to responder
            return state

        # Prepare API details for the LLM
        api_details = "\n".join([
            f"- operationId: {api['operationId']}\n  method: {api['method']}\n  path: {api['path']}\n  summary: {api['summary']}\n  Payload Description: {state.payload_descriptions.get(api['operationId'], 'No description generated.')}"
            for api in state.identified_apis[:50] # Limit APIs described in prompt for brevity
        ])

        goal_instruction = f"The user's goal is: {goal}" if goal else "The user has not specified a specific goal for the workflow."
        additional_instructions = f"The user provided these additional instructions: {instructions}" if instructions else "There are no specific instructions for the graph structure."
        existing_graph_desc = f"An existing graph description is available:\n{state.execution_graph.description}" if state.execution_graph else "No existing graph description available."
        regeneration_reason = f"Reason for regeneration: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else ""


        graph_prompt = f"""
        You are an AI assistant tasked with describing a plausible API execution workflow based on a loaded OpenAPI specification. Your goal is to generate a Directed Acyclic Graph (DAG) description that represents a sequence of API calls that could be executed to achieve a task or demonstrate a feature of the API.

        Identify relevant API operations from the provided list and describe how they could be chained together, explaining the data dependencies between them (i.e., output of one API call is used as input for another).

        Use the following API operations available in the spec:
        {api_details}
        {'... (potentially more APIs are available)' if len(state.identified_apis) > 50 else ''}

        User Goal: {goal_instruction}
        Additional Instructions: {additional_instructions}
        Existing Graph Description: {existing_graph_desc}
        {regeneration_reason}

        Describe the execution workflow as a DAG. Your output should be a JSON object matching the `GraphOutput` Pydantic model.

        Model Structure (JSON):
        {{
          "nodes": [
            {{
              "operationId": "string", // The operationId from the API list
              "display_name": "string", // A unique name for this specific node instance (e.g., "createUser_step1"), REQUIRED if using the same operationId multiple times
              "summary": "string", // Summary of the operation
              "description": "string", // Description of the operation
              "payload_description": "string", // Description of the payload/parameters
              "input_mappings": [ // Describes how inputs for this node come from previous nodes
                {{
                  "source_operation_id": "string", // operationId or display_name of the source node
                  "source_data_path": "string", // Path (e.g., JSONPath like '$.id') to extract data from source node's described result
                  "target_parameter_name": "string", // Name of the parameter/field in this node
                  "target_parameter_in": "string", // Location (path, query, header, cookie, body) - if applicable
                  "transformation": "string" // Optional transformation instruction
                }}
              ]
            }}
            // ... more nodes
          ],
          "edges": [
            {{
              "from_node": "string", // effective_id (operationId or display_name) of the source node
              "to_node": "string", // effective_id (operationId or display_name) of the target node
              "description": "string", // Description of the dependency (e.g., "requires user ID from creation step")
              // input_mapping is now primarily on the node, but could be here too if needed
            }}
            // ... more edges
          ],
          "description": "string" // Overall natural language description of the workflow
        }}

        Instructions for Generation:
        - Select a small, plausible subset of API operations (e.g., 3-7) to form a simple workflow description. Do not try to include all APIs unless specifically requested and feasible.
        - Ensure the graph is a DAG (no cycles).
        - Use 'display_name' if you use the same 'operationId' more than once in the graph.
        - Describe logical data dependencies using `input_mappings` on the nodes and edge descriptions.
        - Provide a clear overall `description` of the workflow.
        - If the goal is complex or unclear, generate a simple, representative workflow (e.g., create resource, get resource, delete resource).
        - If regenerating due to an error ({regeneration_reason}), carefully adjust the nodes/edges to fix the problem (e.g., remove cycles, add missing nodes).
        - Output ONLY the JSON object. Do not include any other text or markdown formatting outside the JSON.
        """

        try:
            llm_response = llm_call_helper(self.worker_llm, graph_prompt)
            # Parse and validate the LLM's JSON output against the GraphOutput model
            graph_output = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output:
                state.execution_graph = graph_output
                state.response = "Successfully generated a description of a possible API execution graph."
                state.update_scratchpad_reason(tool_name, "Graph description generated and validated.")
                logger.info("Execution graph description generated.")
                state.next_step = "verify_graph" # Route to graph verification
                state.graph_regeneration_reason = None # Clear regeneration reason on success
            else:
                # LLM output was not valid JSON or didn't match the model
                logger.error("LLM failed to produce valid GraphOutput JSON.")
                state.response = "Failed to generate a valid execution graph description. The AI did not provide the correct format."
                state.execution_graph = None # Clear potentially invalid graph state
                state.update_scratchpad_reason(tool_name, "LLM failed to generate valid graph JSON.")
                state.graph_regeneration_reason = "LLM output format was incorrect." # Set regeneration reason
                state.next_step = "generate_execution_graph" # Loop back to try generating again

                state.scratchpad['graph_gen_attempts'] = state.scratchpad.get('graph_gen_attempts', 0) + 1
                if state.scratchpad['graph_gen_attempts'] > 2: # Prevent infinite loops
                     state.next_step = "handle_unknown" # Give up after a few attempts
                     state.response += "\nMultiple attempts to generate the graph failed."
                     state.graph_regeneration_reason = None # Clear reason if giving up


        except Exception as e:
            logger.error(f"Error during graph generation LLM call or parsing: {e}", exc_info=True)
            state.response = f"An error occurred while trying to generate the execution graph description: {str(e)}. Please try again."
            state.execution_graph = None
            state.update_scratchpad_reason(tool_name, f"Error during graph generation: {e}")
            state.graph_regeneration_reason = f"Error during generation: {str(e)}" # Set regeneration reason
            state.next_step = "generate_execution_graph" # Loop back

            state.scratchpad['graph_gen_attempts'] = state.scratchpad.get('graph_gen_attempts', 0) + 1
            if state.scratchpad['graph_gen_attempts'] > 2: # Prevent infinite loops
                 state.next_step = "handle_unknown" # Give up
                 state.response += "\nMultiple attempts to generate the graph failed."
                 state.graph_regeneration_reason = None # Clear reason if giving up


        return state

    def process_schema_pipeline(self, state: BotState) -> BotState:
        """
        Node to run the full schema processing pipeline:
        Summary -> Identify APIs -> Generate Payloads -> Generate Graph -> Verify Graph.
        Assumes state.openapi_schema is already populated.
        Returns the updated state.
        """
        tool_name = "process_schema_pipeline"
        state.update_scratchpad_reason(tool_name, "Starting schema processing pipeline.")
        logger.info("Executing process_schema_pipeline.")

        if not state.openapi_schema:
            state.response = "Cannot run processing pipeline: No OpenAPI schema loaded."
            state.update_scratchpad_reason(tool_name, "No schema to process.")
            state.next_step = "handle_unknown" # Cannot proceed without schema
            return state

        try:
            # Step 1: Generate a detailed schema summary
            if not state.schema_summary: # Only generate if not already present (e.g., from cache)
                 self._generate_llm_schema_summary(state)

            # Step 2: Identify all APIs in the spec
            if not state.identified_apis: # Only identify if not already present
                 self._identify_apis_from_schema(state)

            # Step 3: Generate example payload descriptions for all APIs
            # Generate descriptions for all identified APIs initially
            # Only generate if payload_descriptions is empty or if explicitly requested (not the case in pipeline)
            if not state.payload_descriptions:
                 self._generate_payload_descriptions(state, target_apis=[api['operationId'] for api in state.identified_apis])

            # Step 4: Create an execution graph description
            # Only generate if graph is missing
            if not state.execution_graph:
                default_goal = f"Describe a typical workflow using the {state.openapi_schema.get('info', {}).get('title', 'API')}."
                # Clear previous graph generation instructions before generating a new one in the pipeline
                state.graph_generation_instructions = None
                self._generate_execution_graph(state, goal=default_goal) # This sets state.next_step to verify_graph or retry/fail

            # The _generate_execution_graph method sets state.next_step.
            # If graph generation succeeded, state.next_step is "verify_graph".
            # If it failed and wants to retry, state.next_step is "generate_execution_graph".
            # If it failed critically, state.next_step is "handle_unknown".

            # After graph generation (or if graph already existed), proceed based on state.next_step
            # If graph generation happened, state.next_step is already set.
            # If graph already existed, state.next_step might still be "process_schema_pipeline".
            # In that case, we should proceed to verification if graph exists.
            if state.execution_graph and state.next_step == "process_schema_pipeline":
                 logger.info("Graph already exists after pipeline steps. Proceeding to verification.")
                 state.next_step = "verify_graph"

            # If graph generation failed and set next_step to retry or unknown,
            # the pipeline node is finished and the graph will handle the routing.
            if state.next_step in ["generate_execution_graph", "handle_unknown"]:
                 logger.warning(f"Graph generation step within pipeline set next_step to {state.next_step}. Exiting pipeline node.")
                 return state # Exit pipeline node, graph handles next step

            # If graph generation succeeded and set next_step to verify_graph,
            # the pipeline node is finished and the graph will handle the routing.
            if state.next_step == "verify_graph":
                 logger.info("Graph generation succeeded. Proceeding to verification (next step already set).")
                 return state # Exit pipeline node, graph handles next step

            # Fallback if next_step wasn't set correctly by graph generation
            if state.next_step == "process_schema_pipeline":
                 logger.error("Graph generation step failed to set next_step correctly in pipeline.")
                 state.response = state.response or "An issue occurred after graph generation."
                 state.next_step = "handle_unknown" # Prevent hanging
                 state.update_scratchpad_reason(tool_name, "Graph generation failed to set next_step.")
                 return state


        except Exception as e:
            error_msg = f"Error during schema processing pipeline: {str(e)}"
            logger.critical(error_msg, exc_info=True)
            state.response = f"An unexpected error occurred during the API analysis pipeline: {str(e)}. Please try again."
            state.update_scratchpad_reason(tool_name, error_msg)
            state.next_step = "responder" # Go to responder with error

        return state # Return updated state

    def verify_graph(self, state: BotState) -> BotState:
        """
        Verifies the generated execution graph:
        1. Checks for cycles (DAG).
        2. Checks if all identified APIs are included as nodes (strict check).
        Updates state.response and sets state.next_step.
        Returns the updated state.
        """
        tool_name = "verify_graph"
        state.update_scratchpad_reason(tool_name, "Verifying execution graph.")
        logger.info("Executing verify_graph node.")

        graph = state.execution_graph
        identified_apis = state.identified_apis

        if not graph:
            state.response = "Cannot verify graph: No execution graph description found in the state."
            state.update_scratchpad_reason(tool_name, "No graph to verify.")
            state.next_step = "handle_unknown" # Cannot verify without a graph
            return state

        verification_messages = []
        is_valid = True

        # 1. Check for cycles
        is_dag, cycle_message = check_for_cycles(graph)
        if not is_dag:
            is_valid = False
            verification_messages.append(f"Graph validation failed: {cycle_message}")
            state.update_scratchpad_reason(tool_name, f"Cycle detected: {cycle_message}")
            logger.warning(f"Graph verification: Cycle detected - {cycle_message}")
        else:
             verification_messages.append("Graph validation: No cycles detected (DAG).")
             state.update_scratchpad_reason(tool_name, "No cycles detected.")
             logger.debug("Graph verification: No cycles detected.")


        # 2. Check if all identified APIs are included as nodes
        # Get operationIds from identified APIs
        identified_operation_ids = {api['operationId'] for api in identified_apis} if identified_apis else set()
        # Get operationIds from graph nodes (using the original operationId, not effective_id for this check)
        graph_node_operation_ids = {node.operationId for node in graph.nodes if isinstance(node, Node)}

        # Check for missing APIs (identified but not in graph)
        missing_apis = identified_operation_ids - graph_node_operation_ids
        if missing_apis:
            # Treat missing APIs as a validation failure per instruction.
            is_valid = False
            missing_apis_list = list(missing_apis)[:10] # List up to 10 missing
            missing_msg = f"Graph validation failed: The following identified APIs are not included as nodes in the graph: {', '.join(missing_apis_list)}{'...' if len(missing_apis) > 10 else ''}."
            verification_messages.append(missing_msg)
            state.update_scratchpad_reason(tool_name, missing_msg)
            logger.warning(f"Graph verification: Missing APIs - {missing_msg}")
        else:
            verification_messages.append("Graph validation: All identified APIs are included as nodes.")
            state.update_scratchpad_reason(tool_name, "All identified APIs are included.")
            logger.debug("Graph verification: All identified APIs included.")


        # --- Set next step based on verification results ---
        if is_valid:
            state.response = "Graph verification successful: " + " ".join(verification_messages)
            state.update_scratchpad_reason(tool_name, "Graph verification successful.")
            logger.info("Graph verification successful.")

            # If this was part of the initial pipeline (triggered by spec upload)
            if state.input_is_spec:
                 # Add a user-friendly summary of the initial processing
                 api_title = state.openapi_schema.get('info', {}).get('title', 'the API')
                 total_apis = len(state.identified_apis) if state.identified_apis else 0
                 num_graph_nodes = len(state.execution_graph.nodes) if state.execution_graph else 0
                 num_payload_descs = len(state.payload_descriptions) if state.payload_descriptions else 0

                 state.response = f"""
                 I've successfully processed your OpenAPI specification for {api_title}.

                 **Analysis Summary:**
                 - Identified {total_apis} API operations.
                 - Generated descriptions for {num_payload_descs} API payloads/parameters.
                 - Created a sample execution graph with {num_graph_nodes} steps.

                 **Graph Verification:** {', '.join(verification_messages)}

                 You can now ask me questions about the API (e.g., "List all endpoints", "Describe the 'createUser' API", "What parameters does 'getProduct' need?") or ask me to plan a workflow (e.g., "Plan how to create a user and then get their details").
                 """
                 state.next_step = "responder" # Go to responder with success message
                 state.input_is_spec = False # Reset the flag after initial processing
                 state.scratchpad['graph_gen_attempts'] = 0 # Reset attempts on final success
            else: # If verification was triggered by a user command like "verify graph"
                 state.response = "Graph verification successful:\n" + "\n".join(verification_messages)
                 state.next_step = "describe_graph" # Route to describe_graph to show the user

        else:
            # Verification failed - Cycle or missing APIs
            state.response = "Graph verification failed:\n" + "\n".join(verification_messages)
            state.update_scratchpad_reason(tool_name, "Graph verification failed. Routing to regenerate graph.")
            logger.warning("Graph verification failed. Routing to regenerate graph.")

            # Loop back to generate graph, with feedback in state.graph_regeneration_reason
            state.graph_regeneration_reason = state.response # Set the reason for regeneration
            state.next_step = "generate_execution_graph" # Route back to graph generation

            # Prevent infinite verification/generation loops
            state.scratchpad['graph_gen_attempts'] = state.scratchpad.get('graph_gen_attempts', 0) + 1
            if state.scratchpad['graph_gen_attempts'] > 3: # Allow a few regeneration attempts
                 state.next_step = "handle_unknown" # Give up after multiple failures
                 state.response += "\nMultiple attempts to generate a valid graph failed."
                 state.graph_regeneration_reason = None # Clear reason if giving up
                 state.update_scratchpad_reason(tool_name, "Multiple regeneration attempts failed. Routing to handle_unknown.")


        return state

    def describe_graph(self, state: BotState) -> BotState:
        """
        Generates a natural language description of the current execution graph.
        Updates state.response.
        Returns the updated state.
        """
        tool_name = "describe_graph"
        state.update_scratchpad_reason(tool_name, "Describing execution graph.")
        logger.info("Executing describe_graph node.")

        graph = state.execution_graph

        if not graph:
            state.response = "No execution graph description found in the state to describe."
            state.update_scratchpad_reason(tool_name, "No graph to describe.")
            state.next_step = "responder" # Go to responder
            return state

        description_prompt = f"""
        Provide a concise, natural language description of the following API execution workflow graph.
        The graph describes a sequence of API calls and their dependencies.

        Graph Description: {graph.description or 'No overall description provided.'}

        Nodes (API Calls):
        {json.dumps([node.model_dump() for node in graph.nodes], indent=2)[:2000] + '...' if graph.nodes else 'None'}

        Edges (Dependencies):
        {json.dumps([edge.model_dump() for edge in graph.edges], indent=2)[:1000] + '...' if graph.edges else 'None'}

        Explain the purpose of the workflow, the key steps (API calls), and how data flows between them based on the nodes and edges. Be clear and easy to understand for a human user.

        Output the description only.
        """

        try:
            description = llm_call_helper(self.worker_llm, description_prompt)
            state.response = f"Here is a description of the current execution graph:\n{description}"
            state.update_scratchpad_reason(tool_name, "Generated graph description.")
            logger.info("Generated graph description.")
        except Exception as e:
            logger.error(f"Error generating graph description: {e}", exc_info=True)
            state.response = f"An error occurred while trying to describe the graph: {str(e)}. You can try getting the raw JSON instead."
            state.update_scratchpad_reason(tool_name, f"Failed to generate graph description: {e}")

        state.next_step = "responder" # Go to responder with the description
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        """
        Provides the raw JSON of the current execution graph description.
        Updates state.response.
        Returns the updated state.
        """
        tool_name = "get_graph_json"
        state.update_scratchpad_reason(tool_name, "Getting graph JSON.")
        logger.info("Executing get_graph_json node.")

        graph = state.execution_graph

        if not graph:
            state.response = "No execution graph description found in the state to provide as JSON."
            state.update_scratchpad_reason(tool_name, "No graph to provide as JSON.")
        else:
            try:
                # Use model_dump_json to get a JSON string from the Pydantic model
                graph_json = graph.model_dump_json(indent=2)
                # It might be helpful to wrap this in a markdown code block
                state.response = f"Here is the JSON representation of the current execution graph:\n```json\n{graph_json}\n```"
                state.update_scratchpad_reason(tool_name, "Provided graph JSON.")
                logger.info("Provided graph JSON.")
            except Exception as e:
                logger.error(f"Error serializing graph to JSON: {e}", exc_info=True)
                state.response = f"An error occurred while trying to generate the graph JSON: {str(e)}."
                state.update_scratchpad_reason(tool_name, f"Failed to serialize graph to JSON: {e}")

        state.next_step = "responder" # Go to responder
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        """
        Answers a general user query about the loaded OpenAPI spec or artifacts.
        Uses the LLM to formulate the answer based on available state information.
        Updates state.response.
        Returns the updated state.
        """
        tool_name = "answer_openapi_query"
        state.update_scratchpad_reason(tool_name, "Answering general OpenAPI query.")
        logger.info("Executing answer_openapi_query node.")

        user_input = state.user_input

        if not state.openapi_schema:
            state.response = "I don't have an OpenAPI specification loaded yet. Please provide one first."
            state.update_scratchpad_reason(tool_name, "No schema loaded to answer query about.")
            state.next_step = "responder" # Go to responder
            return state

        # Provide context to the LLM
        context = f"""
        You are an AI assistant specializing in OpenAPI specifications. A user has asked a question about the loaded API spec or the artifacts derived from it (summary, identified APIs, payload descriptions, graph description).

        Answer the user's question based on the available information below. If you don't have enough information in the provided state, state that you cannot fully answer.

        User Question: "{user_input}"

        Available Information:
        - Schema Summary: {state.schema_summary or 'Not available.'}
        - Identified APIs ({len(state.identified_apis) if state.identified_apis else 0}):
          {''.join([f"  - operationId: {api['operationId']}, method: {api['method']}, path: {api['path']}, summary: {api['summary']}\n" for api in state.identified_apis[:30]])[:1500] + '...' if state.identified_apis else 'None'}
        - Payload Descriptions ({len(state.payload_descriptions) if state.payload_descriptions else 0}):
          {''.join([f"  - operationId: {op_id}: {desc[:200]}...\n" for op_id, desc in state.payload_descriptions.items()])[:1500] + '...' if state.payload_descriptions else 'None'}
        - Execution Graph Description: {state.execution_graph.description if state.execution_graph else 'None'}
        - Execution Graph Structure (Nodes: {len(state.execution_graph.nodes) if state.execution_graph else 0}, Edges: {len(state.execution_graph.edges) if state.execution_graph else 0}):
          {json.dumps({'nodes': [n.effective_id for n in state.execution_graph.nodes[:10]] if state.execution_graph else [], 'edges': [(e.from_node, e.to_node) for e in state.execution_graph.edges[:10]] if state.execution_graph else []}, indent=2) if state.execution_graph else 'None'}
        - Raw user input was identified as a spec: {state.input_is_spec} (For context, but the current input is a query)


        Instructions:
        - Directly answer the user's question based on the provided context.
        - If the question is about a specific API, refer to its summary and payload description if available.
        - If the question is about the graph, refer to the graph description and structure.
        - If the information is not in the provided context, state that you cannot answer based on the current loaded data.
        - Be helpful and concise.
        """

        try:
            answer = llm_call_helper(self.worker_llm, context)
            state.response = answer
            state.update_scratchpad_reason(tool_name, "Generated answer for OpenAPI query.")
            logger.info("Generated answer for OpenAPI query.")
        except Exception as e:
            logger.error(f"Error answering OpenAPI query: {e}", exc_info=True)
            state.response = f"An error occurred while trying to answer your question: {str(e)}. Please try rephrasing."
            state.update_scratchpad_reason(tool_name, f"Failed to answer query: {e}")

        state.next_step = "responder" # Go to responder
        return state

    def handle_unknown(self, state: BotState) -> BotState:
        """
        Handles cases where the user's intent could not be determined.
        Provides a helpful message.
        Returns the updated state.
        """
        tool_name = "handle_unknown"
        state.update_scratchpad_reason(tool_name, "Handling unknown intent.")
        logger.warning("Executing handle_unknown node.")

        if state.user_input and len(state.user_input) > 150 and not state.openapi_schema:
             # If input was long and looked like a spec but wasn't parsed (e.g. format error)
             state.response = "I couldn't understand your request or parse the input you provided. If you were trying to provide an OpenAPI specification, please ensure it is valid JSON or YAML."
        elif state.openapi_schema:
            state.response = "I'm not sure how to process that request based on the loaded OpenAPI specification. You can ask me questions about the API, generate a graph, or ask me to plan a workflow (e.g., 'Plan how to create a user')."
        else:
            state.response = "I couldn't understand your request. Please provide an OpenAPI specification (JSON or YAML) to get started."

        state.update_scratchpad_reason(tool_name, "Provided unknown intent message.")
        state.next_step = "responder" # Go to responder
        return state

    def handle_loop(self, state: BotState) -> BotState:
        """
        Handles cases where a potential loop in routing is detected.
        Informs the user and resets the loop counter.
        Returns the updated state.
        """
        tool_name = "handle_loop"
        state.update_scratchpad_reason(tool_name, "Handling potential loop.")
        logger.warning("Executing handle_loop node. Loop detected.")

        state.response = "It looks like we might be stuck in a loop. Could you please rephrase your last request or try a different command?"
        state.update_scratchpad_reason(tool_name, "Provided loop detected message.")
        state.loop_counter = 0 # Reset loop counter
        state.next_step = "responder" # Go to responder
        return state

    # --- Plan and Execute Nodes ---

    def plan_execution(self, state: BotState) -> BotState:
        """
        The 'Plan' node. Takes the user's goal and generates a sequence of steps
        (operationIds) to achieve it using the available APIs.
        Stores the plan in state.execution_plan and routes to execute_plan_step.
        Returns the updated state.
        """
        tool_name = "plan_execution"
        state.update_scratchpad_reason(tool_name, "Starting planning phase.")
        logger.info("Executing plan_execution node.")

        user_goal = state.user_input # The user's input is the goal for planning
        state.plan_execution_goal = user_goal # Store the goal in state

        if not state.identified_apis:
            state.response = "Cannot plan execution: No APIs have been identified from the loaded specification."
            state.update_scratchpad_reason(tool_name, "No identified APIs for planning.")
            state.next_step = "responder"
            return state

        # Prepare API details for the LLM planner
        api_details = "\n".join([
            f"- operationId: {api['operationId']}\n  method: {api['method']}\n  path: {api['path']}\n  summary: {api['summary']}\n  Payload Description: {state.payload_descriptions.get(api['operationId'], 'No description generated.')}"
            for api in state.identified_apis[:50] # Limit APIs described in prompt
        ])

        planning_prompt = f"""
        You are an AI planner. Based on the user's goal and the available API operations, create a simple, sequential plan to achieve the goal.

        Available API Operations:
        {api_details}
        {'... (potentially more APIs are available)' if len(state.identified_apis) > 50 else ''}

        User Goal: "{user_goal}"

        Instructions:
        1. Identify the relevant API operations from the list above needed to achieve the user's goal.
        2. Create a sequential plan as a list of `operationId`s.
        3. If the goal requires data from a previous step (e.g., getting an ID after creation), include the necessary steps in order.
        4. If the goal is complex or ambiguous, create a plausible simplified plan.
        5. If the goal cannot be achieved with the available APIs, state that.
        6. Output ONLY a JSON object with a single key "plan" containing a list of strings (operationIds), and a key "description" with a natural language description of the plan.

        Example Output:
        {{
          "plan": ["createUser", "getUserDetails"],
          "description": "Plan to create a user and then get their details."
        }}

        Output JSON:
        """
        try:
            llm_response = llm_call_helper(self.worker_llm, planning_prompt)
            # Use a simple dict validation for the plan structure
            plan_output = parse_llm_json_output_with_model(llm_response, expected_model=None) # No strict model for plan for flexibility

            if plan_output and isinstance(plan_output, dict) and isinstance(plan_output.get("plan"), list):
                plan = plan_output["plan"]
                plan_description = plan_output.get("description", "No description provided.")

                # Basic validation: check if planned operationIds exist in identified_apis
                identified_op_ids = {api['operationId'] for api in state.identified_apis}
                valid_plan = [op_id for op_id in plan if op_id in identified_op_ids]
                invalid_steps = [op_id for op_id in plan if op_id not in identified_op_ids]

                if invalid_steps:
                    logger.warning(f"Planner included invalid operationIds: {invalid_steps}. Filtering plan.")
                    state.response = f"Generated a plan, but some steps reference APIs that were not identified: {', '.join(invalid_steps)}. Proceeding with the valid steps."
                    state.update_scratchpad_reason(tool_name, f"Filtered invalid steps from plan: {invalid_steps}")
                else:
                     state.response = f"Plan generated: {plan_description}"
                     state.update_scratchpad_reason(tool_name, f"Plan generated: {plan_description}")


                state.execution_plan = valid_plan # Store the validated plan
                state.current_plan_step = 0 # Start at the first step

                if state.execution_plan:
                    logger.info(f"Plan generated: {state.execution_plan}. Routing to execute_plan_step.")
                    state.next_step = "execute_plan_step" # Route to the executor
                else:
                    state.response = state.response or "Could not generate a valid plan for your goal using the available APIs."
                    state.update_scratchpad_reason(tool_name, "Generated empty or invalid plan.")
                    state.next_step = "responder" # Go to responder if plan is empty

            else:
                logger.error("LLM failed to produce valid plan JSON output.")
                state.response = "Failed to generate a plan. The AI did not provide the correct format."
                state.update_scratchpad_reason(tool_name, "LLM failed to generate valid plan JSON.")
                state.next_step = "handle_unknown" # Cannot plan, handle as unknown

        except Exception as e:
            logger.error(f"Error during planning LLM call or parsing: {e}", exc_info=True)
            state.response = f"An error occurred while trying to generate a plan: {str(e)}. Please try again."
            state.update_scratchpad_reason(tool_name, f"Error during planning: {e}")
            state.next_step = "handle_unknown" # Error during planning, handle as unknown

        return state

    def execute_plan_step(self, state: BotState) -> BotState:
        """
        The 'Execute' node. Executes the current step in the plan (simulated).
        Simulates the API call, updates state with simulated result,
        advances the plan step, and routes accordingly.
        Returns the updated state.
        """
        tool_name = "execute_plan_step"
        state.update_scratchpad_reason(tool_name, f"Executing plan step {state.current_plan_step}.")
        logger.info(f"Executing plan step {state.current_plan_step}.")

        plan = state.execution_plan
        current_step_index = state.current_plan_step

        if not plan or current_step_index >= len(plan):
            logger.warning("Execute node called with invalid plan or step index.")
            state.response = state.response or "Execution finished or called incorrectly."
            state.next_step = "responder" # Plan finished or error
            return state

        current_operation_id = plan[current_step_index]
        logger.info(f"Simulating execution for operation: {current_operation_id}")
        state.update_scratchpad_reason(tool_name, f"Simulating operation: {current_operation_id}")

        # Find the API details for the current operationId
        api_details = next((api for api in state.identified_apis if api['operationId'] == current_operation_id), None)

        if not api_details:
            error_msg = f"Cannot execute step '{current_operation_id}': API details not found."
            logger.error(error_msg)
            state.response = state.response or f"Error executing plan step: {error_msg}"
            state.update_scratchpad_reason(tool_name, error_msg)
            state.next_step = "responder" # Cannot proceed with plan
            return state

        # --- Simulate API Execution ---
        # This is the core simulation part. The LLM pretends to be the API.
        # It needs context about the API, the step, and any data from previous steps.

        # Get simulated results from previous steps from scratchpad
        previous_results = state.scratchpad.get('simulated_results', {})

        # Prepare context for the simulation LLM call
        simulation_context = f"""
        You are simulating the response of an API call based on its OpenAPI specification details.
        The user wants to understand the output of a planned sequence of API calls.
        DO NOT make a real API call. Generate a plausible, example JSON response based on the API's likely behavior and the context provided.

        Current API Operation:
        operationId: {api_details.get('operationId')}
        method: {api_details.get('method')}
        path: {api_details.get('path')}
        summary: {api_details.get('summary')}
        description: {api_details.get('description', 'None')}
        Payload Description: {state.payload_descriptions.get(current_operation_id, 'None')}

        Context from previous simulated steps:
        {json.dumps(previous_results, indent=2)[:1500] + '...' if previous_results else 'None'}

        User's overall goal for this plan: "{state.plan_execution_goal}"

        Instructions:
        1. Consider the API details, the overall goal, and the results from previous steps.
        2. Generate a realistic, example JSON object that this API operation might return on success.
        3. The JSON should reflect the API's purpose (e.g., creating a user should return a user object, getting a list should return an array).
        4. If the API typically returns a simple status, simulate that.
        5. If the API might fail based on context (e.g., trying to get a user that wasn't created), you could simulate an error response, but for simplicity, assume success unless explicitly instructed otherwise.
        6. Include key fields that might be needed by subsequent steps in the plan (e.g., an ID after creation).
        7. Output ONLY the JSON object. Do not include any other text or markdown formatting outside the JSON.
        """

        try:
            simulated_response_str = llm_call_helper(self.worker_llm, simulation_context)
            # Attempt to parse the simulated response as JSON
            simulated_result = parse_llm_json_output_with_model(simulated_response_str)

            if simulated_result is None:
                 logger.warning(f"Simulated response for {current_operation_id} was not valid JSON. Storing as raw string.")
                 # Store the raw string if JSON parsing fails
                 simulated_result = simulated_response_str
                 state.update_scratchpad_reason(tool_name, f"Simulated response for {current_operation_id} was not valid JSON.")
            else:
                 state.update_scratchpad_reason(tool_name, f"Simulated execution for {current_operation_id} successful.")


            # Store the simulated result in scratchpad, keyed by the operationId (or effective_id if nodes had display_name)
            # Using operationId for simplicity here, assuming unique operationIds in the plan
            if 'simulated_results' not in state.scratchpad:
                 state.scratchpad['simulated_results'] = {}
            state.scratchpad['simulated_results'][current_operation_id] = simulated_result
            logger.debug(f"Stored simulated result for {current_operation_id}")


            # --- Advance Plan ---
            state.current_plan_step += 1

            # Determine next step
            if state.current_plan_step < len(plan):
                # More steps in the plan
                state.response = f"Completed step {current_step_index + 1}/{len(plan)} ({current_operation_id}). Simulating next step..."
                state.next_step = "execute_plan_step" # Route back to execute the next step
                logger.info(f"Plan continuing. Next step: {state.current_plan_step}")
            else:
                # Plan is finished
                final_plan_message = f"Plan execution finished. All {len(plan)} steps simulated."
                # Optionally summarize results
                results_summary = json.dumps(state.scratchpad.get('simulated_results', {}), indent=2)[:1000] + '...'
                state.response = f"{final_plan_message}\n\nSimulated Results:\n```json\n{results_summary}\n```"
                state.update_scratchpad_reason(tool_name, "Plan execution finished.")
                logger.info("Plan execution finished.")
                # Clear plan execution state
                state.execution_plan = []
                state.current_plan_step = 0
                state.plan_execution_goal = None
                # Keep simulated_results in scratchpad for potential follow-up questions

                state.next_step = "responder" # Go to responder with final result

        except Exception as e:
            error_msg = f"Error simulating execution for step '{current_operation_id}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            state.response = state.response or f"An error occurred during plan execution: {error_msg}. Stopping plan."
            state.update_scratchpad_reason(tool_name, error_msg)
            # Clear plan execution state on error
            state.execution_plan = []
            state.current_plan_step = 0
            state.plan_execution_goal = None
            # Keep simulated_results up to the point of error

            state.next_step = "responder" # Go to responder with error message

        return state


    # --- Placeholder/Skeleton nodes for future graph modification logic ---

    def update_graph(self, state: BotState) -> BotState:
        """
        Placeholder node to handle requests to update the graph (add/delete edges).
        This would require parsing parameters from user input and modifying
        state.execution_graph.
        Returns the updated state.
        """
        tool_name = "update_graph"
        state.update_scratchpad_reason(tool_name, "Attempting to update graph.")
        logger.info(f"Executing update_graph node with input: {state.user_input}")

        # This is where logic to parse the user's instruction (e.g., "add edge from A to B")
        # and modify state.execution_graph would go.
        # For now, it's just a placeholder.

        state.response = "Graph update functionality is not yet fully implemented."
        state.update_scratchpad_reason(tool_name, "Graph update logic not implemented.")
        logger.warning("Graph update logic not implemented.")

        state.extracted_params = None # Clear extracted params after use
        state.next_step = "responder" # Route to responder
        return state


------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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


------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# filename: main.py

import logging
import uuid
import json
import os
import sys # Import sys to check Python version
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, File, UploadFile, Form # Added UploadFile, Form for potential file upload
from fastapi.responses import HTMLResponse, JSONResponse # Added JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.memory import MemorySaver # Assuming you are still using MemorySaver
# If using a different checkpointer (e.g., SQL), import it here

# Import necessary components from your project files
from graph import build_graph # Imports build_graph function
from models import BotState # Imports BotState model
# Import utils for cache closing on shutdown if needed
# try:
#     import utils
# except ImportError:
#     utils = None
#     logging.warning("Could not import utils. Cache will not be explicitly closed on shutdown.")


# --- Basic Logging Setup ---
# Configure logging to output to console
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM Initialization (REPLACE PLACEHOLDER) ---
# Use python-dotenv to load environment variables from a .env file
try:
    from dotenv import load_dotenv
    load_dotenv() # Load environment variables from .env file
    logger.info(".env file loaded.")
except ImportError:
    logger.warning("python-dotenv not installed. Cannot load environment variables from .env file.")
except Exception as e:
    logger.warning(f"Error loading .env file: {e}")


def initialize_llms():
    """
    Initializes and returns the router and worker LLM instances.
    Replace this with your actual LLM setup using your preferred LLM provider.
    Ensure API keys are handled securely (e.g., environment variables).
    """
    logger.info("Initializing LLMs...")

    # Example using environment variables for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # google_api_key = os.getenv("GOOGLE_API_KEY") # Example for Google

    router_llm = None
    worker_llm = None

    # --- START OF REAL LLM EXAMPLE (OpenAI) ---
    # UNCOMMENT AND REPLACE WITH YOUR ACTUAL LLM INSTANTIATION
    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Attempting to initialize ChatOpenAI.")
            # Ensure model names are appropriate for your OpenAI account/tier
            # gpt-3.5-turbo is usually sufficient and cheaper for routing/simple tasks
            router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
            # gpt-4o-mini or gpt-4-turbo might be better for complex tasks like planning/simulating
            worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=openai_api_key)
            logger.info("ChatOpenAI LLMs initialized.")
        except ImportError:
             logger.error("langchain-openai not installed. Cannot initialize ChatOpenAI.")
        except Exception as e:
             logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
    else:
         logger.warning("OPENAI_API_KEY environment variable not set. Cannot initialize ChatOpenAI.")
    # --- END REAL LLM EXAMPLE ---


    # --- START OF PLACEHOLDER FALLBACK ---
    # This block provides dummy LLMs if real ones fail to initialize.
    # REMOVE THIS BLOCK IN YOUR ACTUAL PRODUCTION IMPLEMENTATION if you require real LLMs.
    if router_llm is None or worker_llm is None:
        logger.warning("Using PlaceholderLLMs due to missing API key or initialization failure.")
        # Define a simple placeholder class that mimics the LLM interface expected by utils.llm_call_helper
        class PlaceholderLLM:
            def __init__(self, name="PlaceholderLLM"): self.name = name
            # Simulate invoke returning an object with .content or a string directly
            def invoke(self, prompt: Any, **kwargs) -> Any:
                logger.warning(f"Using {self.name}. This needs replacement with a real LLM.")
                prompt_str = str(prompt)

                # Simple simulation based on prompt keywords
                if "Determine the user's high-level intent" in prompt_str:
                    # Simulate router response (string intent) - needs to match router's logic
                    if any(sig in prompt_str for sig in ['"openapi":', 'swagger:', '{', '-']):
                         return "parse_openapi_spec"
                    elif "list apis" in prompt_str.lower() or "endpoints" in prompt_str.lower():
                         return "answer_openapi_query"
                    elif "describe graph" in prompt_str.lower() or "show graph" in prompt_str.lower():
                         return "describe_graph"
                    elif "get graph json" in prompt_str.lower() or "show graph json" in prompt_str.lower():
                         return "get_graph_json"
                    elif "generate graph" in prompt_str.lower():
                         return "generate_execution_graph"
                    elif "plan execution" in prompt_str.lower() or "create plan" in prompt_str.lower() or "workflow for" in prompt_str.lower():
                         return "plan_execution"
                    elif "add edge" in prompt_str.lower() or "delete edge" in prompt_str.lower():
                         return "update_graph"
                    else:
                         return "handle_unknown"

                # Simulate core_logic responses (text or JSON string)
                if "Create a concise but comprehensive summary" in prompt_str:
                    return "Simulating schema summary: This is a placeholder summary for the API."
                if "Identify relevant API operations" in prompt_str:
                     # Simulate JSON output for identified APIs
                     return json.dumps([{"operationId": "getPlaceholder", "method": "GET", "path": "/placeholder", "summary": "Get placeholder data"}])
                if "Generate a clear, natural language description of an EXAMPLE payload" in prompt_str:
                     # Simulate payload description - try to extract operationId from prompt
                     match = re.search(r"Operation ID: (\w+)", prompt_str)
                     op_id = match.group(1) if match else "an API"
                     return f"Simulating payload description for {op_id}: Requires an optional 'id' parameter."
                if "Generate a description of an API execution workflow graph" in prompt_str:
                     # Simulate valid GraphOutput JSON
                     simulated_graph = {
                         "nodes": [
                             {"operationId": "createPlaceholder", "display_name": "create_step", "summary": "Create placeholder", "payload_description": "Requires data in body.", "input_mappings": []},
                             {"operationId": "getPlaceholder", "display_name": "get_step", "summary": "Get placeholder", "payload_description": "Requires ID.", "input_mappings": [{"source_operation_id": "create_step", "source_data_path": "$.id", "target_parameter_name": "id", "target_parameter_in": "path"}]}
                         ],
                         "edges": [
                              {"from_node": "create_step", "to_node": "get_step", "description": "Uses ID from creation."}
                         ],
                         "description": "Simulated workflow to create and then get a placeholder."
                     }
                     return json.dumps(simulated_graph)
                if "Provide a concise, natural language description of the following API execution workflow graph" in prompt_str:
                     # Simulate graph description based on prompt content
                     return "Simulating graph description: This workflow creates a placeholder resource and then retrieves it using the ID obtained from the creation step."
                if "Answer the user's question based on the available information" in prompt_str:
                    # Simulate answering a query - needs to *look* at context in prompt
                    if "list apis" in prompt_str or "endpoints" in prompt_str:
                        return "Simulating answer: Available APIs include getPlaceholder, createPlaceholder."
                    else:
                        return "Simulating answer: I can answer questions about the loaded API or the simulated graph/plan."
                if "create a simple, sequential plan" in prompt_str:
                     # Simulate plan JSON
                     return json.dumps({"plan": ["createPlaceholder", "getPlaceholder"], "description": "Simulated plan to create and get placeholder."})
                if "Simulate the response of an API call" in prompt_str:
                     # Simulate API response JSON - try to extract operationId
                     match = re.search(r"operationId: (\w+)", prompt_str)
                     op_id = match.group(1) if match else "unknown_op"
                     if op_id == "createPlaceholder":
                         return json.dumps({"id": "sim-123", "status": "created"})
                     elif op_id == "getPlaceholder":
                         # Simulate using previous results if available in prompt context
                         if "sim-123" in prompt_str: # Crude check for previous ID
                              return json.dumps({"id": "sim-123", "data": "simulated data for 123"})
                         else:
                              return json.dumps({"id": "sim-abc", "data": "simulated data"})
                     else:
                         return json.dumps({"status": "simulated success", "operation": op_id})


                # Default response for other LLM calls
                return "Simulating LLM response for an unspecified task."

            # For LangChain integration, ensure other required methods (_call, ainvoke, etc.) are present if needed
            # For async streaming, ainvoke is necessary.
            # Since LangGraph astream is used, ainvoke is important for non-blocking behavior.
            async def ainvoke(self, prompt: Any, **kwargs) -> Any:
                 # Simulate async behavior
                 import asyncio
                 await asyncio.sleep(0.05) # Small delay
                 # Call the sync invoke method
                 sync_result = self.invoke(prompt, **kwargs)
                 return sync_result # Return the simulated LLM text output


        # Instantiate PlaceholderLLMs if real ones couldn't be initialized
        if router_llm is None:
             router_llm = PlaceholderLLM("RouterLLM")
        if worker_llm is None:
             worker_llm = PlaceholderLLM("WorkerLLM")

    # --- END OF PLACEHOLDER FALLBACK ---


    # Final check to ensure LLMs were initialized (either real or placeholder)
    if router_llm is None or worker_llm is None:
         raise RuntimeError("Failed to initialize both real and placeholder LLMs.")

    # Validate that the LLMs have the required methods for the graph (invoke/ainvoke)
    if not hasattr(router_llm, 'invoke') or not hasattr(worker_llm, 'invoke'):
        raise TypeError("Initialized LLMs must have an 'invoke' method.")
    # If using astream, they must also have ainvoke
    if not hasattr(router_llm, 'ainvoke') or not hasattr(worker_llm, 'ainvoke'):
         logger.warning("LLMs do not have 'ainvoke' method. Adding dummy ainvoke for compatibility with astream.")
         # Add dummy ainvoke if missing to prevent errors with astream
         if not hasattr(router_llm, 'ainvoke'): router_llm.ainvoke = lambda p, **k: router_llm.invoke(p, **k)
         if not hasattr(worker_llm, 'ainvoke'): worker_llm.ainvoke = lambda p, **k: worker_llm.invoke(p, **k)


    logger.info("LLM clients initialized.")
    return router_llm, worker_llm

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Serve Static Files and Templates (Optional) ---
# Serve files from a 'static' directory (e.g., for HTML, CSS, JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# Set up templates directory (e.g., for index.html)
# templates = Jinja2Templates(directory="templates")

# --- Global LangGraph Instance ---
# Initialize LLMs and build the graph once on startup
router_llm_instance: Any = None
worker_llm_instance: Any = None
langgraph_app: Any = None
# Use MemorySaver for in-memory state persistence per session.
# Replace with a persistent checkpointer (e.g., SQL) for production.
checkpointer = MemorySaver()

@app.on_event("startup")
async def startup_event():
    """Initializes LLMs and builds the LangGraph application on FastAPI startup."""
    global router_llm_instance, worker_llm_instance, langgraph_app
    logger.info("FastAPI startup event: Initializing LLMs and building graph...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        # Pass the checkpointer when compiling the app to enable state persistence
        langgraph_app = build_graph(router_llm=router_llm_instance, worker_llm=worker_llm_instance).compile(checkpointer=checkpointer)
        logger.info("LangGraph application compiled successfully with MemorySaver checkpointer.")
    except Exception as e:
        logger.critical(f"Failed to initialize LLMs or build graph on startup: {e}", exc_info=True)
        # Depending on your needs, you might want to raise the exception
        # or set a flag to indicate the service is not fully operational.
        # For this example, we'll just log and continue, but requests might fail.
        langgraph_app = None # Ensure app is None if build fails

@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources on FastAPI shutdown (if needed)."""
    logger.info("FastAPI shutdown event.")
    # If using diskcache, consider explicitly closing it if needed (MemorySaver doesn't need close)
    # if 'utils' in sys.modules and hasattr(utils, 'SCHEMA_CACHE') and utils.SCHEMA_CACHE:
    #      try:
    #          utils.SCHEMA_CACHE.close()
    #          logger.info("Schema cache closed.")
    #      except Exception as e:
    #          logger.error(f"Error closing schema cache: {e}")


# --- WebSocket Endpoint ---
@app.websocket("/ws/submit_openapi")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for OpenAPI analysis."""
    await websocket.accept()
    logger.info(f"WebSocket accepted connection from {websocket.client.host}:{websocket.client.port}")

    # Generate a unique session ID for this WebSocket connection
    # This will be used as the thread_id for LangGraph's checkpointer
    # Can also try to get a user identifier if available in a real auth scenario
    session_id = str(uuid.uuid4())
    logger.info(f"Assigned session ID: {session_id}")

    # Provide initial welcome message
    await websocket.send_json({"type": "info", "content": f"Connected. Session ID: {session_id}. Please provide an OpenAPI spec (JSON/YAML) or ask a question."})

    if langgraph_app is None:
         error_msg = "Backend initialization failed. Cannot process requests."
         logger.error(error_msg)
         await websocket.send_json({"type": "error", "content": error_msg})
         await websocket.close(code=1011) # Internal Error
         return # Exit the handler if app failed to build

    try:
        while True:
            # Receive message from client (assuming text for simplicity for now)
            data = await websocket.receive_text()
            user_input = data.strip()

            if not user_input:
                logger.warning("Received empty message from client.")
                await websocket.send_json({"type": "warning", "content": "Received empty message. Please provide input."})
                continue

            logger.info(f"Received message for session {session_id}: '{user_input[:200]}...'") # Log snippet
            await websocket.send_json({"type": "info", "content": "Processing your request..."}) # Acknowledge receipt

            # Prepare input for LangGraph
            # The initial input to astream is the state object itself for nodes that expect state
            config = {"configurable": {"thread_id": session_id}}

            # Load the latest state for this thread_id from the checkpointer
            # The checkpointer's get method returns a dictionary representing the state, or {} if none exists
            thread_state_dict = checkpointer.get(config.get("configurable", {})).get("channel_values", {}) if checkpointer else {}

            # Create or load BotState
            loaded_state = None
            if thread_state_dict:
                # Try to load from checkpoint state dictionary using Pydantic model_validate
                try:
                    loaded_state = BotState.model_validate(thread_state_dict)
                    loaded_state.user_input = user_input # Update with current user input
                    # Clear previous intermediate response and next_step for the new turn
                    loaded_state.response = None
                    loaded_state.next_step = None
                    # Clear last sent intermediate message tracker from scratchpad for the new turn
                    if loaded_state.scratchpad and "last_sent_intermediate_response" in loaded_state.scratchpad:
                         del loaded_state.scratchpad["last_sent_intermediate_response"]

                    logger.debug(f"Loaded state for session {session_id} from checkpoint.")
                except ValidationError as e:
                    logger.error(f"Failed to validate loaded state for session {session_id}: {e}. Starting with new state.", exc_info=True)
                    # If validation fails, start fresh
                    loaded_state = BotState(session_id=session_id, user_input=user_input)
                except Exception as e:
                     logger.error(f"Unexpected error loading state for session {session_id}: {e}. Starting with new state.", exc_info=True)
                     loaded_state = BotState(session_id=session_id, user_input=user_input)
            else:
                # No existing state found for this thread_id, create new
                loaded_state = BotState(session_id=session_id, user_input=user_input)
                logger.debug(f"No checkpoint state found for session {session_id}. Starting with new state.")


            # The input to `astream` is the initial state object for the first node (router)
            current_state_input = loaded_state

            # Use astream for asynchronous streaming in FastAPI
            # stream_mode="values" yields the full state object (or dict representation) after each node
            # The graph nodes now return BotState objects, which astream yields.
            final_state_snapshot: Optional[BotState] = None # Keep track of the latest state object yielded

            try:
                # astream yields the state after each node completes
                async for intermediate_state_dict in langgraph_app.astream(current_state_input, config=config, stream_mode="values"):
                     # intermediate_state_dict is the dictionary representation of the BotState object
                     # returned by the node. We need to validate it back to a BotState model.
                     try:
                         intermediate_state = BotState.model_validate(intermediate_state_dict)
                         final_state_snapshot = intermediate_state # Keep track of the latest valid state

                         # Check for intermediate response messages set by nodes
                         # Only send if the response field has been updated and is not None
                         response_message = intermediate_state.response
                         # Use a scratchpad key to track the last sent intermediate response
                         # Note: This tracking in scratchpad might not be perfectly reliable with streaming
                         # as the scratchpad update might not be flushed immediately.
                         last_sent_response = intermediate_state.scratchpad.get("last_sent_intermediate_response_sent") # Use a different key

                         # Simple check to avoid sending the same message repeatedly within one turn
                         # This is imperfect with streaming but better than nothing.
                         # A more robust approach might involve tracking response hashes or sequence numbers.
                         if response_message and response_message != last_sent_response:
                             logger.debug(f"Sending intermediate message for session {session_id}: {response_message[:200]}...")
                             await websocket.send_json({"type": "intermediate", "content": response_message})
                             # Update scratchpad to mark this message as sent (imperfectly)
                             # This update might not be saved until the end of the turn.
                             intermediate_state.scratchpad["last_sent_intermediate_response_sent"] = response_message
                             # Note: The state object is passed by reference, so modifying intermediate_state
                             # here *does* modify the state object that will be potentially checkpointed.


                     except ValidationError as e:
                         logger.error(f"Failed to validate intermediate state from stream for session {session_id}: {e}. Skipping this state update.", exc_info=True)
                         await websocket.send_json({"type": "warning", "content": f"Received invalid state update from backend: {e}"})
                     except Exception as e:
                         logger.error(f"Unexpected error processing intermediate state for session {session_id}: {e}", exc_info=True)
                         await websocket.send_json({"type": "warning", "content": f"Unexpected error processing state update: {e}"})


                # After the stream finishes, process the final state snapshot
                if final_state_snapshot and isinstance(final_state_snapshot, BotState):
                     # The responder node should have put the final user-facing message here
                     final_response = final_state_snapshot.final_response

                     # Send the final response message
                     if final_response:
                         logger.info(f"Sending final response for session {session_id}: {final_response[:200]}...")
                         await websocket.send_json({"type": "final", "content": final_response})
                     else:
                         # Fallback if final_response isn't set (e.g., error before responder)
                         logger.warning(f"Graph execution finished for session {session_id}, but 'final_response' was empty.")
                         # Check if there was an intermediate response that wasn't marked final
                         fallback_response = final_state_snapshot.response if final_state_snapshot.response else "Processing finished, but no specific final result message was generated."
                         await websocket.send_json({"type": "warning", "content": fallback_response})

                     # Optionally send other final state info if needed, e.g., the graph JSON
                     # if final_state_snapshot.execution_graph:
                     #      try:
                     #          graph_json = final_state_snapshot.execution_graph.model_dump_json(indent=2)
                     #          await websocket.send_json({"type": "graph_json", "content": graph_json})
                     #      except Exception as e:
                     #          logger.error(f"Error sending final graph JSON for session {session_id}: {e}")
                     #          await websocket.send_json({"type": "warning", "content": f"Could not serialize final graph to JSON: {e}"})

                     # Clear the last sent intermediate response tracker from scratchpad for the next turn
                     if final_state_snapshot.scratchpad and "last_sent_intermediate_response_sent" in final_state_snapshot.scratchpad:
                          del final_state_snapshot.scratchpad["last_sent_intermediate_response_sent"]


                else:
                     logger.error(f"Graph execution finished for session {session_id} without a valid final state object.")
                     await websocket.send_json({"type": "error", "content": "Internal error: Failed to get final processing state."})


            except Exception as e:
                # Catch exceptions during graph execution (e.g., LLM call errors)
                logger.critical(f"Error during LangGraph execution for session {session_id}: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"An error occurred during processing: {e}"})
                # Decide if you want to close the connection on error or continue
                # await websocket.close(code=1011) # Example: Close on internal error
                # break # Example: Break loop on error

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        # Catch any other exceptions in the WebSocket loop itself
        logger.critical(f"Unexpected error in WebSocket loop for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": f"An unexpected error occurred: {e}"})
        except:
            pass # Ignore errors sending error message
    finally:
        # Ensure WebSocket is closed if not already
        try:
            await websocket.close()
        except:
            pass # Ignore errors closing


# --- How to Run ---
# 1. Save the files: models.py, utils.py, core_logic.py, router.py, graph.py, main.py, requirements.txt
# 2. Install dependencies: pip install -r requirements.txt
# 3. Set your LLM API key in a .env file in the same directory (e.g., OPENAI_API_KEY=your_key_here)
# 4. Run the FastAPI application: uvicorn main:app --reload
# 5. Connect to the WebSocket endpoint ws://localhost:8000/ws/submit_openapi from a client.
#    You can use a simple HTML page with JavaScript or a WebSocket testing tool.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# filename: models.py
import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Literal, Union
from pydantic import BaseModel, Field, ValidationError, model_validator # Import model_validator for Pydantic v2
from datetime import datetime # Import datetime for timestamping

# Module-level logger
logger = logging.getLogger(__name__)

# --- Graph Representation Models ---

# Define a model for input mapping instructions (useful for describing the plan)
class InputMapping(BaseModel):
    """Defines how to map data from previous results to a parameter of this node (as described in a plan)."""
    source_operation_id: str = Field(..., description="The operationId or effective_id of the previous node whose described result contains the source data.")
    source_data_path: str = Field(..., description="A path or expression (e.g., JSONPath like '$.id') to extract the data from the source node's described result.")
    target_parameter_name: str = Field(..., description="The name of the parameter/field in the current node's operation that this data maps to.")
    # Optional: Add parameter 'in' (path, query, header, cookie, body) for clarity/validation
    target_parameter_in: Optional[Literal["path", "query", "header", "cookie", "body"]] = Field(None, description="The location of the target parameter (path, query, header, cookie, body).")
    # Optional: Add transformation instructions if needed (e.g., format date)
    transformation: Optional[str] = Field(None, description="Optional instructions for transforming the data before mapping.")


class Node(BaseModel):
    """Represents a node (an API call description) in the execution graph."""
    operationId: str = Field(..., description="Unique identifier for the API operation (from OpenAPI spec).")
    display_name: Optional[str] = Field(None, description="A unique name for this specific node instance (e.g., 'createUser_step1'), required if using the same operationId multiple times in one graph.")
    summary: Optional[str] = Field(None, description="Short summary of the operation (from OpenAPI spec).")
    description: Optional[str] = Field(None, description="Detailed description of the operation.")
    payload_description: Optional[str] = Field(None, description="A string description of an example payload for this API call.")
    input_mappings: List[InputMapping] = Field(default_factory=list, description="Instructions on how data would be mapped from previous described results.")
    # Add fields to store actual parameters/request body descriptions if needed for description refinement
    parameters: Optional[List[Dict[str, Any]]] = Field(None, description="Description or example of parameters for this operation.")
    request_body: Optional[Dict[str, Any]] = Field(None, description="Description or example of the request body for this operation.")


    # Add a computed property or method to get the effective node ID for graph structure
    # This ensures edges/cycle checks use a unique identifier
    @property
    def effective_id(self) -> str:
        """Returns the unique identifier for this node instance in the graph."""
        return self.display_name if self.display_name else self.operationId

    # Pydantic v2 model_validator to ensure display_name is set if operationId is duplicated in a list of nodes
    # This validation would typically happen in the GraphOutput model, not Node itself,
    # as it requires context of other nodes in the list.
    # Keeping it simple for now, relying on LLM to provide display_name when needed.
    # @model_validator(mode='after')
    # def check_display_name_if_duplicated(self) -> 'Node':
    #     # This validation is tricky at the individual node level.
    #     # It's better done at the GraphOutput level after all nodes are parsed.
    #     return self


class Edge(BaseModel):
    """Represents a directed edge (dependency) in the execution graph description."""
    # Edges should now reference the effective_id (operationId or display_name)
    from_node: str = Field(..., description="The effective_id (operationId or display_name) of the source node.")
    to_node: str = Field(..., description="The effective_id (operationId or display_name) of the target node.")
    description: Optional[str] = Field(None, description="Optional description of why this dependency exists (e.g., data dependency).")
    # input_mapping is moved to Node in the new structure, but keep it here if edges also describe mappings
    # Decided to keep input_mappings only on the Node model based on the GraphOutput model prompt

    # Make Edge hashable for use in sets (use effective_id tuple)
    def __hash__(self):
        return hash((self.from_node, self.to_node))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        return self.from_node == other.from_node and self.to_node == other.to_node

class GraphOutput(BaseModel):
    """Represents the generated API execution graph description."""
    nodes: List[Node] = Field(default_factory=list, description="List of API operations (nodes) in the graph description.")
    edges: List[Edge] = Field(default_factory=list, description="List of dependencies (edges) between nodes in the graph description.")
    description: Optional[str] = Field(None, description="Natural language description of the overall workflow.")

    @model_validator(mode='after')
    def check_unique_node_ids(self) -> 'GraphOutput':
        """Validates that node effective_ids are unique within the graph."""
        node_ids = {}
        for node in self.nodes:
            if node.effective_id in node_ids:
                raise ValueError(f"Duplicate node effective_id found: '{node.effective_id}'. Use 'display_name' for duplicate operationIds.")
            node_ids[node.effective_id] = True
        return self

    @model_validator(mode='after')
    def check_edge_nodes_exist(self) -> 'GraphOutput':
        """Validates that edge source and target nodes exist in the nodes list."""
        node_effective_ids = {node.effective_id for node in self.nodes}
        for edge in self.edges:
            if edge.from_node not in node_effective_ids:
                raise ValueError(f"Edge source node '{edge.from_node}' not found in the list of nodes.")
            if edge.to_node not in node_effective_ids:
                raise ValueError(f"Edge target node '{edge.to_node}' not found in the list of nodes.")
        return self


# --- Tool Parameter Models (Keep for potential future use by planner/update_graph) ---
# These models could be used by an LLM to structure parameters for specific actions

class AddEdgeParams(BaseModel):
    """Parameters required for the add_edge tool (or planning step)."""
    from_node: str = Field(..., description="The operationId or display_name of the source node.")
    to_node: str = Field(..., description="The operationId or display_name of the target node.")
    description: Optional[str] = Field(None, description="Optional description for the new edge.")
    # Add input_mapping field if edges can define mappings
    input_mapping: List[InputMapping] = Field(default_factory=list, description="Instructions on how data would be mapped for this edge.")


class GeneratePayloadsParams(BaseModel):
    """Parameters/Instructions for generating payloads (descriptions)."""
    instructions: Optional[str] = Field(None, description="Specific user instructions for how payloads should be described.")
    target_apis: Optional[List[str]] = Field(None, description="Optional list of specific operationIds to describe payloads for.")

class GenerateGraphParams(BaseModel):
    """Parameters/Instructions for generating the execution graph description."""
    goal: Optional[str] = Field(None, description="The overall user goal or task to accomplish with the described API workflow.")
    instructions: Optional[str] = Field(None, description="Specific user instructions for how the graph should be structured.")

class PlanExecutionParams(BaseModel):
    """Parameters/Instructions for the planner."""
    goal: str = Field(..., description="The user's goal or task to create a plan for.")
    context: Optional[str] = Field(None, description="Additional context provided by the user or system.")

class ExecutePlanStepParams(BaseModel):
    """Parameters for executing a single step in the plan."""
    operation_id: str = Field(..., description="The operationId of the API call or action to simulate.")
    # Add fields for specific parameters or request body needed for the simulation step
    # These would be populated by the planner or previous execution steps
    parameters: Optional[Dict[str, Any]] = Field(None, description="Simulated parameters for the API call.")
    request_body: Optional[Dict[str, Any]] = Field(None, description="Simulated request body for the API call.")
    # Add a field to hold the simulated result of this step
    simulated_result: Optional[Dict[str, Any]] = Field(None, description="The simulated result (JSON-like) of executing this step.")


# --- State Model ---

class BotState(BaseModel):
    """Represents the full state of the conversation and processing."""
    session_id: str = Field(..., description="Unique identifier for the current session.")
    user_input: Optional[str] = Field(None, description="The latest input from the user.")

    # OpenAPI Specification related fields
    openapi_spec_string: Optional[str] = Field(None, description="Temporary storage for the raw OpenAPI specification text provided by the user in the current turn. Cleared after parsing attempt.") # Added temporary field
    openapi_spec_text: Optional[str] = Field(None, description="The raw OpenAPI specification text that was successfully parsed.") # Store successfully parsed text
    openapi_schema: Optional[Dict[str, Any]] = Field(None, description="The parsed and resolved OpenAPI schema as a dictionary.")
    schema_cache_key: Optional[str] = Field(None, description="Cache key used for the current schema.")
    schema_summary: Optional[str] = Field(None, description="LLM-generated text summary of the OpenAPI schema.")
    # Flag to indicate if the current input is likely a spec (set by router)
    input_is_spec: bool = Field(False, description="Flag indicating if the last user input was identified as an OpenAPI spec.")


    # API Identification and Payload Generation (Descriptions)
    # Store full identified API details including parameters/requestBody for context
    identified_apis: List[Dict[str, Any]] = Field(default_factory=list, description="List of APIs identified from the spec, including method, path, summary, parameters, requestBody.")
    payload_descriptions: Dict[str, str] = Field(default_factory=dict, description="Dictionary mapping operationId to generated example payload descriptions (string).")
    payload_generation_instructions: Optional[str] = Field(None, description="User instructions captured for payload description.")

    # Execution Graph Description (Represents a potential workflow structure)
    execution_graph: Optional[GraphOutput] = Field(None, description="The generated API execution graph description (represents a potential workflow structure).")
    graph_generation_instructions: Optional[str] = Field(None, description="User instructions captured for graph description.")
    # Field to hold reason for graph regeneration, if verification fails
    graph_regeneration_reason: Optional[str] = Field(None, description="Reason why the graph needs to be regenerated (e.g., cycle detected, missing APIs).")


    # Plan and Execute Fields
    # The plan is now a list of operationIds or step descriptions
    execution_plan: List[str] = Field(default_factory=list, description="Ordered list of operationIds or step descriptions for the planned execution.")
    current_plan_step: int = Field(0, description="Index of the current step in the execution_plan.")
    plan_execution_goal: Optional[str] = Field(None, description="The user's goal that initiated the current plan execution.")


    # Routing and Control Flow
    intent: Optional[str] = Field(None, description="The user's high-level intent as determined by the initial router LLM.")
    # Previous intent is now tracked implicitly by the graph history or explicitly in scratchpad if needed
    loop_counter: int = Field(0, description="Counter to detect potential loops in routing.")

    # Parameters extracted by the initial router or the planner (e.g., for update_graph or execute_plan_step)
    extracted_params: Optional[Dict[str, Any]] = Field(None, description="Parameters extracted by the router or planner for the current action.")

    # --- Responder Fields ---
    final_response: str = Field("", description="The final, user-facing response generated by the responder.")

    # Output and Communication (Intermediate messages from core_logic nodes)
    response: Optional[str] = Field(None, description="Intermediate response message set by nodes (e.g., 'Schema parsed successfully'). Cleared by responder.")

    # LangGraph internal key for routing - exclude from serialization
    next_step: Optional[str] = Field(
        None,
        alias="__next__",
        exclude=True,
        description="Internal: the next LangGraph node to execute, set by router or nodes."
    )

    # Internal working memory - useful for storing intermediate results, simulated outputs, etc.
    scratchpad: Dict[str, Any] = Field(default_factory=dict, description="Persistent memory for intermediate results, reasoning, history, planner decisions etc.")

    class Config:
        # Allow extra fields in scratchpad without validation errors
        extra = 'allow'
        # Enforce type validation on assignment to fields
        validate_assignment = True
        # Allow populating by field name as well as alias (__next__)
        populate_by_name = True


    def update_scratchpad_reason(self, tool_name: str, details: str):
        """Helper to append reasoning/details to the scratchpad."""
        # Ensure scratchpad is a dict
        if not isinstance(self.scratchpad, dict):
             self.scratchpad = {}
             logger.warning("Scratchpad was not a dictionary, re-initialized.")

        current_reason_log = self.scratchpad.get('reasoning_log', [])
        # Ensure reason_log is a list
        if not isinstance(current_reason_log, list):
             current_reason_log = []
             logger.warning("Scratchpad['reasoning_log'] was not a list, re-initialized.")

        timestamp = datetime.now().isoformat()
        new_entry = {"timestamp": timestamp, "tool": tool_name, "details": details}
        current_reason_log.append(new_entry)
        # Keep log size manageable, e.g., last 100 entries
        self.scratchpad['reasoning_log'] = current_reason_log[-100:]

        # Optionally also store a simple string log for easier viewing
        current_reason_string = self.scratchpad.get('reasoning_log_string', '')
         # Ensure reason_log_string is a string
        if not isinstance(current_reason_string, str):
             current_reason_string = ""
             logger.warning("Scratchpad['reasoning_log_string'] was not a string, re-initialized.")

        new_string_entry = f"\n---\n[{timestamp}] Tool: {tool_name}\nDetails: {details}\n---\n"
        # Keep string log size manageable, e.g., last 10000 characters
        self.scratchpad['reasoning_log_string'] = (current_reason_string + new_string_entry)[-10000:]
        logger.debug(f"Scratchpad Updated by {tool_name}: {details}")


--------------------------------------------------------------------------------------------------------------------------------
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

--------------------------------------------------------------------------------------------------------------------------------

# filename: utils.py
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, ValidationError
import diskcache # Import diskcache

# Assuming models.py defines GraphOutput, Node, Edge, BaseModel
# Need to import BaseModel if expected_model type hint is used directly
# from pydantic import BaseModel
# Assuming models.py is available
try:
    # Import necessary models and BaseModel from your models.py
    from models import GraphOutput, Node, Edge, BaseModel
except ImportError:
    # Define dummy classes if models.py is not found, to allow utils.py to load
    # This is primarily for linting/basic checks if utils is run standalone
    logger.warning("Could not import models from models.py. Using dummy classes for basic type hinting.")
    class GraphOutput:
        def __init__(self, nodes=None, edges=None):
            self.nodes = nodes or []
            self.edges = edges or []
    class Node:
        def __init__(self, operationId="dummy", display_name=None):
            self.operationId = operationId
            self.display_name = display_name
        @property
        def effective_id(self):
            return self.display_name if self.display_name else self.operationId
    class Edge:
         def __init__(self, from_node, to_node):
             self.from_node = from_node
             self.to_node = to_node
         def __hash__(self): return hash((self.from_node, self.to_node))
         def __eq__(self, other):
             if not isinstance(other, Edge): return False
             return self.from_node == other.from_node and self.to_node == other.to_node
    class BaseModel: # Dummy for type hint if needed
         @classmethod
         def model_validate(cls, data): return data
         def model_dump(self): return self.__dict__
         def model_dump_json(self, indent=None): return json.dumps(self.__dict__, indent=indent)


# Module-level logger
# Configure basic logging if not already configured by the main app
if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Persistent Caching Setup ---
# Use a dedicated cache directory in the current working directory or a temporary one
# Ensure the directory exists
CACHE_DIR = os.path.join(os.getcwd(), ".openapi_cache")
os.makedirs(CACHE_DIR, exist_ok=True) # Create directory if it doesn't exist

SCHEMA_CACHE = None
try:
    # Use a context manager or explicit close in a real application
    SCHEMA_CACHE = diskcache.Cache(CACHE_DIR)
    logger.info(f"Initialized persistent schema cache at: {CACHE_DIR}")
except Exception as e:
    logger.error(f"Failed to initialize disk cache at {CACHE_DIR}: {e}. Caching will not work.", exc_info=True)
    SCHEMA_CACHE = None # Disable cache if initialization fails

def get_cache_key(spec_text: str) -> str:
    """Generates a cache key based on the hash of the spec text."""
    # Use a more robust hash like SHA-256 for longer keys less prone to collisions
    return hashlib.sha256(spec_text.encode('utf-8')).hexdigest()

def load_cached_schema(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads a parsed schema from the persistent cache if it exists."""
    if SCHEMA_CACHE is None:
        return None
    try:
        # Cache stores (key, value, expire) - get returns value or default
        schema = SCHEMA_CACHE.get(cache_key, default=None)
        if schema:
            logger.debug(f"Cache hit for key: {cache_key}")
            return schema
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading schema from cache (key: {cache_key}): {e}", exc_info=True)
        return None

def save_schema_to_cache(cache_key: str, schema: Dict[str, Any]):
    """Saves a parsed schema to the persistent cache."""
    if SCHEMA_CACHE is None:
        return
    try:
        # Set with no expiration by default
        SCHEMA_CACHE.set(cache_key, schema)
        logger.debug(f"Saved schema to cache with key: {cache_key}")
    except Exception as e:
        logger.error(f"Error saving schema to cache (key: {cache_key}): {e}", exc_info=True)


# --- Graph Utilities ---

def check_for_cycles(graph: GraphOutput) -> Tuple[bool, str]:
    """
    Checks if the given execution graph is a Directed Acyclic Graph (DAG).
    Uses the effective_id of nodes for checks.
    Returns a tuple: (is_dag, cycle_message).
    """
    # Ensure graph and graph.nodes are valid before proceeding
    if not isinstance(graph, GraphOutput) or not isinstance(graph.nodes, list):
         logger.warning("Invalid graph object passed to check_for_cycles.")
         return False, "Invalid graph structure provided." # Treat invalid graph as potentially cyclic

    # Build adjacency list using effective_id from valid Node objects
    adj: Dict[str, List[str]] = {}
    node_ids = set()

    for node in graph.nodes:
         if isinstance(node, Node):
              node_id = node.effective_id
              node_ids.add(node_id)
              adj[node_id] = [] # Initialize adjacency list for this node

    if not node_ids:
        return True, "Graph is empty or has no valid nodes."


    if not isinstance(graph.edges, list):
         logger.warning("Graph edges attribute is not a list in check_for_cycles.")
         return False, "Invalid graph edges structure." # Treat invalid graph as potentially cyclic

    for edge in graph.edges:
        if isinstance(edge, Edge) and hasattr(edge, 'from_node') and hasattr(edge, 'to_node'):
            # Edge nodes must exist as effective_ids in the adjacency list keys
            # (which ensures they were valid Node objects)
            if edge.from_node in adj and edge.to_node in node_ids:
                 # Prevent adding duplicate edges if they exist in input
                 if edge.to_node not in adj[edge.from_node]:
                      adj[edge.from_node].append(edge.to_node)
            else:
                 logger.warning(f"Skipping edge in cycle check: Source or target node not found as a valid node effective_id: {edge.from_node} -> {edge.to_node}")
        else:
             logger.warning(f"Skipping invalid edge object in cycle check: {edge}")


    visited: Dict[str, bool] = {node_id: False for node_id in node_ids}
    recursion_stack: Dict[str, bool] = {node_id: False for node_id in node_ids}

    # Store the detected cycle path if found
    cycle_path: List[str] = []

    def dfs_cycle_check_recursive(node_id: str) -> bool:
        """Recursive helper for DFS cycle detection."""
        visited[node_id] = True
        recursion_stack[node_id] = True
        cycle_path.append(node_id) # Add to current path

        # Check neighbors
        for neighbor_id in adj.get(node_id, []):
            if not visited[neighbor_id]:
                if dfs_cycle_check_recursive(neighbor_id):
                    return True # Cycle found in deeper recursion
            elif recursion_stack[neighbor_id]:
                # Cycle detected: neighbor is already in the current recursion stack
                logger.error(f"Cycle detected: Edge from '{node_id}' to '{neighbor_id}' completes a cycle.")
                # Complete the cycle path
                try:
                    cycle_start_index = cycle_path.index(neighbor_id)
                    full_cycle = cycle_path[cycle_start_index:]
                    cycle_path[:] = full_cycle # Trim path to just the cycle
                except ValueError:
                    # Should not happen if neighbor_id is in recursion_stack, but as safeguard
                    logger.error(f"Error finding cycle start index for neighbor {neighbor_id} in path {cycle_path}")
                    cycle_path[:] = [f"Cycle involving {neighbor_id}"] # Fallback message
                return True

        # Remove node from recursion stack as we backtrack if no cycle found from this path
        recursion_stack[node_id] = False
        if cycle_path and cycle_path[-1] == node_id: # Only pop if this is the current path's end
             cycle_path.pop()
        return False

    # Iterate through all nodes to start DFS
    for node_id in node_ids:
        if not visited[node_id]:
            if dfs_cycle_check_recursive(node_id):
                cycle_message = f"Cycle detected: {' -> '.join(cycle_path)}"
                return False, cycle_message

    return True, "No cycles detected."


# --- LLM Call Helper ---

def llm_call_helper(llm: Any, prompt: Any) -> str:
    """
    Helper function to make an LLM call with basic logging and error handling.
    Accepts string or structured prompts (like lists of messages).

    Args:
        llm: The LLM instance with an 'invoke' method.
        prompt: The prompt (string or structured) to send to the LLM.

    Returns:
        The response content (usually text) from the LLM.

    Raises:
        Exception: Re-raises exceptions from the llm.invoke call.
    """
    # Limit prompt logging for privacy/verbosity
    prompt_repr = str(prompt)[:1000] + '...' if len(str(prompt)) > 1000 else str(prompt)
    logger.debug(f"Calling LLM with prompt: {prompt_repr}")
    try:
        # Assuming the LLM's invoke method returns an object with a 'content' attribute
        # (like AIMessage or ChatCompletion). Adjust if your LLM client returns differently.
        response_obj = llm.invoke(prompt)

        # Extract content based on typical LangChain patterns
        if hasattr(response_obj, 'content'):
             response_content = response_obj.content
        elif isinstance(response_obj, str):
             response_content = response_obj # Handle cases where invoke directly returns a string
        else:
             # Handle cases where response_obj might be a Pydantic model or other type
             # Attempt to convert to string or dump to JSON if it's a known model/dict
             try:
                 if isinstance(response_obj, BaseModel):
                     response_content = response_obj.model_dump_json(indent=2)
                 elif isinstance(response_obj, dict) or isinstance(response_obj, list):
                      response_content = json.dumps(response_obj, indent=2)
                 else:
                    logger.warning(f"LLM response object type ({type(response_obj)}) has no 'content' attribute and is not a string/dict/list/BaseModel. Returning raw object representation.")
                    response_content = str(response_obj)
             except Exception as inner_e:
                  logger.warning(f"Could not convert LLM response object to string/JSON: {inner_e}")
                  response_content = str(response_obj)


        # Ensure response is a string
        if not isinstance(response_content, str):
             logger.warning(f"LLM response content is not a string ({type(response_content)}). Converting to string.")
             response_content = str(response_content)

        # Limit response logging for privacy/verbosity
        response_repr = response_content[:1000] + '...' if len(response_content) > 1000 else response_content
        logger.debug(f"LLM call successful. Response: {response_repr}")
        return response_content
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        # Include prompt in error for debugging
        logger.error(f"Failed LLM prompt: {prompt_repr}")
        raise # Re-raise the exception so the calling node can handle it

# --- JSON Parsing Helper ---

def parse_llm_json_output_with_model(llm_output: str, expected_model: Optional[Type[BaseModel]] = None) -> Any:
    """
    Parses JSON output from an LLM string, handling potential markdown formatting
    and optional Pydantic model validation.

    Args:
        llm_output: The raw string output from the LLM.
        expected_model: An optional Pydantic model class (type) to validate against.

    Returns:
        The parsed JSON object (dict, list, or Pydantic model instance),
        or None if parsing or validation fails.
    """
    if not isinstance(llm_output, str):
        logger.error(f"Cannot parse non-string LLM output as JSON. Type: {type(llm_output)}")
        return None

    # Limit logging of raw LLM output for privacy/verbosity
    llm_output_repr = llm_output[:1000] + '...' if len(llm_output) > 1000 else llm_output
    logger.debug(f"Attempting to parse LLM output as JSON: {llm_output_repr}")

    json_block = llm_output.strip()

    # Attempt to extract JSON block if LLM wrapped it in markdown code fences
    # Robustly handle various markdown fences and optional language identifier
    if json_block.startswith('```'):
        end_fence_pos = json_block.rfind('```')
        if end_fence_pos > 0: # Ensure closing fence exists and is not at the start
            first_newline_pos = json_block.find('\n')
            if first_newline_pos != -1 and first_newline_pos < end_fence_pos:
                 # Content starts after the first newline following the opening fence
                 json_block = json_block[first_newline_pos + 1:end_fence_pos].strip()
                 logger.debug("Extracted content between markdown fences.")
            else:
                 # No newline after ``` or newline is after the closing fence,
                 # maybe just ``` without language or newline? Unlikely but handle.
                 # Or closing fence is missing/malformed.
                 logger.warning("Could not find standard markdown fence structure. Attempting to parse full block.")
                 # Fallback: remove only the opening fence and any leading/trailing whitespace
                 json_block = json_block[3:].strip()
                 if json_block.endswith('```'): # If it had a closing fence but no newline
                      json_block = json_block[:-3].strip()


    # Sometimes LLMs might just output the JSON without fences
    # Basic check if it looks like JSON (more permissive than before)
    looks_like_json = (json_block.startswith('{') and json_block.endswith('}')) or \
                      (json_block.startswith('[') and json_block.endswith(']')) or \
                      (json_block.strip().startswith('{') and json_block.strip().endswith('}')) or \
                      (json_block.strip().startswith('[') and json_block.strip().endswith(']'))

    if not looks_like_json:
        logger.warning("LLM output doesn't look like a standard JSON object or array (missing typical start/end chars).")
        # Continue attempting to parse, as it might be valid JSON without typical start/end

    try:
        # Attempt to parse the JSON
        parsed_data = json.loads(json_block)
        logger.debug("Successfully parsed JSON.")

        # If an expected Pydantic model class is provided, validate the data
        # Check if expected_model is actually a subclass of BaseModel and is not the dummy class
        # Use hasattr to check for model_validate as a more robust check than __name__
        if expected_model and issubclass(expected_model, BaseModel) and hasattr(expected_model, 'model_validate'):
            logger.debug(f"Validating parsed JSON against model: {expected_model.__name__}")
            try:
                # Use model_validate for data validation, including type coercion
                validated_data = expected_model.model_validate(parsed_data)
                logger.debug("JSON validated successfully against model.")
                return validated_data # Return the validated model instance
            except ValidationError as e:
                logger.error(f"Pydantic validation failed against model {expected_model.__name__}: {e}")
                # Log the problematic data on debug level
                logger.debug(f"Data that failed validation: {parsed_data}")
                return None # Validation failed
        else:
            # No validation requested or invalid/dummy model provided, return raw parsed data
            if expected_model:
                 logger.warning(f"expected_model ({expected_model}) is not a valid Pydantic model type. Skipping validation.")
            return parsed_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}. Problematic text snippet: '{json_block[max(0, e.pos-20):min(len(json_block), e.pos+20)]}'", exc_info=False) # Log context around error
        logger.debug(f"Full text attempted parsing: {json_block}") # Log full text on debug level
        return None
    except Exception as e:
        # Catch any other unexpected errors (e.g., during validation if not ValidationError)
        logger.error(f"An unexpected error occurred during JSON parsing/validation: {e}", exc_info=True)
        return None


--------------------------------------------------------------------------------------------------------------------------------
uvicorn main:app --reload --host 0.0.0.0 --port 8000