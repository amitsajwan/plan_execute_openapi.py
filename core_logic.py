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

