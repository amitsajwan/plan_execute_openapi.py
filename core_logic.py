# core_logic.py
import json
import logging
from typing import Any, Dict, List, Optional
import yaml

from models import BotState, GraphOutput, Node, InputMapping 
from utils import (
    llm_call_helper, load_cached_schema, save_schema_to_cache,
    get_cache_key, check_for_cycles, parse_llm_json_output_with_model,
    SCHEMA_CACHE 
)
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class OpenAPICoreLogic:
    def __init__(self, worker_llm: Any):
        if not hasattr(worker_llm, 'invoke'):
            raise TypeError("worker_llm must have an 'invoke' method.")
        self.worker_llm = worker_llm
        logger.info("OpenAPICoreLogic initialized.")

    def parse_openapi_spec(self, state: BotState) -> BotState:
        tool_name = "parse_openapi_spec"
        state.response = "Parsing OpenAPI specification..." 
        state.update_scratchpad_reason(tool_name, "Attempting to parse OpenAPI spec.")
        # logger.info("Executing parse_openapi_spec node.") 

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
                if state.execution_graph:
                    try:
                        state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                    except Exception as e:
                        logger.error(f"Error serializing cached graph for sending: {e}")
                state.next_step = "responder"
            return state

        parsed_schema = None
        error_message = None
        try:
            parsed_schema = json.loads(spec_text)
        except json.JSONDecodeError:
            try:
                parsed_schema = yaml.safe_load(spec_text)
            except yaml.YAMLError as yaml_e: error_message = f"YAML parsing failed: {yaml_e}"
            except Exception as e_yaml: error_message = f"Unexpected error during YAML parsing: {e_yaml}"
        except Exception as e_json: error_message = f"Unexpected error during JSON parsing: {e_json}"

        if parsed_schema and isinstance(parsed_schema, dict) and \
           ('openapi' in parsed_schema or 'swagger' in parsed_schema) and 'info' in parsed_schema:
            state.openapi_schema = parsed_schema
            state.schema_cache_key = cache_key
            state.openapi_spec_text = spec_text
            state.openapi_spec_string = None
            if SCHEMA_CACHE: save_schema_to_cache(cache_key, parsed_schema) 
            else: logger.warning("SCHEMA_CACHE is None, schema not saved to disk cache.")
            logger.info("Successfully parsed OpenAPI spec.")
            state.response = "OpenAPI specification parsed. Starting analysis pipeline..." 
            state.next_step = "process_schema_pipeline"
        else:
            state.openapi_schema = None; state.openapi_spec_string = None
            final_error = error_message or "Parsed content is not a valid OpenAPI/Swagger spec."
            state.response = f"Failed to parse specification: {final_error}" 
            logger.error(f"Parsing failed: {final_error}")
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

        spec, info = state.openapi_schema, state.openapi_schema.get('info', {})
        paths_preview = "\n".join([f"  {p}: {list(m.keys()) if isinstance(m, dict) else '[?]'}" for p, m in list(spec.get('paths', {}).items())[:3]])
        summary_prompt = f"Summarize API: {info.get('title', 'N/A')} (v{info.get('version', 'N/A')}). Desc: {info.get('description', 'N/A')[:200]}... Paths: {len(spec.get('paths', {}))}. Preview:\n{paths_preview}\nFocus: Purpose, key resources, auth. Concise (150 words)."
        try:
            state.schema_summary = llm_call_helper(self.worker_llm, summary_prompt)
            logger.info("Schema summary generated.")
            state.response = "API summary created." 
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}", exc_info=False) 
            state.schema_summary = f"Error generating summary: {str(e)[:150]}..." 
            state.response = state.schema_summary 
        state.update_scratchpad_reason(tool_name, f"Summary status: {'Success' if state.schema_summary and not state.schema_summary.startswith('Error') else 'Failed'}")

    def _identify_apis_from_schema(self, state: BotState):
        tool_name = "_identify_apis_from_schema"
        state.response = "Identifying API operations..." 
        state.update_scratchpad_reason(tool_name, "Identifying APIs.")
        if not state.openapi_schema:
            state.identified_apis = []; logger.warning("No schema to identify APIs from.")
            state.response = "Cannot identify APIs: No schema loaded."
            return
        apis = []
        for path, item in state.openapi_schema.get('paths', {}).items():
            if not isinstance(item, dict): continue
            for method, op in item.items():
                if method.lower() not in {'get', 'post', 'put', 'delete', 'patch', 'options', 'head', 'trace'} or not isinstance(op, dict): continue
                op_id_path = path.replace('/', '_').replace('{', '').replace('}', '').strip('_')
                apis.append({'operationId': op.get('operationId', f"{method.lower()}_{op_id_path or 'root'}"),
                             'path': path, 'method': method.upper(), 'summary': op.get('summary', ''),
                             'parameters': op.get('parameters', []), 'requestBody': op.get('requestBody', {}),
                             'responses': op.get('responses', {})})
        state.identified_apis = apis
        logger.info(f"Identified {len(apis)} API operations.")
        state.response = f"Identified {len(apis)} API operations." 
        state.update_scratchpad_reason(tool_name, f"Identified {len(apis)} APIs.")

    def _generate_payload_descriptions(self, state: BotState, target_apis: Optional[List[str]] = None, context_override: Optional[str] = None):
        tool_name = "_generate_payload_descriptions"
        state.response = "Creating payload and response examples..." 
        state.update_scratchpad_reason(tool_name, f"Generating payload descriptions. Targets: {target_apis or 'subset'}. Context: {bool(context_override)}")
        if not state.identified_apis:
            logger.warning("No APIs for payload descriptions."); 
            state.response = "Cannot create payload examples: No APIs identified."
            return

        payload_descs = state.payload_descriptions or {}
        if target_apis:
            apis_to_process = [api for api in state.identified_apis if api['operationId'] in target_apis]
        else: 
            apis_with_payload_info = [api for api in state.identified_apis if api.get('parameters') or api.get('requestBody')]
            apis_to_process = apis_with_payload_info[:3] 

        logger.info(f"Attempting to generate payload descriptions for {len(apis_to_process)} APIs.")
        processed_count = 0
        for api in apis_to_process:
            op_id = api['operationId']
            if op_id in payload_descs and not context_override and not target_apis: 
                continue
            state.response = f"Generating payload example for '{op_id}'..." 
            param_str = json.dumps(api.get('parameters',[]))[:200]; body_str = json.dumps(api.get('requestBody',{}))[:200]
            resp_str = json.dumps(api.get('responses',{}).get('200',{}).get('content',{}).get('application/json',{}).get('schema',{}))[:200]
            ctx_str = f" Context: {context_override}." if context_override else ""
            prompt = f"API: {op_id} ({api['method']} {api['path']}). Summary: {api['summary']}.{ctx_str} Params: {param_str}. Body: {body_str}. RespSchema: {resp_str}. Describe typical request (key fields/values) & response structure. Brief. Format: Req: ... Resp: ..."
            try:
                payload_descs[op_id] = llm_call_helper(self.worker_llm, prompt)
                processed_count += 1
            except Exception as e:
                logger.error(f"Error for payload desc {op_id}: {e}", exc_info=False)
                payload_descs[op_id] = f"Error generating description: {str(e)[:100]}..." 
                state.response = f"Error creating payload example for '{op_id}': {str(e)[:100]}..."
                if "quota" in str(e).lower() or "429" in str(e):
                    logger.warning(f"Quota error during payload description for {op_id}. Stopping further payload generation for this turn.")
                    state.response += " Hit API limits."
                    break 
        state.payload_descriptions = payload_descs
        if processed_count > 0 :
            state.response = f"Generated payload examples for {processed_count} API(s)."
        elif not apis_to_process:
             state.response = "No relevant APIs found requiring payload examples."
        state.update_scratchpad_reason(tool_name, f"Payload descs updated for {processed_count} of {len(apis_to_process)} targeted APIs.")

    def _generate_execution_graph(self, state: BotState, goal: Optional[str] = None) -> BotState:
        tool_name = "_generate_execution_graph"
        current_goal = goal or state.plan_generation_goal or "General workflow"
        state.response = f"Building initial API workflow graph for goal: '{current_goal[:50]}...'" 
        state.update_scratchpad_reason(tool_name, f"Generating graph. Goal: {current_goal}")
        # logger.info(f"Generating graph. Goal: {current_goal}") 

        if not state.identified_apis:
            state.response = "Cannot generate graph: No APIs identified."
            state.execution_graph = None 
            state.update_scratchpad_reason(tool_name, state.response)
            state.next_step = "responder"
            return state

        apis_str = "\n".join([f"- {a['operationId']}: {a['summary']}" for a in state.identified_apis[:15]])
        fbk_str = f"Feedback: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else ""
        prompt = f"Goal: \"{current_goal}\". {fbk_str} APIs (sample):\n{apis_str}\nDesign a JSON API graph (nodes, edges, input_mappings for data flow e.g. '$.id', overall description). Use 3-5 relevant APIs. CRITICAL: All `operationId` or `display_name` values used in `edges` (for `from_node` and `to_node`) MUST correspond to an `operationId` or `display_name` of a node defined in the `nodes` list of THIS SAME JSON output. Model:\n{{\"nodes\":[{{\"operationId\":\"id\",\"summary\":\"s\",\"description\":\"d\",\"payload_description\":\"p\",\"input_mappings\":[{{\"source_operation_id\":\"sid\",\"source_data_path\":\"spath\",\"target_parameter_name\":\"tpn\",\"target_parameter_in\":\"tin\"}}]}}],\"edges\":[{{\"from_node\":\"f\",\"to_node\":\"t\",\"description\":\"d\"}}],\"description\":\"desc\",\"refinement_summary\":\"Initial graph\"}}\nOutput ONLY JSON."
        
        try:
            llm_response = llm_call_helper(self.worker_llm, prompt)
            graph_output = parse_llm_json_output_with_model(llm_response, expected_model=GraphOutput)

            if graph_output:
                state.execution_graph = graph_output
                state.response = "Initial API workflow graph generated." 
                logger.info(f"Initial graph generated. Description: {graph_output.description if graph_output else 'N/A'}")
                if graph_output.refinement_summary: 
                    logger.info(f"LLM summary for initial graph: {graph_output.refinement_summary}")
                state.graph_regeneration_reason = None
                state.graph_refinement_iterations = 0
                state.next_step = "verify_graph"
            else:
                error_msg = "LLM failed to produce a valid and self-consistent GraphOutput JSON structure for the initial graph."
                logger.error(error_msg + f" Raw LLM output snippet: {llm_response[:300]}...")
                state.response = "Failed to generate a valid execution graph (AI output format or structure error)."
                state.execution_graph = None # Ensure graph is None
                state.graph_regeneration_reason = "LLM output was not a valid GraphOutput object."
                if state.scratchpad.get('graph_gen_attempts', 0) < 1:
                    state.scratchpad['graph_gen_attempts'] = state.scratchpad.get('graph_gen_attempts', 0) + 1
                    logger.info("Retrying initial graph generation once.")
                    state.next_step = "_generate_execution_graph" 
                else:
                    logger.error("Max initial graph generation attempts reached. Routing to handle_unknown.")
                    state.next_step = "handle_unknown" 
                    state.scratchpad['graph_gen_attempts'] = 0 
        except Exception as e: 
            logger.error(f"Error during initial graph generation LLM call or processing: {e}", exc_info=False)
            state.response = f"Error generating graph: {str(e)[:150]}..."
            state.execution_graph = None 
            state.graph_regeneration_reason = f"LLM call/processing error: {str(e)[:100]}..."
            state.next_step = "handle_unknown" 
        state.update_scratchpad_reason(tool_name, f"Initial graph gen status: {'Success' if state.execution_graph else 'Failed'}. Resp: {state.response}")
        return state

    def process_schema_pipeline(self, state: BotState) -> BotState:
        tool_name = "process_schema_pipeline"
        state.response = "Starting API analysis pipeline..." 
        state.update_scratchpad_reason(tool_name, "Starting schema pipeline.")
        # logger.info("Executing process_schema_pipeline.") 

        if not state.openapi_schema:
            state.response = "Cannot run pipeline: No schema."
            state.next_step = "handle_unknown"; return state
        
        state.schema_summary = None; state.identified_apis = []; state.payload_descriptions = {}
        state.execution_graph = None; state.graph_refinement_iterations = 0
        state.plan_generation_goal = state.plan_generation_goal or "Provide a general overview workflow."
        state.scratchpad['graph_gen_attempts'] = 0
        state.scratchpad['refinement_validation_failures'] = 0


        self._generate_llm_schema_summary(state) 
        if state.schema_summary and "Error generating summary: 429" in state.schema_summary : 
            state.next_step = "responder"; return state

        self._identify_apis_from_schema(state) 
        if not state.identified_apis:
            state.response = (state.response or "") + " No API operations were identified. Cannot generate payloads or graph."
            state.next_step = "responder"; return state

        self._generate_payload_descriptions(state) 
        if any("Error generating description: 429" in desc for desc in state.payload_descriptions.values()):
             state.response = (state.response or "") + " Partial success: Hit API limits while generating some payload examples."
        
        self._generate_execution_graph(state, goal=state.plan_generation_goal) 
        state.update_scratchpad_reason(tool_name, f"Pipeline initiated. Next: {state.next_step}")
        return state

    def verify_graph(self, state: BotState) -> BotState:
        tool_name = "verify_graph"
        state.response = "Verifying API workflow graph..." 
        state.update_scratchpad_reason(tool_name, "Verifying graph.")
        # logger.info("Executing verify_graph node.") 

        if not state.execution_graph: 
            state.response = state.response or "No execution graph to verify (possibly due to generation error)."
            state.graph_regeneration_reason = state.graph_regeneration_reason or "No graph was generated to verify."
            logger.warning(f"verify_graph: No graph found. Reason: {state.graph_regeneration_reason}. Routing to _generate_execution_graph.")
            state.next_step = "_generate_execution_graph"; return state

        is_dag, cycle_msg = check_for_cycles(state.execution_graph)
        
        if is_dag:
            state.response = "Graph verification successful (DAG and basic structure)." 
            state.update_scratchpad_reason(tool_name, "Graph verification successful.")
            logger.info("Graph verification successful.")
            state.graph_regeneration_reason = None 
            state.scratchpad['refinement_validation_failures'] = 0 
            
            try:
                state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
                logger.info("Graph marked to be sent to UI after verification.")
            except Exception as e:
                logger.error(f"Error serializing graph for sending after verification: {e}")

            if state.graph_refinement_iterations < state.max_refinement_iterations:
                logger.info(f"Proceeding to refinement iteration {state.graph_refinement_iterations + 1}.")
                state.next_step = "refine_api_graph"
            else:
                logger.info("Max refinement iterations reached or refinement complete.")
                if state.input_is_spec: 
                    api_title = state.openapi_schema.get('info', {}).get('title', 'API') # type: ignore
                    state.response = (
                        f"Successfully processed '{api_title}'. Identified {len(state.identified_apis)} APIs, "
                        f"generated payload examples, and created an API workflow graph with {len(state.execution_graph.nodes)} steps. "
                        "You can now ask questions or request specific plan refinements."
                    )
                    state.input_is_spec = False 
                else: state.response = "Graph is verified. " + (state.execution_graph.refinement_summary or "No specific refinement summary.")
                state.next_step = "describe_graph" 
        else: 
            state.response = f"Graph verification failed: {cycle_msg}. " 
            state.graph_regeneration_reason = f"Verification failed (not a DAG): {cycle_msg}."
            logger.warning(f"Graph verification failed (cycle detected): {cycle_msg}.")
            if state.graph_refinement_iterations < state.max_refinement_iterations:
                state.next_step = "refine_api_graph"
            else: 
                logger.warning("Max refinements hit, but graph still has cycles. Attempting full regeneration.")
                state.next_step = "_generate_execution_graph"
        state.update_scratchpad_reason(tool_name, f"Verification result: {state.response}")
        return state

    def refine_api_graph(self, state: BotState) -> BotState:
        tool_name = "refine_api_graph"
        iteration = state.graph_refinement_iterations + 1
        state.response = f"Refining API workflow graph (Attempt {iteration}/{state.max_refinement_iterations})..." 
        state.update_scratchpad_reason(tool_name, f"Refining graph. Iteration: {iteration}")
        # logger.info(f"Executing refine_api_graph. Iteration: {iteration}")

        if not state.execution_graph: 
            state.response = "No graph to refine. Please generate a graph first."
            state.next_step = "_generate_execution_graph"; return state
        if iteration > state.max_refinement_iterations:
            state.response = f"Max refinement iterations ({state.max_refinement_iterations}) reached. Using current graph: {state.execution_graph.description or 'N/A'}"
            state.next_step = "describe_graph"; return state

        graph_json = state.execution_graph.model_dump_json(indent=2)
        apis_ctx = "\n".join([f"- {a['operationId']}: {a['summary']}" for a in state.identified_apis[:10]])
        prompt = f"Goal: \"{state.plan_generation_goal or 'General workflow'}\". Feedback: {state.graph_regeneration_reason or 'N/A. This might be the first refinement attempt or previous attempts were structurally okay.'}. Current Graph (JSON):\n```json\n{graph_json}\n```\nAPIs (sample):\n{apis_ctx}\nRefine this graph. Focus: Goal alignment, logical flow, data flow (input_mappings like '$.id'), clarity. CRITICAL: All `operationId` or `display_name` values used in `edges` (for `from_node` and `to_node`) MUST correspond to an `operationId` or `display_name` of a node defined in the `nodes` list of THIS SAME JSON output. Output ONLY refined GraphOutput JSON with a 'refinement_summary' field. Model:\n{{\"nodes\":[{{\"operationId\":\"id\",\"summary\":\"s\",\"description\":\"d\",\"payload_description\":\"p\",\"input_mappings\":[{{\"source_operation_id\":\"sid\",\"source_data_path\":\"spath\",\"target_parameter_name\":\"tpn\",\"target_parameter_in\":\"tin\"}}]}}],\"edges\":[{{\"from_node\":\"f\",\"to_node\":\"t\",\"description\":\"d\"}}],\"description\":\"desc\",\"refinement_summary\":\"summary\"}}"
        
        try:
            llm_response_str = llm_call_helper(self.worker_llm, prompt)
            refined_graph_candidate = parse_llm_json_output_with_model(llm_response_str, expected_model=GraphOutput)

            if refined_graph_candidate:
                logger.info(f"Refinement attempt (iter {iteration}) produced a structurally valid GraphOutput.")
                state.execution_graph = refined_graph_candidate 
                refinement_summary = refined_graph_candidate.refinement_summary or "AI provided no specific summary for this refinement."
                state.update_scratchpad_reason(tool_name, f"LLM Refinement Summary: {refinement_summary}")
                
                state.graph_refinement_iterations = iteration
                state.response = f"Graph refined (Iteration {iteration}). Summary: {refinement_summary}" 
                state.graph_regeneration_reason = None 
                state.scratchpad['refinement_validation_failures'] = 0 
                state.next_step = "verify_graph" 
            else:
                error_msg = "LLM refinement failed to produce a valid and self-consistent GraphOutput JSON structure."
                logger.error(error_msg + f" Raw LLM output snippet for refinement: {llm_response_str[:300]}...")
                state.response = f"Error during graph refinement (iteration {iteration}): AI output was invalid. Will retry or regenerate."
                state.graph_regeneration_reason = "LLM output for refinement was not a valid GraphOutput object, or had structural issues (e.g., invalid edge references)."
                
                state.scratchpad['refinement_validation_failures'] = state.scratchpad.get('refinement_validation_failures', 0) + 1
                
                if iteration < state.max_refinement_iterations:
                    if state.scratchpad['refinement_validation_failures'] >= 2: 
                        logger.warning(f"Multiple consecutive refinement validation failures (iter {iteration}). Escalating to full graph regeneration.")
                        state.response += " Attempting full regeneration due to persistent refinement issues."
                        state.next_step = "_generate_execution_graph" 
                        state.graph_refinement_iterations = 0 
                        state.scratchpad['refinement_validation_failures'] = 0
                    else:
                        state.next_step = "refine_api_graph" 
                else:
                    logger.warning(f"Max refinement iterations reached after LLM output error. Describing last valid graph.")
                    state.next_step = "describe_graph" 
        
        except Exception as e: 
            logger.error(f"Error during graph refinement LLM call or processing (iter {iteration}): {e}", exc_info=False)
            state.response = f"Error refining graph (iter {iteration}): {str(e)[:150]}..."
            state.graph_regeneration_reason = f"Refinement LLM call/processing error (iter {iteration}): {str(e)[:100]}..."
            state.next_step = "describe_graph"
        return state

    def describe_graph(self, state: BotState) -> BotState:
        tool_name = "describe_graph"
        state.response = "Preparing graph description..." 
        # logger.info("Executing describe_graph node.") 
        if not state.execution_graph:
            state.response = state.response or "No execution graph available to describe."
            logger.warning("describe_graph: No execution_graph found in state.")
        else:
            desc = state.execution_graph.description
            if not desc or len(desc) < 20:
                nodes_str = "\n".join([f"- {n.effective_id}: {n.summary or n.operationId}" for n in state.execution_graph.nodes[:3]])
                prompt = f"Describe API graph for goal '{state.plan_generation_goal or 'general use'}'. Nodes ({len(state.execution_graph.nodes)} total, sample):\n{nodes_str}\nExplain purpose & flow concisely. Use Markdown for readability (headings, lists)."
                try: desc = llm_call_helper(self.worker_llm, prompt)
                except Exception as e: desc = f"Error generating dynamic desc: {str(e)[:100]}... Stored: {state.execution_graph.description or 'N/A'}"
            
            # Ensure final description is user-friendly
            final_desc = f"### Current API Workflow for: '{state.plan_generation_goal or 'General Use'}'\n\n{desc}"
            if state.execution_graph.refinement_summary: 
                final_desc += f"\n\n**Last Refinement Note:** {state.execution_graph.refinement_summary}"
            state.response = final_desc

        state.update_scratchpad_reason(tool_name, f"Described graph. Response set: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        tool_name = "get_graph_json"
        state.response = "Fetching graph JSON..." 
        if not state.execution_graph: state.response = "No execution graph."
        else:
            try: 
                graph_json_str = state.execution_graph.model_dump_json(indent=2)
                state.response = f"Graph JSON:\n```json\n{graph_json_str}\n```" 
                state.scratchpad['graph_to_send'] = graph_json_str
            except Exception as e: state.response = f"Error serializing graph: {e}"
        state.update_scratchpad_reason(tool_name, "Provided graph JSON.")
        state.next_step = "responder"
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        tool_name = "answer_openapi_query"
        state.response = "Thinking about your question..." 
        # logger.info("Executing answer_openapi_query node.") 
        if not state.openapi_schema:
            state.response = "No OpenAPI spec loaded."; state.next_step = "responder"; return state
        
        # Prepare a detailed context string
        api_list_md = ""
        if state.identified_apis:
            api_list_md = "\nIdentified API Operations:\n"
            for api in state.identified_apis[:10]: # Show up to 10
                api_list_md += f"- **{api['operationId']}**: {api['method']} {api['path']} - _{api['summary']}_\n"
            if len(state.identified_apis) > 10:
                api_list_md += "- ... and more.\n"
        
        graph_desc_md = ""
        if state.execution_graph and state.execution_graph.description:
            graph_desc_md = f"\nCurrent Workflow Graph ('{state.plan_generation_goal or 'General'}'):\n{state.execution_graph.description}\n"
            if state.execution_graph.refinement_summary:
                graph_desc_md += f"Last Refinement: {state.execution_graph.refinement_summary}\n"

        payload_info_md = ""
        if state.user_input:
            for op_id, desc_text in state.payload_descriptions.items():
                if op_id.lower() in state.user_input.lower(): # Simple check if op_id is mentioned
                    payload_info_md = f"\nPayload/Response Example for '{op_id}':\n{desc_text}\n"
                    break
        
        prompt = f"""
        You are an expert API assistant. Answer the user's question based on the provided context.
        Use Markdown for formatting (headings, lists, bolding, italics, code blocks for JSON).

        User Question: "{state.user_input}"

        Context:
        ### API Specification Summary
        {state.schema_summary or 'Not available.'}
        {api_list_md or 'No specific API operations listed in this context.'}
        {graph_desc_md or 'No execution graph currently described.'}
        {payload_info_md or 'No specific payload example requested in this query.'}

        Please provide a clear, well-formatted, and helpful answer. If the information is not available in the context, state that.
        """
        try: 
            state.response = llm_call_helper(self.worker_llm, prompt) 
        except Exception as e: 
            state.response = f"### Error Answering Query\nSorry, I encountered an error: {str(e)[:100]}..."
        state.update_scratchpad_reason(tool_name, f"Answered query. Response: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        tool_name = "interactive_query_planner"
        state.response = "Planning how to address your query..." 
        # logger.info(f"Executing {tool_name} for query: {state.user_input}") 
        state.scratchpad.pop('interactive_action_plan', None); state.scratchpad.pop('current_interactive_action_idx', None)
        state.scratchpad.pop('current_interactive_results', None)

        graph_sum = state.execution_graph.description if state.execution_graph else "No graph."
        pl_keys = list(state.payload_descriptions.keys())[:3]
        prompt = f"""
        User Query: "{state.user_input}"
        Current State Context:
        - API Spec Summary: {state.schema_summary[:100] if state.schema_summary else 'N/A'}...
        - Identified APIs (first 5 opIds): {", ".join([api['operationId'] for api in state.identified_apis[:5]])}...
        - Example Payload Descriptions available for (first 5 opIds): {pl_keys}...
        - Current Execution Graph Goal: {state.plan_generation_goal or 'N/A'}
        - Current Graph Description: {graph_sum[:100]}...

        Available Internal Actions (choose one or more in sequence):
        1.  `rerun_payload_generation`: Regenerate payload/response examples for specific APIs with new context.
            Params: {{ "operation_ids_to_update": ["opId1", "opId2"], "new_context": "User's new context string" }}
        2.  `contextualize_graph_descriptions`: Rewrite descriptions within the *existing* graph (overall, nodes, edges) to reflect new user context.
            Params: {{ "target_node_operation_ids": ["opId1"], "new_context": "User's new context string" }}
        3.  `regenerate_graph_with_new_goal`: Create a *new* graph if the user states a completely different high-level goal OR requests a significant structural change (add/remove/reorder steps).
            Params: {{ "new_goal_string": "User's new goal, incorporating the structural change (e.g., 'Workflow to X, Y, and then Z as the last step')" }}
        4.  `refine_existing_graph_further`: For minor adjustments to descriptions, data mappings, or slight non-structural improvements based on user feedback.
            Params: {{ "refinement_instructions": "User's specific feedback for refinement (e.g., 'Clarify the payload for node X')" }}
        5.  `answer_query_directly`: If the query can be answered using existing information without modifications.
            Params: {{ "query_for_synthesizer": "The original user query or a rephrased one for direct answering." }}
        6.  `synthesize_final_answer`: (Usually the last step) Formulate a comprehensive answer to the user based on the outcomes of previous internal actions.
            Params: {{ "synthesis_prompt_instructions": "Instructions for the LLM on what to include in the final answer." }}

        Task:
        1. Understand the user's query.
        2. Create a short, logical "interactive_action_plan" (a list of action objects) to address it.
           For structural changes like "add X at the end", prefer `regenerate_graph_with_new_goal` by formulating a new goal that includes this change.
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
            data = parse_llm_json_output_with_model(llm_call_helper(self.worker_llm, prompt))
            if data and "interactive_action_plan" in data and isinstance(data["interactive_action_plan"], list):
                state.scratchpad.update(data); state.scratchpad['current_interactive_action_idx'] = 0; state.scratchpad['current_interactive_results'] = []
                state.response = f"Understood query: {data.get('user_query_understanding', 'N/A')}. Starting internal actions..." 
                state.next_step = "interactive_query_executor"
            else: raise ValueError("LLM failed to produce valid interactive plan.")
        except Exception as e:
            logger.error(f"Error in {tool_name}: {e}", exc_info=False)
            state.response = f"Sorry, error planning for your request: {str(e)[:100]}..."
            state.next_step = "responder"
        state.update_scratchpad_reason(tool_name, f"Interactive plan generated. Next: {state.next_step}")
        return state

    def _internal_rerun_payload_generation(self, state: BotState, op_ids: List[str], ctx: str) -> str:
        self._generate_payload_descriptions(state, target_apis=op_ids, context_override=ctx) 
        return f"Payloads updated for {op_ids} with context '{ctx[:30]}...'."
    def _internal_contextualize_graph(self, state: BotState, targets: Optional[List[str]], ctx: str) -> str:
        if not state.execution_graph: return "No graph to contextualize."
        state.response = f"Contextualizing graph for '{ctx[:30]}...'" 
        prompt = f"New context: \"{ctx}\". Current graph desc: \"{state.execution_graph.description}\". Rewrite desc. New Desc:"
        try: state.execution_graph.description = llm_call_helper(self.worker_llm, prompt)
        except: pass 
        if state.execution_graph:
            try: state.scratchpad['graph_to_send'] = state.execution_graph.model_dump_json(indent=2)
            except Exception as e: logger.error(f"Error serializing graph for sending after contextualization: {e}")
        return f"Graph descriptions contextualized for '{ctx[:30]}...'."
    def _internal_answer_query_directly(self, state: BotState, query: str) -> str:
        state.response = f"Answering query: '{query[:50]}...'" 
        orig_input, state.user_input = state.user_input, query
        self.answer_openapi_query(state); direct_ans = state.response 
        state.user_input, state.response = orig_input, direct_ans 
        return direct_ans or "Could not answer directly."
    def _internal_synthesize_final_answer(self, state: BotState, instr: str) -> str:
        state.response = "Synthesizing final answer..." 
        results_sum = "\n".join([str(r)[:100]+"..." for r in state.scratchpad.get('current_interactive_results',[])])
        prompt = f"User Query Understanding: {state.scratchpad.get('user_query_understanding','N/A')}. Synthesis Instr: {instr}. Internal Results:\n{results_sum}\nGraph Desc: {state.execution_graph.description if state.execution_graph else 'N/A'}. Formulate final user answer using Markdown for readability."
        try: state.response = llm_call_helper(self.worker_llm, prompt) 
        except Exception as e: state.response = f"Error synthesizing: {str(e)[:100]}..."
        return state.response or "Synthesis error."

    def interactive_query_executor(self, state: BotState) -> BotState:
        tool_name = "interactive_query_executor"
        plan = state.scratchpad.get('interactive_action_plan', [])
        idx = state.scratchpad.get('current_interactive_action_idx', 0)
        results = state.scratchpad.get('current_interactive_results', [])

        if not plan or idx >= len(plan):
            state.response = state.response or "Finished interactive processing."; state.next_step = "responder"; return state

        action = plan[idx]; name = action.get("action_name"); params = action.get("action_params", {}); desc = action.get("description","")
        state.response = f"Executing internal step ({idx+1}/{len(plan)}): {desc[:50]}..." 
        state.update_scratchpad_reason(tool_name, f"Executing interactive action ({idx+1}/{len(plan)}): {name} - {desc}")
        # logger.info(f"Executing interactive action: {name} with params: {params}") 

        action_result_message = f"Action '{name}' executed." 
        
        try:
            if name == "rerun_payload_generation": action_result_message = self._internal_rerun_payload_generation(state, params.get("operation_ids_to_update",[]), params.get("new_context",""))
            elif name == "contextualize_graph_descriptions": action_result_message = self._internal_contextualize_graph(state, params.get("target_node_operation_ids"), params.get("new_context",""))
            elif name == "regenerate_graph_with_new_goal" and params.get("new_goal_string"):
                state.plan_generation_goal = params["new_goal_string"]; state.graph_refinement_iterations = 0; state.execution_graph = None
                state.scratchpad['graph_gen_attempts'] = 0
                state.scratchpad['refinement_validation_failures'] = 0
                self._generate_execution_graph(state, goal=state.plan_generation_goal) 
                results.append(f"Started new graph for: {state.plan_generation_goal}"); state.scratchpad['current_interactive_results'] = results
                return state 
            elif name == "refine_existing_graph_further" and state.execution_graph:
                state.graph_regeneration_reason = params.get("refinement_instructions","")
                state.scratchpad['refinement_validation_failures'] = 0
                self.refine_api_graph(state) 
                results.append(f"Started graph refinement."); state.scratchpad['current_interactive_results'] = results
                return state 
            elif name == "answer_query_directly": state.response = self._internal_answer_query_directly(state, params.get("query_for_synthesizer", state.user_input or ""))
            elif name == "synthesize_final_answer": state.response = self._internal_synthesize_final_answer(state, params.get("synthesis_prompt_instructions","Summarize."))
            else: action_result_message = f"Unknown action: {name}."
        except Exception as e: 
            logger.error(f"Error in action {name}: {e}", exc_info=False); 
            action_result_message = f"Error in '{name}': {str(e)[:100]}..."
            state.response = action_result_message 
        
        results.append(action_result_message); state.scratchpad['current_interactive_results'] = results
        state.scratchpad['current_interactive_action_idx'] = idx + 1
        
        if state.scratchpad['current_interactive_action_idx'] < len(plan): state.next_step = "interactive_query_executor"
        else: logger.info("All interactive actions done."); state.next_step = "responder"
        return state

    def handle_unknown(self, state: BotState) -> BotState:
        tool_name = "handle_unknown"
        # logger.warning(f"Executing {tool_name}. Current response (if any): {state.response}") 
        if not state.response or "error" not in str(state.response).lower(): 
            state.response = "I'm not sure how to process that. Could you rephrase or provide an OpenAPI spec first?"
        state.update_scratchpad_reason(tool_name, f"Handling unknown. Final response to be: {state.response}")
        state.next_step = "responder"
        return state

    def handle_loop(self, state: BotState) -> BotState:
        state.response = "It seems we're in a loop. Please rephrase."
        state.loop_counter = 0; state.next_step = "responder"
        state.update_scratchpad_reason("handle_loop", "Loop detected, routing to responder.")
        return state

