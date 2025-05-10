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
            state.response = "OpenAPI specification parsed. Analyzing..."
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
        state.update_scratchpad_reason(tool_name, "Generating schema summary.")
        if not state.openapi_schema:
            state.schema_summary = "Could not generate summary: No schema loaded."
            logger.warning(state.schema_summary)
            return

        spec, info = state.openapi_schema, state.openapi_schema.get('info', {})
        paths_preview = "\n".join([f"  {p}: {list(m.keys()) if isinstance(m, dict) else '[?]'}" for p, m in list(spec.get('paths', {}).items())[:3]])
        summary_prompt = f"Summarize API: {info.get('title', 'N/A')} (v{info.get('version', 'N/A')}). Desc: {info.get('description', 'N/A')[:200]}... Paths: {len(spec.get('paths', {}))}. Preview:\n{paths_preview}\nFocus: Purpose, key resources, auth. Concise (150 words)."
        try:
            state.schema_summary = llm_call_helper(self.worker_llm, summary_prompt)
            logger.info("Schema summary generated.")
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}", exc_info=False) # Reduce log noise for quota errors
            state.schema_summary = f"Error generating summary: {str(e)[:150]}..." # Truncate long error messages
        state.update_scratchpad_reason(tool_name, f"Summary status: {'Success' if state.schema_summary and not state.schema_summary.startswith('Error') else 'Failed'}")

    def _identify_apis_from_schema(self, state: BotState):
        tool_name = "_identify_apis_from_schema"
        state.update_scratchpad_reason(tool_name, "Identifying APIs.")
        if not state.openapi_schema:
            state.identified_apis = []; logger.warning("No schema to identify APIs from.")
            return
        apis = []
        for path, item in state.openapi_schema.get('paths', {}).items():
            if not isinstance(item, dict): continue
            for method, op in item.items():
                if method.lower() not in {'get', 'post', 'put', 'delete', 'patch'} or not isinstance(op, dict): continue
                op_id_path = path.replace('/', '_').replace('{', '').replace('}', '').strip('_')
                apis.append({'operationId': op.get('operationId', f"{method.lower()}_{op_id_path or 'root'}"),
                             'path': path, 'method': method.upper(), 'summary': op.get('summary', ''),
                             'parameters': op.get('parameters', []), 'requestBody': op.get('requestBody', {}),
                             'responses': op.get('responses', {})})
        state.identified_apis = apis
        logger.info(f"Identified {len(apis)} API operations.")
        state.update_scratchpad_reason(tool_name, f"Identified {len(apis)} APIs.")

    def _generate_payload_descriptions(self, state: BotState, target_apis: Optional[List[str]] = None, context_override: Optional[str] = None):
        tool_name = "_generate_payload_descriptions"
        state.update_scratchpad_reason(tool_name, f"Generating payload descriptions. Targets: {target_apis or 'subset'}. Context: {bool(context_override)}")
        if not state.identified_apis:
            logger.warning("No APIs for payload descriptions."); return

        payload_descs = state.payload_descriptions or {}
        # Reduce LLM calls: Process only first 5 APIs with bodies/params, or specified targets
        apis_to_consider = [api for api in state.identified_apis if api.get('parameters') or api.get('requestBody')]
        apis_to_process = [api for api in apis_to_consider if (target_apis is None or api['operationId'] in target_apis)][:5] # Limit to 5 unless specific targets
        if target_apis: # If specific targets, use them irrespective of the limit of 5
            apis_to_process = [api for api in state.identified_apis if api['operationId'] in target_apis]


        logger.info(f"Attempting to generate payload descriptions for {len(apis_to_process)} APIs.")
        for api in apis_to_process:
            op_id = api['operationId']
            if op_id in payload_descs and not context_override and not target_apis: continue # Skip if already done unless forced

            param_str = json.dumps(api.get('parameters',[]))[:200]; body_str = json.dumps(api.get('requestBody',{}))[:200]
            resp_str = json.dumps(api.get('responses',{}).get('200',{}).get('content',{}).get('application/json',{}).get('schema',{}))[:200]
            ctx_str = f" Context: {context_override}." if context_override else ""
            prompt = f"API: {op_id} ({api['method']} {api['path']}). Summary: {api['summary']}.{ctx_str} Params: {param_str}. Body: {body_str}. RespSchema: {resp_str}. Describe typical request (key fields/values) & response structure. Brief. Format: Req: ... Resp: ..."
            try:
                payload_descs[op_id] = llm_call_helper(self.worker_llm, prompt)
            except Exception as e:
                logger.error(f"Error for payload desc {op_id}: {e}", exc_info=False)
                payload_descs[op_id] = f"Error: {str(e)[:100]}..."
        state.payload_descriptions = payload_descs
        state.update_scratchpad_reason(tool_name, f"Payload descs updated for {len(apis_to_process)} APIs.")

    def _generate_execution_graph(self, state: BotState, goal: Optional[str] = None) -> BotState:
        tool_name = "_generate_execution_graph"
        current_goal = goal or state.plan_generation_goal or "General workflow"
        state.update_scratchpad_reason(tool_name, f"Generating graph. Goal: {current_goal}")
        logger.info(f"Generating graph. Goal: {current_goal}")

        if not state.identified_apis:
            state.response = "Cannot generate graph: No APIs identified."
            state.execution_graph = None # Ensure it's None
            state.update_scratchpad_reason(tool_name, state.response)
            state.next_step = "responder"
            return state

        apis_str = "\n".join([f"- {a['operationId']}: {a['summary']}" for a in state.identified_apis[:15]])
        fbk_str = f"Feedback: {state.graph_regeneration_reason}" if state.graph_regeneration_reason else ""
        prompt = f"Goal: \"{current_goal}\". {fbk_str} APIs (sample):\n{apis_str}\nDesign a JSON API graph (nodes, edges, input_mappings for data flow e.g. '$.id', overall description). Use 3-5 relevant APIs. Ensure edges use nodes defined in YOUR output. Model:\n{{\"nodes\":[{{\"operationId\":\"id\",\"summary\":\"s\",\"description\":\"d\",\"payload_description\":\"p\",\"input_mappings\":[{{\"source_operation_id\":\"sid\",\"source_data_path\":\"spath\",\"target_parameter_name\":\"tpn\",\"target_parameter_in\":\"tin\"}}]}}],\"edges\":[{{\"from_node\":\"f\",\"to_node\":\"t\",\"description\":\"d\"}}],\"description\":\"desc\"}}\nOutput ONLY JSON."
        try:
            graph_output = parse_llm_json_output_with_model(llm_call_helper(self.worker_llm, prompt), GraphOutput)
            if graph_output:
                state.execution_graph = graph_output
                state.response = "API execution graph generated."
                state.graph_regeneration_reason = None; state.graph_refinement_iterations = 0
                state.next_step = "verify_graph"
            else:
                raise ValueError("LLM failed to produce valid GraphOutput JSON.")
        except Exception as e:
            logger.error(f"Error generating graph: {e}", exc_info=False)
            state.response = f"Error generating graph: {str(e)[:150]}..."
            state.execution_graph = None # Ensure graph is None on error
            state.graph_regeneration_reason = f"Error: {str(e)[:100]}..."
            # More robust: if graph_gen_attempts > X, then handle_unknown
            state.next_step = "handle_unknown" # Route to handle_unknown on generation error
        state.update_scratchpad_reason(tool_name, f"Graph gen status: {'Success' if state.execution_graph else 'Failed'}. Resp: {state.response}")
        return state

    def process_schema_pipeline(self, state: BotState) -> BotState:
        tool_name = "process_schema_pipeline"
        state.update_scratchpad_reason(tool_name, "Starting schema pipeline.")
        logger.info("Executing process_schema_pipeline.")

        if not state.openapi_schema:
            state.response = "Cannot run pipeline: No schema."
            state.next_step = "handle_unknown"; return state
        
        state.schema_summary = None; state.identified_apis = []; state.payload_descriptions = {}
        state.execution_graph = None; state.graph_refinement_iterations = 0
        state.plan_generation_goal = state.plan_generation_goal or "General overview workflow."
        state.scratchpad['graph_gen_attempts'] = 0

        self._generate_llm_schema_summary(state)
        if state.schema_summary and state.schema_summary.startswith("Error"): # Stop if summary failed due to quota
            state.response = state.schema_summary
            state.next_step = "responder"; return state

        self._identify_apis_from_schema(state)
        if not state.identified_apis:
            state.response = (state.response or "") + " No API operations identified. Cannot generate payloads or graph."
            state.next_step = "responder"; return state

        self._generate_payload_descriptions(state) # Processes a subset
        # Check if payload generation hit a quota error for all its attempts
        if all(desc.startswith("Error:") for desc in state.payload_descriptions.values() if state.payload_descriptions):
             state.response = "Failed to generate payload descriptions, likely due to API limits."
             state.next_step = "responder"; return state
        
        self._generate_execution_graph(state, goal=state.plan_generation_goal)
        # _generate_execution_graph sets next_step to "verify_graph" or "handle_unknown"/"responder"
        state.update_scratchpad_reason(tool_name, f"Pipeline initiated. Next: {state.next_step}")
        return state

    def verify_graph(self, state: BotState) -> BotState:
        tool_name = "verify_graph"
        state.update_scratchpad_reason(tool_name, "Verifying graph.")
        logger.info("Executing verify_graph node.")

        if not state.execution_graph: # Should have been caught by _generate_execution_graph
            state.response = state.response or "No graph to verify."
            state.graph_regeneration_reason = "No graph available for verification."
            state.next_step = "_generate_execution_graph"; return state

        is_dag, cycle_msg = check_for_cycles(state.execution_graph)
        if is_dag:
            state.response = "Graph verification successful."
            state.graph_regeneration_reason = None
            if state.graph_refinement_iterations < state.max_refinement_iterations:
                state.next_step = "refine_api_graph"
            else:
                if state.input_is_spec:
                    api_title = state.openapi_schema.get('info', {}).get('title', 'API') # type: ignore
                    state.response = f"Processed '{api_title}'. Identified {len(state.identified_apis)} APIs, generated examples, and a workflow graph. Ask questions or request refinements."
                    state.input_is_spec = False
                else: state.response = "Graph verified. " + (state.execution_graph.refinement_summary or "")
                state.next_step = "describe_graph" # Changed from responder to show the graph
        else:
            state.response = f"Graph verification failed: {cycle_msg}."
            state.graph_regeneration_reason = f"Verification failed: {cycle_msg}."
            if state.graph_refinement_iterations < state.max_refinement_iterations:
                state.next_step = "refine_api_graph"
            else: state.next_step = "_generate_execution_graph"
        state.update_scratchpad_reason(tool_name, f"Verification result: {state.response}")
        return state

    def refine_api_graph(self, state: BotState) -> BotState:
        tool_name = "refine_api_graph"
        iteration = state.graph_refinement_iterations + 1
        state.update_scratchpad_reason(tool_name, f"Refining graph. Iteration: {iteration}")
        logger.info(f"Executing refine_api_graph. Iteration: {iteration}")

        if not state.execution_graph:
            state.response = "No graph to refine."
            state.next_step = "_generate_execution_graph"; return state
        if iteration > state.max_refinement_iterations:
            state.response = f"Max refinements reached. Using current graph: {state.execution_graph.description or 'N/A'}"
            state.next_step = "describe_graph"; return state

        graph_json = state.execution_graph.model_dump_json(indent=2)
        apis_ctx = "\n".join([f"- {a['operationId']}: {a['summary']}" for a in state.identified_apis[:10]])
        prompt = f"Goal: \"{state.plan_generation_goal or 'General workflow'}\". Feedback: {state.graph_regeneration_reason or 'N/A'}. Current Graph (JSON):\n```json\n{graph_json}\n```\nAPIs (sample):\n{apis_ctx}\nRefine this graph. Focus: Goal alignment, logical flow, data flow (input_mappings like '$.id'), clarity. CRITICAL: All edge nodes MUST be in YOUR output's nodes list. Output ONLY refined GraphOutput JSON with a 'refinement_summary' field. Model:\n{{\"nodes\":[{{\"operationId\":\"id\",\"summary\":\"s\",\"description\":\"d\",\"payload_description\":\"p\",\"input_mappings\":[{{\"source_operation_id\":\"sid\",\"source_data_path\":\"spath\",\"target_parameter_name\":\"tpn\",\"target_parameter_in\":\"tin\"}}]}}],\"edges\":[{{\"from_node\":\"f\",\"to_node\":\"t\",\"description\":\"d\"}}],\"description\":\"desc\",\"refinement_summary\":\"summary\"}}"
        try:
            raw_dict = parse_llm_json_output_with_model(llm_call_helper(self.worker_llm, prompt))
            if not raw_dict or not isinstance(raw_dict, dict): raise ValueError("Refinement LLM output not valid dict.")
            summary = raw_dict.pop("refinement_summary", "AI provided no summary.")
            state.update_scratchpad_reason(tool_name, f"LLM Refinement Summary: {summary}")
            
            refined_graph = GraphOutput.model_validate(raw_dict)
            state.execution_graph = refined_graph
            state.graph_refinement_iterations = iteration
            state.response = f"Graph refined (Iter {iteration}). Summary: {summary}"
            state.graph_regeneration_reason = None
            state.next_step = "verify_graph"
        except Exception as e:
            logger.error(f"Error refining graph (iter {iteration}): {e}", exc_info=False)
            state.response = f"Error refining graph (iter {iteration}): {str(e)[:150]}..."
            state.graph_regeneration_reason = f"Refinement error (iter {iteration}): {str(e)[:100]}..."
            state.next_step = "describe_graph" # Fallback to describe current potentially unrefined graph
        return state

    def describe_graph(self, state: BotState) -> BotState:
        tool_name = "describe_graph"
        logger.info("Executing describe_graph node.")
        if not state.execution_graph:
            state.response = state.response or "No execution graph available to describe."
        else:
            desc = state.execution_graph.description
            if not desc or len(desc) < 20:
                nodes_str = "\n".join([f"- {n.effective_id}: {n.summary or n.operationId}" for n in state.execution_graph.nodes[:3]])
                prompt = f"Describe API graph for goal '{state.plan_generation_goal or 'general use'}'. Nodes ({len(state.execution_graph.nodes)} total, sample):\n{nodes_str}\nExplain purpose & flow concisely."
                try: desc = llm_call_helper(self.worker_llm, prompt)
                except Exception as e: desc = f"Error generating dynamic desc: {e}. Stored: {state.execution_graph.description or 'N/A'}"
            state.response = f"Current API Workflow for '{state.plan_generation_goal or 'general use'}':\n{desc}"
            if state.execution_graph.refinement_summary: state.response += f"\nLast Refinement: {state.execution_graph.refinement_summary}"
        state.update_scratchpad_reason(tool_name, f"Described graph. Response set: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def get_graph_json(self, state: BotState) -> BotState:
        tool_name = "get_graph_json"
        if not state.execution_graph: state.response = "No execution graph."
        else:
            try: state.response = f"Graph JSON:\n```json\n{state.execution_graph.model_dump_json(indent=2)}\n```"
            except Exception as e: state.response = f"Error serializing graph: {e}"
        state.update_scratchpad_reason(tool_name, "Provided graph JSON.")
        state.next_step = "responder"
        return state

    def answer_openapi_query(self, state: BotState) -> BotState:
        tool_name = "answer_openapi_query"
        logger.info("Executing answer_openapi_query node.")
        if not state.openapi_schema:
            state.response = "No OpenAPI spec loaded."; state.next_step = "responder"; return state
        
        ctx = [f"Q: \"{state.user_input}\"", f"API Summary: {state.schema_summary or 'N/A'}",
               f"APIs ({len(state.identified_apis)}): {', '.join(a['operationId'] for a in state.identified_apis[:5])}...",
               f"Graph Goal: {state.plan_generation_goal or 'N/A'}",
               f"Graph Desc: {state.execution_graph.description if state.execution_graph else 'N/A'}"]
        if state.user_input and any(op_id in state.user_input for op_id in state.payload_descriptions): # type: ignore
            for op_id, desc in state.payload_descriptions.items():
                if op_id in state.user_input: ctx.append(f"Payload for '{op_id}': {desc[:100]}..."); break # type: ignore
        prompt = "\n".join(ctx) + "\nAnswer Q based on API info & graph. Concise. If missing, state that."
        try: state.response = llm_call_helper(self.worker_llm, prompt)
        except Exception as e: state.response = f"Error answering: {str(e)[:100]}..."
        state.update_scratchpad_reason(tool_name, f"Answered query. Response: {state.response[:100]}...")
        state.next_step = "responder"
        return state

    def interactive_query_planner(self, state: BotState) -> BotState:
        tool_name = "interactive_query_planner"
        logger.info(f"Executing {tool_name} for query: {state.user_input}")
        state.scratchpad.pop('interactive_action_plan', None); state.scratchpad.pop('current_interactive_action_idx', None)
        state.scratchpad.pop('current_interactive_results', None)

        graph_sum = state.execution_graph.description if state.execution_graph else "No graph."
        pl_keys = list(state.payload_descriptions.keys())[:3]
        prompt = f"User Query: \"{state.user_input}\"\nContext: API Summary: {state.schema_summary[:100]}... APIs: {len(state.identified_apis)}. Payloads for: {pl_keys}... Graph Goal: {state.plan_generation_goal or 'N/A'}. Graph Desc: {graph_sum[:100]}...\nInternal Actions: rerun_payload_generation(op_ids, new_context), contextualize_graph_descriptions(target_nodes, new_context), regenerate_graph_with_new_goal(new_goal), refine_existing_graph_further(instructions), answer_query_directly(query), synthesize_final_answer(instructions).\nTask: Understand query. Create JSON 'interactive_action_plan' (list of {{'action_name':'...', 'action_params':{{...}}, 'description':'...'}}). Also add 'user_query_understanding'. Output ONLY JSON."
        try:
            data = parse_llm_json_output_with_model(llm_call_helper(self.worker_llm, prompt))
            if data and "interactive_action_plan" in data and isinstance(data["interactive_action_plan"], list):
                state.scratchpad.update(data); state.scratchpad['current_interactive_action_idx'] = 0; state.scratchpad['current_interactive_results'] = []
                state.response = f"Understood: {data.get('user_query_understanding', 'N/A')}. Processing..."
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
        # Simplified: LLM updates overall description. More granular update is complex.
        prompt = f"New context: \"{ctx}\". Current graph desc: \"{state.execution_graph.description}\". Rewrite desc. New Desc:"
        try: state.execution_graph.description = llm_call_helper(self.worker_llm, prompt)
        except: pass # Ignore if LLM fails here, keep old desc
        return f"Graph descriptions contextualized for '{ctx[:30]}...'."
    def _internal_answer_query_directly(self, state: BotState, query: str) -> str:
        orig_input, state.user_input = state.user_input, query
        self.answer_openapi_query(state); direct_ans = state.response
        state.user_input, state.response = orig_input, None
        return direct_ans or "Could not answer directly."
    def _internal_synthesize_final_answer(self, state: BotState, instr: str) -> str:
        results_sum = "\n".join([str(r)[:100]+"..." for r in state.scratchpad.get('current_interactive_results',[])])
        prompt = f"User Query Understanding: {state.scratchpad.get('user_query_understanding','N/A')}. Synthesis Instr: {instr}. Internal Results:\n{results_sum}\nGraph Desc: {state.execution_graph.description if state.execution_graph else 'N/A'}. Formulate final user answer."
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
        state.update_scratchpad_reason(tool_name, f"Executing ({idx+1}/{len(plan)}): {name} - {desc}")
        logger.info(f"Executing interactive action: {name} with params: {params}")
        res_msg = f"Action '{name}' done."; current_step_resp = ""
        try:
            if name == "rerun_payload_generation": res_msg = self._internal_rerun_payload_generation(state, params.get("operation_ids_to_update",[]), params.get("new_context",""))
            elif name == "contextualize_graph_descriptions": res_msg = self._internal_contextualize_graph(state, params.get("target_node_operation_ids"), params.get("new_context",""))
            elif name == "regenerate_graph_with_new_goal" and params.get("new_goal_string"):
                state.plan_generation_goal = params["new_goal_string"]; state.graph_refinement_iterations = 0; state.execution_graph = None
                self._generate_execution_graph(state, goal=state.plan_generation_goal)
                results.append(f"Started new graph for: {state.plan_generation_goal}"); state.scratchpad['current_interactive_results'] = results
                return state # Main graph flow takes over
            elif name == "refine_existing_graph_further" and state.execution_graph:
                state.graph_regeneration_reason = params.get("refinement_instructions","")
                self.refine_api_graph(state)
                results.append(f"Started graph refinement."); state.scratchpad['current_interactive_results'] = results
                return state # Main graph flow takes over
            elif name == "answer_query_directly": state.response = self._internal_answer_query_directly(state, params.get("query_for_synthesizer", state.user_input or ""))
            elif name == "synthesize_final_answer": state.response = self._internal_synthesize_final_answer(state, params.get("synthesis_prompt_instructions","Summarize."))
            else: res_msg = f"Unknown action: {name}."
        except Exception as e: logger.error(f"Error in action {name}: {e}", exc_info=False); res_msg = f"Error in '{name}': {str(e)[:100]}..."
        
        results.append(res_msg); state.scratchpad['current_interactive_results'] = results
        state.scratchpad['current_interactive_action_idx'] = idx + 1
        state.response = state.response or current_step_resp or f"Step {idx+1} ({name}) done."
        
        if state.scratchpad['current_interactive_action_idx'] < len(plan): state.next_step = "interactive_query_executor"
        else: logger.info("All interactive actions done."); state.next_step = "responder"
        return state

    def handle_unknown(self, state: BotState) -> BotState:
        tool_name = "handle_unknown"
        logger.warning(f"Executing {tool_name}. Current response (if any): {state.response}")
        # Preserve existing error message if one was set by a failing node
        if not state.response or "error" not in state.response.lower(): # type: ignore
            state.response = "I'm not sure how to process that. Could you rephrase or provide an OpenAPI spec first?"
        state.update_scratchpad_reason(tool_name, f"Handling unknown. Final response to be: {state.response}")
        state.next_step = "responder"
        return state

    def handle_loop(self, state: BotState) -> BotState:
        state.response = "It seems we're in a loop. Please rephrase."
        state.loop_counter = 0; state.next_step = "responder"
        state.update_scratchpad_reason("handle_loop", "Loop detected, routing to responder.")
        return state

