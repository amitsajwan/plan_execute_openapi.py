# workflow_executor.py
import asyncio
import logging
import operator
from typing import Any, Callable, Awaitable, Dict, Optional, List, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # Or another checkpointer
from langchain_core.runnables import RunnableLambda

# Assuming api_executor.py exists with APIExecutor class
# from api_executor import APIExecutor # Placeholder, ensure this exists
class APIExecutor: # Basic Placeholder
    async def execute_api(
        self,
        operationId: str,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        path_params: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # This is a mock implementation. Replace with your actual API calling logic.
        logging.info(f"MOCK EXECUTE: {method} {endpoint}, Payload: {str(payload)[:200]}..., PathParams: {path_params}, QueryParams: {query_params}")
        await asyncio.sleep(0.5) # Simulate network latency
        # Simulate response based on method or operationId
        if method.upper() == "GET":
            return {
                "status_code": 200,
                "response_body": {"id": "123", "data": f"Mock data for {operationId}", "source_op": operationId},
                "execution_time": 0.5,
                "headers": {"content-type": "application/json"}
            }
        elif method.upper() in ["POST", "PUT"]:
             return {
                "status_code": 201 if method.upper() == "POST" else 200,
                "response_body": {"id": "new_resource_123", "message": f"{operationId} executed successfully", "input_payload": payload},
                "execution_time": 0.5,
                "headers": {"content-type": "application/json"}
            }
        return { # Default error response
            "status_code": 500,
            "response_body": {"error": "Mock internal server error for " + operationId},
            "execution_time": 0.5,
            "headers": {"content-type": "application/json"}
        }


# Importing from the existing models.py
from models import GraphOutput, Edge, Node, InputMapping, OutputMapping, BotState

logger = logging.getLogger(__name__)


class WorkflowExecutionState(BaseModel):
    """Internal state for the workflow execution graph."""
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    execution_log: List[Dict[str, Any]] = Field(default_factory=list) # Keep this as Dict[str, Any] but ensure contents are simple
    current_node_id: Optional[str] = None
    interrupt_payload_override: Optional[Dict[str, Any]] = None # Should be JSON-serializable

    class Config:
        extra = 'allow'


class WorkflowExecutor:
    def __init__(
        self,
        workflow_definition: GraphOutput,
        api_executor_instance: APIExecutor,
        websocket_callback: Callable[[str, Dict, str], Awaitable[None]],
        session_id: str,
        initial_extracted_data: Optional[Dict[str, Any]] = None
    ):
        if not workflow_definition or not workflow_definition.nodes:
            raise ValueError("Workflow definition (GraphOutput) is empty or has no nodes.")

        self.workflow_def = workflow_definition
        self.api_executor = api_executor_instance
        self.websocket_callback = websocket_callback
        self.session_id = session_id
        self.resume_queue: asyncio.Queue = asyncio.Queue()

        builder = StateGraph(WorkflowExecutionState)

        for node_def in self.workflow_def.nodes:
            if node_def.effective_id.upper() not in ["START_NODE", "END_NODE"]:
                builder.add_node(
                    node_def.effective_id,
                    RunnableLambda(self._execute_api_node_wrapper(node_def))
                )

        for edge_def in self.workflow_def.edges:
            if edge_def.from_node.upper() == "START_NODE":
                builder.add_edge(START, edge_def.to_node)
            elif edge_def.to_node.upper() == "END_NODE":
                builder.add_edge(edge_def.from_node, END)
            else:
                builder.add_edge(edge_def.from_node, edge_def.to_node)

        # Added debug=True for potentially more verbose logging from LangGraph
        self.compiled_graph = builder.compile(checkpointer=MemorySaver(), debug=True)
        logger.info(f"[{self.session_id}] WorkflowExecutor: Graph compiled with debug mode, {len(self.workflow_def.nodes)} nodes and {len(self.workflow_def.edges)} edges.")

        self.initial_execution_state = WorkflowExecutionState(extracted_data=initial_extracted_data or {})


    def _resolve_json_path(self, data_source: Dict[str, Any], json_path: str) -> Any:
        if not isinstance(data_source, dict): # Guard against non-dict data_source
            logger.warning(f"[{self.session_id}] _resolve_json_path: data_source is not a dict for path '{json_path}'. Type: {type(data_source)}")
            return None
        if not json_path.startswith('$.'):
            logger.warning(f"[{self.session_id}] Invalid JSONPath (must start with '$'): {json_path}")
            return None
        
        parts = json_path[2:].split('.')
        current_data = data_source
        for part in parts:
            if isinstance(current_data, dict):
                current_data = current_data.get(part)
            elif isinstance(current_data, list):
                try:
                    idx = int(part)
                    if 0 <= idx < len(current_data):
                        current_data = current_data[idx]
                    else:
                        logger.warning(f"[{self.session_id}] JSONPath index out of bounds: {part} in {json_path}")
                        return None
                except ValueError:
                    logger.warning(f"[{self.session_id}] JSONPath invalid list index: {part} in {json_path}")
                    return None
            else:
                logger.warning(f"[{self.session_id}] JSONPath cannot traverse part '{part}' in {json_path} for data: {current_data}")
                return None
            if current_data is None:
                logger.debug(f"[{self.session_id}] JSONPath resolution returned None at part '{part}' for path '{json_path}'")
                return None
        return current_data

    def _prepare_api_inputs(self, node_def: Node, current_state: WorkflowExecutionState) -> Dict[str, Any]:
        path_params: Dict[str, Any] = {}
        query_params: Dict[str, Any] = {}
        headers: Dict[str, Any] = {}
        request_body_parts: Dict[str, Any] = {}
        final_request_body: Optional[Dict[str, Any]] = None

        for mapping in node_def.input_mappings:
            # Ensure current_state.extracted_data is a dict before resolving
            if not isinstance(current_state.extracted_data, dict):
                logger.error(f"[{self.session_id}] Extracted data is not a dictionary. Cannot process input mappings for node {node_def.effective_id}.")
                break # Stop processing mappings if extracted_data is corrupt

            value = self._resolve_json_path(current_state.extracted_data, mapping.source_data_path)
            if value is None:
                logger.warning(f"[{self.session_id}] Could not resolve value for input '{mapping.target_parameter_name}' from source path '{mapping.source_data_path}' for node '{node_def.effective_id}'.")
                continue

            if mapping.target_parameter_in == "path":
                path_params[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "query":
                query_params[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "header":
                headers[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "body":
                if isinstance(value, dict):
                    if final_request_body is None: final_request_body = {}
                    final_request_body.update(value)
                else: # If value is not a dict, it becomes the body directly
                    final_request_body = value
            elif mapping.target_parameter_in.startswith("body."):
                field_name = mapping.target_parameter_in.split(".", 1)[1]
                keys = field_name.split('.')
                current_level = request_body_parts
                for i, key_part in enumerate(keys):
                    if i == len(keys) - 1:
                        current_level[key_part] = value
                    else:
                        current_level = current_level.setdefault(key_part, {})
            else:
                logger.warning(f"[{self.session_id}] Unsupported target_parameter_in: {mapping.target_parameter_in} for node {node_def.effective_id}")
        
        if not final_request_body and request_body_parts:
            final_request_body = request_body_parts
        
        logger.debug(f"[{self.session_id}] Prepared inputs for '{node_def.effective_id}': PathP={path_params}, QueryP={query_params}, Body={str(final_request_body)[:200]}...")
        return {
            "path_params": path_params,
            "query_params": query_params,
            "headers": headers,
            "request_body": final_request_body, # This should be JSON-serializable
        }

    def _execute_api_node_wrapper(self, node_def: Node) -> Callable[[WorkflowExecutionState], Awaitable[WorkflowExecutionState]]:
        async def _run_node_instance(current_state: WorkflowExecutionState) -> WorkflowExecutionState:
            node_id = node_def.effective_id
            logger.info(f"[{self.session_id}] Executing node: {node_id} (OpID: {node_def.operationId})")
            await self.websocket_callback("node_execution_started", {"node_id": node_id, "operationId": node_def.operationId}, self.session_id)
            
            current_state.current_node_id = node_id
            
            prepared_inputs = self._prepare_api_inputs(node_def, current_state)
            method = node_def.method
            api_path = node_def.path

            log_entry_base = {"node_id": node_id, "operationId": node_def.operationId, "method": method, "path_template": api_path}

            if not method or not api_path:
                error_msg = f"Node '{node_id}' is missing 'method' or 'path' definition."
                logger.error(f"[{self.session_id}] {error_msg}")
                await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg}, self.session_id)
                current_state.execution_log.append({**log_entry_base, "status": "error", "error": error_msg})
                return current_state

            final_api_path = api_path
            for p_name, p_val in prepared_inputs["path_params"].items():
                final_api_path = final_api_path.replace(f"{{{p_name}}}", str(p_val))
            log_entry_base["calculated_path"] = final_api_path

            actual_payload_for_api = prepared_inputs["request_body"] # Should be JSON-serializable

            if node_def.requires_confirmation:
                if current_state.interrupt_payload_override:
                    logger.info(f"[{self.session_id}] Node {node_id} using user-provided override payload.")
                    actual_payload_for_api = current_state.interrupt_payload_override
                    current_state.interrupt_payload_override = None
                else:
                    logger.info(f"[{self.session_id}] Node {node_id} requires confirmation. Interrupting workflow.")
                    await self.websocket_callback(
                        "interrupt_confirmation_required",
                        {
                            "node_id": node_id, "operationId": node_def.operationId,
                            "method": method, "path_template": api_path,
                            "calculated_path": final_api_path,
                            "payload_template": node_def.payload_description, # String
                            "calculated_payload": actual_payload_for_api, # JSON-serializable dict/list
                            "query_params": prepared_inputs["query_params"], # Dict of simple types
                            "path_params": prepared_inputs["path_params"]    # Dict of simple types
                        },
                        self.session_id
                    )
                    try:
                        logger.info(f"[{self.session_id}] Node {node_id} awaiting user confirmation via resume_queue...")
                        confirmed_payload_from_user = await asyncio.wait_for(self.resume_queue.get(), timeout=600)
                        self.resume_queue.task_done()
                        logger.info(f"[{self.session_id}] Node {node_id} received confirmation payload: {str(confirmed_payload_from_user)[:100]}...")
                        actual_payload_for_api = confirmed_payload_from_user # This should also be JSON-serializable
                        await self.websocket_callback("node_resumed_with_payload", {"node_id": node_id}, self.session_id)
                    except asyncio.TimeoutError:
                        error_msg = f"Node {node_id} confirmation timed out."
                        logger.error(f"[{self.session_id}] {error_msg}")
                        await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg, "reason": "timeout"}, self.session_id)
                        current_state.execution_log.append({**log_entry_base, "status": "error", "error": error_msg, "reason": "timeout"})
                        current_state.current_node_id = None
                        return current_state
                    except Exception as e_q:
                        error_msg = f"Error during confirmation wait for {node_id}: {str(e_q)}"
                        logger.error(f"[{self.session_id}] {error_msg}", exc_info=True)
                        await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg}, self.session_id)
                        current_state.execution_log.append({**log_entry_base, "status": "error", "error": error_msg})
                        current_state.current_node_id = None
                        return current_state
            
            log_entry_base["payload_sent_preview"] = str(actual_payload_for_api)[:200] # Log a preview

            try:
                api_result = await self.api_executor.execute_api(
                    operationId=node_def.operationId, method=method, endpoint=final_api_path,
                    payload=actual_payload_for_api, # Must be JSON serializable if method requires body
                    query_params=prepared_inputs["query_params"],
                    headers=prepared_inputs["headers"]
                )
                # api_result is a Dict from APIExecutor, assumed to be JSON-serializable
                
                status_code = api_result.get("status_code")
                response_body_preview = str(api_result.get("response_body"))[:200] # Keep it short for logging
                execution_time = api_result.get("execution_time")

                logger.info(f"[{self.session_id}] API call for node {node_id} completed. Status: {status_code}, Time: {execution_time}s")
                
                newly_extracted_data = {}
                if status_code and 200 <= status_code < 300:
                    await self.websocket_callback("node_execution_succeeded", {"node_id": node_id, "result_preview": response_body_preview}, self.session_id)
                    
                    # Ensure response_body is a dict for _resolve_json_path
                    response_body_for_extraction = api_result.get("response_body", {})
                    if not isinstance(response_body_for_extraction, dict):
                        logger.warning(f"[{self.session_id}] Response body for {node_id} is not a dict, type: {type(response_body_for_extraction)}. Output extraction may fail.")
                        response_body_for_extraction = {} # Default to empty dict to prevent errors

                    for out_map in node_def.output_mappings:
                        extracted_value = self._resolve_json_path(response_body_for_extraction, out_map.source_data_path)
                        if extracted_value is not None:
                            # Ensure extracted_value is serializable if it's complex
                            # For now, assume _resolve_json_path returns simple types or serializable dicts/lists
                            newly_extracted_data[out_map.target_data_key] = extracted_value
                            logger.debug(f"[{self.session_id}] Extracted for '{node_id}': '{out_map.target_data_key}' = {str(extracted_value)[:50]}...")
                    
                    current_state.execution_log.append({
                        **log_entry_base, "status": "success", 
                        "status_code": status_code, "response_preview": response_body_preview,
                        "execution_time_seconds": execution_time,
                        "extracted_here": newly_extracted_data.copy() # Store copy of what was extracted by this node
                    })
                else:
                    error_body_str = str(api_result.get("response_body", "No error body"))
                    logger.error(f"[{self.session_id}] Node {node_id} API call failed or non-2xx status: {status_code}. Response: {error_body_str[:200]}")
                    await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_body_str[:500], "status_code": status_code}, self.session_id)
                    current_state.execution_log.append({
                        **log_entry_base, "status": "api_error", 
                        "status_code": status_code, "error_response_preview": error_body_str[:200],
                        "execution_time_seconds": execution_time
                    })

                current_state.extracted_data.update(newly_extracted_data)

            except Exception as e_api_call:
                error_str = str(e_api_call)
                logger.critical(f"[{self.session_id}] Unhandled exception during API execution for node {node_id}: {error_str}", exc_info=True)
                await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_str[:500]}, self.session_id)
                current_state.execution_log.append({**log_entry_base, "status": "exception", "error": error_str[:200]})
            
            current_state.current_node_id = None
            return current_state
        return _run_node_instance


    async def run_workflow_streaming(self, thread_config: Dict[str, Any]):
        logger.info(f"[{self.session_id}] Starting workflow streaming. Config: {thread_config}")
        await self.websocket_callback("workflow_execution_started", {"session_id": self.session_id, "graph_description": self.workflow_def.description}, self.session_id)

        final_state_to_return = None # Keep track of the most recent valid state
        try:
            async for event in self.compiled_graph.astream_events(self.initial_execution_state, config=thread_config, version="v1"):
                event_type = event["event"]
                event_name = event.get("name", "unknown_event_source")
                event_data = event.get("data", {})

                logger.debug(f"[{self.session_id}] Workflow Stream Event: Type='{event_type}', Name='{event_name}', DataKeys='{list(event_data.keys())}'")

                if event_type == "on_chain_end": 
                    node_id_finished = event_name
                    output_state_candidate = event_data.get("output")
                    
                    if isinstance(output_state_candidate, WorkflowExecutionState):
                        final_state_to_return = output_state_candidate 
                        if node_id_finished and node_id_finished not in [START, END, "__start__", "__end__"]:
                            logger.info(f"[{self.session_id}] Event: Node '{node_id_finished}' finished processing.")
                            last_log_entry = next((log for log in reversed(final_state_to_return.execution_log) if log.get("node_id") == node_id_finished), None)
                            if last_log_entry:
                                await self.websocket_callback("node_log_update", {"node_id": node_id_finished, "log_entry": last_log_entry}, self.session_id)
                        
                        if final_state_to_return.current_node_id:
                             logger.warning(f"[{self.session_id}] Node '{final_state_to_return.current_node_id}' appears finished but current_node_id is still set in state. This might indicate an issue if it wasn't an intended pause.")

                elif event_type == "on_graph_end":
                    final_output_data = event_data.get("output")
                    if isinstance(final_output_data, WorkflowExecutionState):
                        final_state_to_return = final_output_data
                    else: # If output is not WorkflowExecutionState, try to use the last known valid state
                        logger.warning(f"[{self.session_id}] on_graph_end output was not WorkflowExecutionState. Type: {type(final_output_data)}. Using last known valid state if available.")
                    
                    log_final_state = final_state_to_return.dict() if final_state_to_return else {}
                    logger.info(f"[{self.session_id}] Workflow execution finished. Final state (from on_graph_end or last known): {str(log_final_state)[:500]}...")
                    await self.websocket_callback("workflow_execution_completed", {"final_state": log_final_state}, self.session_id)
                    return 

                elif event_type in ["on_tool_error", "on_chain_error", "on_node_error", "on_llm_error", "on_retriever_error"]: # Broader error catching
                    node_name_with_error = event_name
                    # Try to get error from 'output' if it's a dict, else stringify event_data
                    error_details_raw = event_data.get("output", event_data)
                    error_details_str = str(error_details_raw)
                    
                    logger.error(f"[{self.session_id}] Error event '{event_type}' in workflow at '{node_name_with_error}': {error_details_str[:500]}")
                    await self.websocket_callback("workflow_execution_failed", {"node_id": node_name_with_error, "error_event_type": event_type, "error": error_details_str[:1000]}, self.session_id) # Send more details
                    return 
        
        except Exception as e_stream:
            error_str_stream = str(e_stream)
            logger.critical(f"[{self.session_id}] Unhandled exception during workflow streaming: {error_str_stream}", exc_info=True)
            await self.websocket_callback("workflow_execution_failed", {"error": f"Critical streaming error: {error_str_stream[:500]}"}, self.session_id)
            return

        # Fallback if on_graph_end wasn't explicitly caught (e.g., stream ended prematurely)
        logger.info(f"[{self.session_id}] Workflow streaming loop completed (possibly without explicit on_graph_end).")
        final_state_dict = {}
        if final_state_to_return and isinstance(final_state_to_return, WorkflowExecutionState):
            try:
                final_state_dict = final_state_to_return.model_dump(exclude={'execution_log': {'__all__': {'result', 'full_result'}}}) # Example of excluding large fields
            except Exception: # Fallback if model_dump fails
                final_state_dict = {"message": "State could not be fully serialized."}
        
        await self.websocket_callback("workflow_execution_completed", {"message": "Streaming ended.", "final_state_summary": final_state_dict}, self.session_id)


    async def submit_interrupt_value(self, value: Dict[str, Any]):
        """ Called by the server to provide data for a paused node. """
        # Ensure value is JSON-serializable as it might be part of state if an error occurs during its processing
        try:
            json.dumps(value) # Test serializability
            logger.info(f"[{self.session_id}] Interrupt value submitted to queue: {str(value)[:100]}...")
            await self.resume_queue.put(value)
        except TypeError as e:
            logger.error(f"[{self.session_id}] Failed to submit non-JSON-serializable interrupt value: {e}. Value: {str(value)[:200]}")
            # Optionally, notify client about this error if possible, or handle appropriately.

