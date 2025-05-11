# workflow_executor.py
import asyncio
import logging
import json # For testing serializability
from typing import Any, Callable, Awaitable, Dict, Optional, List, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda

# Assuming models.py and api_executor.py are correctly defined
from models import GraphOutput, Node, InputMapping, OutputMapping # BotState is not directly used here
# from api_executor import APIExecutor # Assuming this is your actual API executor

# Basic Placeholder for APIExecutor if not using the one from api_executor.py
class APIExecutor:
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
        logging.info(f"MOCK EXECUTE: {method} {endpoint}, Payload: {str(payload)[:200]}..., PathParams: {path_params}, QueryParams: {query_params}")
        await asyncio.sleep(0.2) # Reduced sleep for faster testing
        if method.upper() == "GET":
            return {
                "status_code": 200,
                "response_body": {"id": operationId + "_123", "data": f"Mock data for {operationId}", "source_op": operationId},
                "execution_time": 0.2,
                "headers": {"content-type": "application/json"}
            }
        elif method.upper() in ["POST", "PUT"]:
             return {
                "status_code": 201 if method.upper() == "POST" else 200,
                "response_body": {"id": "new_" + operationId + "_789", "message": f"{operationId} executed successfully", "input_payload": payload},
                "execution_time": 0.2,
                "headers": {"content-type": "application/json"}
            }
        return {
            "status_code": 500,
            "response_body": {"error": "Mock internal server error for " + operationId},
            "execution_time": 0.2,
            "headers": {"content-type": "application/json"}
        }

logger = logging.getLogger(__name__)

class WorkflowExecutionState(BaseModel):
    """Internal state for the workflow execution graph."""
    extracted_data: Dict[str, Any] = Field(default_factory=dict) # Ensure values are serializable
    execution_log: List[Dict[str, Any]] = Field(default_factory=list) # Ensure dicts contain serializable values
    current_node_id: Optional[str] = None
    interrupt_payload_override: Optional[Dict[str, Any]] = None # Should be JSON-serializable

    class Config:
        extra = 'allow'
        # It's good practice for state objects used with LangGraph to be fully serializable
        # Pydantic models are generally good, but ensure complex custom types are handled.


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
                # Create a new instance of the wrapper for each node to capture node_def
                node_runner_instance = self._execute_api_node_wrapper(node_def)
                builder.add_node(node_def.effective_id, node_runner_instance)


        for edge_def in self.workflow_def.edges:
            if edge_def.from_node.upper() == "START_NODE":
                builder.add_edge(START, edge_def.to_node)
            elif edge_def.to_node.upper() == "END_NODE":
                builder.add_edge(edge_def.from_node, END)
            else:
                builder.add_edge(edge_def.from_node, edge_def.to_node)

        self.compiled_graph = builder.compile(checkpointer=MemorySaver(), debug=True)
        logger.info(f"[{self.session_id}] WorkflowExecutor: Graph compiled with debug mode for {len(self.workflow_def.nodes)} nodes.")
        self.initial_execution_state = WorkflowExecutionState(extracted_data=initial_extracted_data or {})

    def _is_json_serializable(self, data: Any) -> bool:
        """Helper to check if data is JSON serializable."""
        if data is None: return True
        try:
            json.dumps(data)
            return True
        except (TypeError, OverflowError):
            return False

    def _ensure_serializable_dict(self, d: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Ensures all values in a dictionary are JSON serializable, converting if necessary."""
        serializable_d = {}
        for k, v in d.items():
            if not self._is_json_serializable(v):
                logger.warning(f"[{self.session_id}] Non-serializable value for key '{k}' in {context}. Converting to string. Type: {type(v)}")
                serializable_d[k] = str(v)
            else:
                serializable_d[k] = v
        return serializable_d

    def _resolve_json_path(self, data_source: Dict[str, Any], json_path: str) -> Any:
        if not isinstance(data_source, dict):
            logger.warning(f"[{self.session_id}] _resolve_json_path: data_source is not a dict for path '{json_path}'. Type: {type(data_source)}")
            return None
        if not json_path.startswith('$.'):
            logger.warning(f"[{self.session_id}] Invalid JSONPath (must start with '$'): {json_path}")
            return None
        
        parts = json_path[2:].split('.')
        current_data = data_source
        for part_idx, part in enumerate(parts):
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
                logger.warning(f"[{self.session_id}] JSONPath cannot traverse part '{part}' (index {part_idx}) in '{json_path}' for data type: {type(current_data)}. Current data preview: {str(current_data)[:100]}")
                return None
            if current_data is None and part_idx < len(parts) -1: # Path resolved to None before reaching the end
                logger.debug(f"[{self.session_id}] JSONPath resolution became None at part '{part}' for path '{json_path}' before reaching end.")
                return None
        return current_data


    def _prepare_api_inputs(self, node_def: Node, current_state: WorkflowExecutionState) -> Dict[str, Any]:
        path_params: Dict[str, Any] = {}
        query_params: Dict[str, Any] = {}
        headers: Dict[str, Any] = {}
        request_body_parts: Dict[str, Any] = {}
        final_request_body: Optional[Any] = None # Can be dict, list, or primitive for some APIs

        for mapping in node_def.input_mappings:
            if not isinstance(current_state.extracted_data, dict):
                logger.error(f"[{self.session_id}] Extracted data is not a dictionary. Node: {node_def.effective_id}.")
                break 

            value = self._resolve_json_path(current_state.extracted_data, mapping.source_data_path)
            if value is None:
                logger.warning(f"[{self.session_id}] Input mapping: Could not resolve '{mapping.source_data_path}' for target '{mapping.target_parameter_name}' in node '{node_def.effective_id}'.")
                continue
            
            # Ensure value is serializable if it's complex (though resolve_json_path should yield simple types from JSON)
            if not self._is_json_serializable(value):
                logger.warning(f"[{self.session_id}] Resolved input value for '{mapping.target_parameter_name}' is not JSON serializable. Path: '{mapping.source_data_path}'. Converting to string. Type: {type(value)}")
                value = str(value)

            if mapping.target_parameter_in == "path":
                path_params[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "query":
                query_params[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "header":
                headers[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "body":
                if isinstance(value, dict) and isinstance(final_request_body, dict):
                    final_request_body.update(value)
                elif final_request_body is None:
                    final_request_body = value # Allows body to be list, string, dict, etc.
                else: # final_request_body exists but is not a dict, or value is not a dict to merge
                    logger.warning(f"[{self.session_id}] Multiple 'body' mappings for node {node_def.effective_id} with incompatible types or overwrite. Current body type: {type(final_request_body)}, new value type: {type(value)}. Overwriting.")
                    final_request_body = value

            elif mapping.target_parameter_in.startswith("body."):
                if final_request_body is not None and not isinstance(final_request_body, dict):
                    logger.warning(f"[{self.session_id}] Cannot map 'body.fieldName' for node {node_def.effective_id} because existing 'body' is not a dictionary. Type: {type(final_request_body)}. Ignoring 'body.fieldName' mapping.")
                    continue
                if final_request_body is None: final_request_body = {} # Initialize as dict if using fieldName

                field_name = mapping.target_parameter_in.split(".", 1)[1]
                keys = field_name.split('.')
                current_level = final_request_body # type: ignore
                for i, key_part in enumerate(keys):
                    if i == len(keys) - 1:
                        current_level[key_part] = value
                    else:
                        current_level = current_level.setdefault(key_part, {})
            else:
                logger.warning(f"[{self.session_id}] Unsupported target_parameter_in: {mapping.target_parameter_in} for node {node_def.effective_id}")
        
        # If request_body_parts were used (e.g. body.fieldName) and final_request_body was built as a dict
        if isinstance(final_request_body, dict) and request_body_parts: # request_body_parts is usually for building final_request_body
            pass # final_request_body is already being built correctly

        logger.debug(f"[{self.session_id}] Prepared inputs for '{node_def.effective_id}': PathP={path_params}, QueryP={query_params}, BodyPreview={str(final_request_body)[:150]}...")
        return {
            "path_params": self._ensure_serializable_dict(path_params, "path_params"),
            "query_params": self._ensure_serializable_dict(query_params, "query_params"),
            "headers": self._ensure_serializable_dict(headers, "headers"),
            "request_body": final_request_body, # Assumed to be serializable by this point or handled by APIExecutor
        }

    def _execute_api_node_wrapper(self, node_def: Node) -> Callable[[WorkflowExecutionState], Awaitable[WorkflowExecutionState]]:
        async def _run_node_instance(current_state: WorkflowExecutionState) -> WorkflowExecutionState:
            node_id = node_def.effective_id
            logger.info(f"[{self.session_id}] Executing node: {node_id} (OpID: {node_def.operationId})")
            await self.websocket_callback("node_execution_started", {"node_id": node_id, "operationId": node_def.operationId}, self.session_id)
            
            current_state.current_node_id = node_id # Set before any potential await
            
            prepared_inputs = self._prepare_api_inputs(node_def, current_state)
            method = node_def.method
            api_path = node_def.path

            log_entry_base = {
                "node_id": node_id, "operationId": node_def.operationId, 
                "method": method, "path_template": api_path
            }

            if not method or not api_path:
                error_msg = f"Node '{node_id}' is missing 'method' or 'path'."
                logger.error(f"[{self.session_id}] {error_msg}")
                await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg}, self.session_id)
                current_state.execution_log.append({**log_entry_base, "status": "config_error", "error": error_msg})
                current_state.current_node_id = None # Clear before returning
                return current_state

            final_api_path = api_path
            for p_name, p_val in prepared_inputs["path_params"].items():
                final_api_path = final_api_path.replace(f"{{{p_name}}}", str(p_val))
            log_entry_base["calculated_path"] = final_api_path

            actual_payload_for_api = prepared_inputs["request_body"]

            if node_def.requires_confirmation:
                if current_state.interrupt_payload_override:
                    logger.info(f"[{self.session_id}] Node {node_id} using override payload: {str(current_state.interrupt_payload_override)[:100]}...")
                    actual_payload_for_api = current_state.interrupt_payload_override
                    current_state.interrupt_payload_override = None # Consume
                else:
                    logger.info(f"[{self.session_id}] Node {node_id} requires confirmation. Interrupting.")
                    # Ensure all data sent to websocket is serializable
                    ws_payload_data = {
                        "node_id": node_id, "operationId": node_def.operationId,
                        "method": method, "path_template": api_path,
                        "calculated_path": final_api_path,
                        "payload_template": node_def.payload_description,
                        "calculated_payload": actual_payload_for_api if self._is_json_serializable(actual_payload_for_api) else str(actual_payload_for_api),
                        "query_params": prepared_inputs["query_params"],
                        "path_params": prepared_inputs["path_params"]
                    }
                    await self.websocket_callback("interrupt_confirmation_required", ws_payload_data, self.session_id)
                    
                    confirmed_payload_from_user = None
                    try:
                        logger.info(f"[{self.session_id}] Node {node_id} awaiting confirmation via resume_queue...")
                        confirmed_payload_from_user = await asyncio.wait_for(self.resume_queue.get(), timeout=600)
                        logger.info(f"[{self.session_id}] Node {node_id} received from queue: {str(confirmed_payload_from_user)[:100]}...")
                        actual_payload_for_api = confirmed_payload_from_user
                        await self.websocket_callback("node_resumed_with_payload", {"node_id": node_id}, self.session_id)
                    except asyncio.TimeoutError:
                        error_msg = f"Node {node_id} confirmation timed out."
                        logger.error(f"[{self.session_id}] {error_msg}")
                        await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg, "reason": "timeout"}, self.session_id)
                        current_state.execution_log.append({**log_entry_base, "status": "timeout_error", "error": error_msg})
                        current_state.current_node_id = None
                        return current_state # Important to return to stop this path
                    except asyncio.CancelledError: # Handle task cancellation gracefully
                        error_msg = f"Node {node_id} confirmation wait cancelled."
                        logger.warning(f"[{self.session_id}] {error_msg}")
                        # Don't send websocket callback here as the connection might be closing
                        current_state.execution_log.append({**log_entry_base, "status": "cancelled_error", "error": error_msg})
                        current_state.current_node_id = None
                        raise # Re-raise CancelledError to allow LangGraph to handle it
                    except Exception as e_q:
                        error_msg = f"Error during confirmation wait for {node_id}: {str(e_q)}"
                        logger.error(f"[{self.session_id}] {error_msg}", exc_info=True)
                        await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg}, self.session_id)
                        current_state.execution_log.append({**log_entry_base, "status": "queue_error", "error": error_msg})
                        current_state.current_node_id = None
                        return current_state
                    finally:
                        if confirmed_payload_from_user is not None: # Only call task_done if get() succeeded
                             self.resume_queue.task_done()
            
            log_entry_base["payload_sent_preview"] = str(actual_payload_for_api)[:150]

            try:
                api_result = await self.api_executor.execute_api(
                    operationId=node_def.operationId, method=method, endpoint=final_api_path,
                    payload=actual_payload_for_api,
                    query_params=prepared_inputs["query_params"],
                    headers=prepared_inputs["headers"]
                )
                
                status_code = api_result.get("status_code")
                # Ensure response_body is serializable for logging and extraction
                response_body_raw = api_result.get("response_body")
                if self._is_json_serializable(response_body_raw):
                    response_body_for_log_and_extract = response_body_raw
                else:
                    logger.warning(f"[{self.session_id}] API response body for {node_id} is not JSON serializable. Type: {type(response_body_raw)}. Converting to string for log/extract.")
                    response_body_for_log_and_extract = str(response_body_raw)
                
                response_body_preview = str(response_body_for_log_and_extract)[:150]
                execution_time = api_result.get("execution_time")

                logger.info(f"[{self.session_id}] API call for {node_id} done. Status: {status_code}, Time: {execution_time}s")
                
                newly_extracted_data = {}
                log_status = "api_error" # Default for non-2xx or if status_code is None

                if status_code and 200 <= status_code < 300:
                    log_status = "success"
                    await self.websocket_callback("node_execution_succeeded", {"node_id": node_id, "result_preview": response_body_preview}, self.session_id)
                    
                    # Use the potentially stringified response_body_for_log_and_extract if original wasn't dict
                    body_to_extract_from = response_body_for_log_and_extract
                    if not isinstance(body_to_extract_from, dict): # If it was stringified, try to parse if it looks like JSON
                        if isinstance(body_to_extract_from, str) and body_to_extract_from.strip().startswith("{") and body_to_extract_from.strip().endswith("}"):
                            try: body_to_extract_from = json.loads(body_to_extract_from)
                            except json.JSONDecodeError:
                                logger.warning(f"[{self.session_id}] Stringified response body for {node_id} looked like JSON but failed to parse for extraction.")
                                body_to_extract_from = {} # Give up on extraction from this body
                        else: # Not a dict, not a JSON string
                             body_to_extract_from = {}


                    for out_map in node_def.output_mappings:
                        extracted_value = self._resolve_json_path(body_to_extract_from, out_map.source_data_path)
                        if extracted_value is not None:
                            if self._is_json_serializable(extracted_value):
                                newly_extracted_data[out_map.target_data_key] = extracted_value
                            else:
                                logger.warning(f"[{self.session_id}] Extracted value for '{out_map.target_data_key}' (node {node_id}) is not JSON serializable. Type: {type(extracted_value)}. Storing as string.")
                                newly_extracted_data[out_map.target_data_key] = str(extracted_value)
                            logger.debug(f"[{self.session_id}] Extracted for '{node_id}': '{out_map.target_data_key}' = {str(newly_extracted_data[out_map.target_data_key])[:50]}...")
                    
                    current_state.execution_log.append({
                        **log_entry_base, "status": log_status, 
                        "status_code": status_code, "response_preview": response_body_preview,
                        "execution_time_seconds": execution_time,
                        "extracted_here": newly_extracted_data.copy() 
                    })
                else: # API call failed or non-2xx
                    error_msg_detail = response_body_preview if status_code else "API call did not return status_code."
                    logger.error(f"[{self.session_id}] Node {node_id} API error. Status: {status_code}. Detail: {error_msg_detail}")
                    await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg_detail, "status_code": status_code}, self.session_id)
                    current_state.execution_log.append({
                        **log_entry_base, "status": log_status, 
                        "status_code": status_code, "error_response_preview": error_msg_detail,
                        "execution_time_seconds": execution_time
                    })
                current_state.extracted_data.update(self._ensure_serializable_dict(newly_extracted_data, "newly_extracted_data"))


            except asyncio.CancelledError:
                logger.warning(f"[{self.session_id}] API call for node {node_id} was cancelled.")
                current_state.execution_log.append({**log_entry_base, "status": "cancelled_error", "error": "API call cancelled"})
                # Do not clear current_node_id yet, let the streaming loop handle cleanup if needed
                raise # Re-raise CancelledError
            except Exception as e_api_call:
                error_str = str(e_api_call)
                logger.critical(f"[{self.session_id}] Unhandled exception during API execution for node {node_id}: {error_str}", exc_info=True)
                await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_str[:500]}, self.session_id)
                current_state.execution_log.append({**log_entry_base, "status": "exception", "error": error_str[:200]})
            
            current_state.current_node_id = None # Clear after successful execution or handled error
            return current_state
        return _run_node_instance


    async def run_workflow_streaming(self, thread_config: Dict[str, Any]):
        logger.info(f"[{self.session_id}] Starting workflow streaming. Config: {thread_config}")
        if self.websocket_callback: # Guard callback
            await self.websocket_callback("workflow_execution_started", {"session_id": self.session_id, "graph_description": self.workflow_def.description}, self.session_id)

        final_state_to_return = self.initial_execution_state # Initialize with initial state
        try:
            async for event in self.compiled_graph.astream_events(self.initial_execution_state, config=thread_config, version="v1"):
                event_type = event["event"]
                event_name = event.get("name", "unknown_event_source")
                event_data = event.get("data", {})

                logger.debug(f"[{self.session_id}] WF Stream Event: Type='{event_type}', Name='{event_name}', DataKeys='{list(event_data.keys())}'")

                if event_type == "on_chain_end": 
                    node_id_finished = event_name
                    output_state_candidate = event_data.get("output")
                    
                    if isinstance(output_state_candidate, WorkflowExecutionState):
                        final_state_to_return = output_state_candidate 
                        if node_id_finished and node_id_finished not in [START, END, "__start__", "__end__"]:
                            logger.info(f"[{self.session_id}] Event: Node '{node_id_finished}' finished processing.")
                            last_log_entry = next((log for log in reversed(final_state_to_return.execution_log) if log.get("node_id") == node_id_finished), None)
                            if last_log_entry and self.websocket_callback:
                                await self.websocket_callback("node_log_update", {"node_id": node_id_finished, "log_entry": last_log_entry}, self.session_id)
                        
                        if final_state_to_return.current_node_id:
                             logger.warning(f"[{self.session_id}] Node '{final_state_to_return.current_node_id}' appears finished but current_node_id is still set in state.")

                elif event_type == "on_graph_end":
                    final_output_data = event_data.get("output")
                    if isinstance(final_output_data, WorkflowExecutionState):
                        final_state_to_return = final_output_data
                    
                    log_final_state = final_state_to_return.dict() if final_state_to_return else {}
                    logger.info(f"[{self.session_id}] Workflow execution finished (on_graph_end). Final state preview: {str(log_final_state)[:300]}...")
                    if self.websocket_callback:
                        await self.websocket_callback("workflow_execution_completed", {"final_state": log_final_state}, self.session_id)
                    return 

                elif event_type in ["on_tool_error", "on_chain_error", "on_node_error", "on_llm_error", "on_retriever_error"]:
                    node_name_with_error = event_name
                    error_details_raw = event_data.get("output", event_data)
                    error_details_str = str(error_details_raw)
                    
                    logger.error(f"[{self.session_id}] Error event '{event_type}' in workflow at '{node_name_with_error}': {error_details_str[:500]}")
                    if self.websocket_callback:
                        await self.websocket_callback("workflow_execution_failed", {"node_id": node_name_with_error, "error_event_type": event_type, "error": error_details_str[:1000]}, self.session_id)
                    return 
        
        except asyncio.CancelledError:
            logger.warning(f"[{self.session_id}] Workflow streaming task was cancelled.")
            if self.websocket_callback:
                await self.websocket_callback("workflow_execution_failed", {"error": "Workflow execution was cancelled by the system."}, self.session_id)
            # final_state_to_return might hold the state at the point of cancellation
            # LangGraph might handle checkpointing on CancelledError if checkpointer supports it
            return # Exit gracefully
        except Exception as e_stream:
            error_str_stream = str(e_stream)
            logger.critical(f"[{self.session_id}] Unhandled exception during workflow streaming: {error_str_stream}", exc_info=True)
            if self.websocket_callback:
                await self.websocket_callback("workflow_execution_failed", {"error": f"Critical streaming error: {error_str_stream[:500]}"}, self.session_id)
            return

        # Fallback if loop finishes without on_graph_end (should be rare with well-formed graphs)
        logger.info(f"[{self.session_id}] Workflow streaming loop completed (without explicit on_graph_end).")
        final_state_dict = {}
        if final_state_to_return and isinstance(final_state_to_return, WorkflowExecutionState):
            try: # Try to create a serializable summary
                final_state_dict = {
                    "extracted_data_keys": list(final_state_to_return.extracted_data.keys()),
                    "log_count": len(final_state_to_return.execution_log),
                    "last_node_id": final_state_to_return.current_node_id or (final_state_to_return.execution_log[-1]["node_id"] if final_state_to_return.execution_log else "None")
                }
            except Exception: 
                final_state_dict = {"message": "State summary could not be generated."}
        
        if self.websocket_callback:
            await self.websocket_callback("workflow_execution_completed", {"message": "Streaming ended (fallback).", "final_state_summary": final_state_dict}, self.session_id)


    async def submit_interrupt_value(self, value: Dict[str, Any]):
        try:
            # Ensure the value itself is serializable before putting it on the queue,
            # as it might be echoed or logged by the node upon reception.
            json.dumps(value) 
            logger.info(f"[{self.session_id}] Interrupt value submitted to queue: {str(value)[:100]}...")
            await self.resume_queue.put(value)
        except TypeError as e:
            logger.error(f"[{self.session_id}] Failed to submit non-JSON-serializable interrupt value to queue: {e}. Value: {str(value)[:200]}")
            # Optionally, inform the client if this submission came from a client action.
            if self.websocket_callback:
                 await self.websocket_callback("workflow_error", {"error": "Submitted data for workflow resume was not valid (not JSON serializable)."}, self.session_id)

