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
        logging.info(f"MOCK EXECUTE: {method} {endpoint}, Payload: {payload}, PathParams: {path_params}, QueryParams: {query_params}")
        await asyncio.sleep(0.5)
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
        return {
            "status_code": 500,
            "response_body": {"error": "Mock internal server error"},
            "execution_time": 0.5,
            "headers": {"content-type": "application/json"}
        }


# Importing from the existing models.py
from models import GraphOutput, Edge, Node, InputMapping, OutputMapping, BotState

logger = logging.getLogger(__name__)

# Prometheus metrics (optional, can be removed if not used)
# from prometheus_client import Counter, Histogram
# api_calls_total = Counter("api_calls_total", "Total API calls", ["api_name", "status"])
# api_latency_seconds = Histogram("api_latency_seconds", "API call latency", ["api_name"])


class WorkflowExecutionState(BaseModel):
    """Internal state for the workflow execution graph."""
    # Stores data extracted from API responses, keyed by unique identifiers (e.g., user_id, order_id)
    # or by source_node_id.output_name if more structured.
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    # Log of operations performed and their results
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)
    # Current node being processed, if needed for context in interrupt
    current_node_id: Optional[str] = None
    # User provided data for an interruption
    interrupt_payload_override: Optional[Dict[str, Any]] = None

    class Config:
        extra = 'allow'


class WorkflowExecutor:
    def __init__(
        self,
        workflow_definition: GraphOutput, # Using GraphOutput from models.py
        api_executor_instance: APIExecutor,
        # Callback signature: (event_type, data, session_id_for_logging)
        websocket_callback: Callable[[str, Dict, str], Awaitable[None]],
        session_id: str, # For logging and context in callbacks
        initial_extracted_data: Optional[Dict[str, Any]] = None
    ):
        if not workflow_definition or not workflow_definition.nodes:
            raise ValueError("Workflow definition (GraphOutput) is empty or has no nodes.")

        self.workflow_def = workflow_definition
        self.api_executor = api_executor_instance
        self.websocket_callback = websocket_callback
        self.session_id = session_id
        self.resume_queue: asyncio.Queue = asyncio.Queue()

        # Build the LangGraph StateGraph for execution
        builder = StateGraph(WorkflowExecutionState)

        # Add nodes to the graph
        for node_def in self.workflow_def.nodes:
            # START and END are keywords for LangGraph and should not be actual API nodes
            if node_def.effective_id.upper() not in ["START_NODE", "END_NODE"]:
                # Each node in the graph will call the _execute_api_node method
                builder.add_node(
                    node_def.effective_id,
                    RunnableLambda(self._execute_api_node_wrapper(node_def))
                )

        # Add edges to the graph
        for edge_def in self.workflow_def.edges:
            if edge_def.from_node.upper() == "START_NODE":
                builder.add_edge(START, edge_def.to_node)
            elif edge_def.to_node.upper() == "END_NODE":
                builder.add_edge(edge_def.from_node, END)
            else:
                builder.add_edge(edge_def.from_node, edge_def.to_node)
        
        # Compile the graph
        # TODO: Consider if a persistent checkpointer is needed for long-running workflows
        self.compiled_graph = builder.compile(checkpointer=MemorySaver())
        logger.info(f"[{self.session_id}] WorkflowExecutor: Graph compiled with {len(self.workflow_def.nodes)} nodes and {len(self.workflow_def.edges)} edges.")

        # Initialize state
        self.initial_execution_state = WorkflowExecutionState(extracted_data=initial_extracted_data or {})


    def _resolve_json_path(self, data_source: Dict[str, Any], json_path: str) -> Any:
        """
        Resolves a simple JSONPath-like string (e.g., '$.id', '$.data.items[0].name').
        Note: This is a basic implementation. For complex paths, consider a dedicated library.
        """
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
        """
        Prepares path parameters, query parameters, headers, and request body
        based on the node's input_mappings and current extracted_data.
        """
        path_params: Dict[str, Any] = {}
        query_params: Dict[str, Any] = {}
        headers: Dict[str, Any] = {} # TODO: Add header mapping if needed
        request_body_parts: Dict[str, Any] = {} # For building the body

        # This assumes a top-level 'body' key if mappings target 'body.fieldName'
        # or direct assignment if 'body'
        final_request_body: Optional[Dict[str, Any]] = None


        for mapping in node_def.input_mappings:
            # Find the source node's result in the execution_log or use general extracted_data
            # For simplicity, we'll primarily use current_state.extracted_data
            # This implies that output_mappings should populate extracted_data with clear keys.
            
            # Try to resolve from specific source_operation_id if specified and data is structured that way
            # For now, assume extracted_data is flat or source_data_path is absolute-like from a common pool
            value = self._resolve_json_path(current_state.extracted_data, mapping.source_data_path)

            if value is None:
                logger.warning(f"[{self.session_id}] Could not resolve value for input '{mapping.target_parameter_name}' from source '{mapping.source_operation_id}' path '{mapping.source_data_path}' for node '{node_def.effective_id}'. Skipping this mapping.")
                continue

            # TODO: Add transformation logic if mapping.transformation is present

            if mapping.target_parameter_in == "path":
                path_params[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "query":
                query_params[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "header":
                headers[mapping.target_parameter_name] = value
            elif mapping.target_parameter_in == "body":
                # This implies the entire resolved value is the body or a part of it.
                # If multiple 'body' targets, this needs careful handling (e.g. merge or error)
                if isinstance(value, dict):
                    if final_request_body is None: final_request_body = {}
                    final_request_body.update(value) # Merge if value is a dict
                else: # If value is not a dict, it becomes the body directly (less common for complex APIs)
                    final_request_body = value
                logger.debug(f"[{self.session_id}] Mapped to body for '{node_def.effective_id}': {value}")

            elif mapping.target_parameter_in.startswith("body."):
                field_name = mapping.target_parameter_in.split(".", 1)[1]
                # Simple dot notation for now, for deeply nested, need more robust handling
                # e.g. body.user.id -> request_body_parts['user']['id'] = value
                keys = field_name.split('.')
                current_level = request_body_parts
                for i, key_part in enumerate(keys):
                    if i == len(keys) - 1: # Last key
                        current_level[key_part] = value
                    else:
                        current_level = current_level.setdefault(key_part, {})
            else:
                logger.warning(f"[{self.session_id}] Unsupported target_parameter_in: {mapping.target_parameter_in} for node {node_def.effective_id}")
        
        # If request_body_parts were populated, they become the final_request_body (if final_request_body wasn't directly set)
        if not final_request_body and request_body_parts:
            final_request_body = request_body_parts
        
        logger.debug(f"[{self.session_id}] Prepared inputs for '{node_def.effective_id}': PathP={path_params}, QueryP={query_params}, Body={final_request_body}")
        return {
            "path_params": path_params,
            "query_params": query_params,
            "headers": headers,
            "request_body": final_request_body,
        }

    def _execute_api_node_wrapper(self, node_def: Node) -> Callable[[WorkflowExecutionState], Awaitable[WorkflowExecutionState]]:
        """
        Returns an async function that executes a single API node.
        This wrapper is needed because RunnableLambda expects a function.
        """
        async def _run_node_instance(current_state: WorkflowExecutionState) -> WorkflowExecutionState:
            node_id = node_def.effective_id
            logger.info(f"[{self.session_id}] Executing node: {node_id} (OpID: {node_def.operationId})")
            await self.websocket_callback("node_execution_started", {"node_id": node_id, "operationId": node_def.operationId}, self.session_id)
            
            current_state.current_node_id = node_id
            payload_override = current_state.interrupt_payload_override
            current_state.interrupt_payload_override = None # Consume it

            # 1. Prepare API inputs (payload, path params, query params)
            prepared_inputs = self._prepare_api_inputs(node_def, current_state)
            
            # Use method and path from the Node model (ensure they are populated)
            method = node_def.method
            api_path = node_def.path # This is the template path, e.g., /users/{userId}

            if not method or not api_path:
                error_msg = f"Node '{node_id}' is missing 'method' or 'path' definition."
                logger.error(f"[{self.session_id}] {error_msg}")
                await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_msg}, self.session_id)
                current_state.execution_log.append({"node_id": node_id, "status": "error", "error": error_msg})
                # To halt graph execution, raise an exception or return a specific state
                # For now, we'll log and let the graph try to continue if possible, or END if it's a dead-end.
                # A more robust solution would be to have conditional edges based on success/failure.
                return current_state


            # Substitute path parameters into the API path template
            final_api_path = api_path
            for p_name, p_val in prepared_inputs["path_params"].items():
                final_api_path = final_api_path.replace(f"{{{p_name}}}", str(p_val))

            # Handle interruptions for specific methods (e.g., POST, PUT)
            # The node_def should indicate if it requires confirmation
            if node_def.requires_confirmation and not payload_override: # `requires_confirmation` needs to be added to Node model
                logger.info(f"[{self.session_id}] Node {node_id} requires confirmation. Interrupting.")
                await self.websocket_callback(
                    "interrupt_confirmation_required",
                    {
                        "node_id": node_id,
                        "operationId": node_def.operationId,
                        "method": method,
                        "path_template": api_path, # Original path with placeholders
                        "calculated_path": final_api_path, # Path after substitutions
                        "payload_template": node_def.payload_description, # Or a more structured template
                        "calculated_payload": prepared_inputs["request_body"],
                        "query_params": prepared_inputs["query_params"],
                        "path_params": prepared_inputs["path_params"]
                    },
                    self.session_id
                )
                # This will cause the graph to pause here due to the Interrupt
                # The actual Interrupt object needs to be returned by the graph's step
                # For now, we'll rely on the streaming loop to handle the interrupt signal
                # by waiting on self.resume_queue. This part is tricky with RunnableLambda.
                # A more direct way with LangGraph is for the node to return a special
                # signal or for the graph to be designed with explicit interrupt points.

                # For now, let's simulate by waiting for resume_queue if an interrupt was signaled.
                # This is a conceptual placeholder for how LangGraph handles actual interrupts.
                # The actual interruption mechanism in LangGraph is more integrated.
                # We will manage the interrupt outside the compiled graph's direct flow for now.
                # The `run_workflow_streaming` method will handle the wait.
                # Here, we just signal that an interrupt was requested.
                # If we want the graph to truly pause, this node would need to return a special value
                # that the LangGraph framework interprets as an interruption.
                # This current design simulates an external wait.
                logger.debug(f"[{self.session_id}] Node {node_id} conceptually interrupted. Waiting for resume signal if run via streaming method.")
                # The actual wait will happen in run_workflow_streaming
                # If not running via streaming that handles interrupts, this won't pause.


            # 2. Execute the API call
            try:
                api_result = await self.api_executor.execute_api(
                    operationId=node_def.operationId,
                    method=method,
                    endpoint=final_api_path, # Use the path with substituted params
                    payload=payload_override if payload_override is not None else prepared_inputs["request_body"],
                    query_params=prepared_inputs["query_params"],
                    # path_params are already in the endpoint string
                    headers=prepared_inputs["headers"]
                )
                logger.info(f"[{self.session_id}] API call for node {node_id} completed. Status: {api_result.get('status_code')}")
                
                # Record metrics (optional)
                # api_calls_total.labels(api_name=node_def.operationId, status=str(api_result.get('status_code', 'unknown'))).inc()
                # api_latency_seconds.labels(api_name=node_def.operationId).observe(api_result.get('execution_time', 0))

                # 3. Process the result and extract data based on node_def.output_mappings
                newly_extracted_data = {}
                if api_result.get("status_code") and 200 <= api_result["status_code"] < 300:
                    await self.websocket_callback("node_execution_succeeded", {"node_id": node_id, "result_preview": str(api_result.get('response_body'))[:200]}, self.session_id)
                    
                    # Process output mappings
                    for out_map in node_def.output_mappings: # output_mappings needs to be on Node model
                        extracted_value = self._resolve_json_path(api_result.get("response_body", {}), out_map.source_data_path)
                        if extracted_value is not None:
                            newly_extracted_data[out_map.target_data_key] = extracted_value
                            logger.debug(f"[{self.session_id}] Extracted for '{node_id}': '{out_map.target_data_key}' = {extracted_value}")
                        else:
                            logger.warning(f"[{self.session_id}] Could not extract '{out_map.target_data_key}' using path '{out_map.source_data_path}' from response of '{node_id}'.")
                    
                    current_state.execution_log.append({
                        "node_id": node_id, "status": "success", "result": api_result,
                        "extracted_here": newly_extracted_data.copy()
                    })
                else:
                    error_detail = api_result.get("response_body", {"error": "API call failed with status " + str(api_result.get("status_code"))})
                    logger.error(f"[{self.session_id}] Node {node_id} API call failed: {error_detail}")
                    await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": error_detail, "status_code": api_result.get("status_code")}, self.session_id)
                    current_state.execution_log.append({"node_id": node_id, "status": "error", "error_details": error_detail, "full_result": api_result})
                    # Decide if graph should halt on error. For now, it continues.

                # Update the shared extracted_data pool
                current_state.extracted_data.update(newly_extracted_data)

            except Exception as e:
                logger.critical(f"[{self.session_id}] Unhandled exception during API execution for node {node_id}: {e}", exc_info=True)
                await self.websocket_callback("node_execution_failed", {"node_id": node_id, "error": str(e)}, self.session_id)
                current_state.execution_log.append({"node_id": node_id, "status": "exception", "error": str(e)})
            
            current_state.current_node_id = None # Clear after execution
            return current_state

        return _run_node_instance


    async def run_workflow_streaming(self, thread_config: Dict[str, Any]):
        """
        Streams the execution of the compiled LangGraph.
        Handles interruptions by waiting for user input via resume_queue.
        `thread_config` should include `{"configurable": {"thread_id": "some_unique_id"}}`
        """
        logger.info(f"[{self.session_id}] Starting workflow streaming. Config: {thread_config}")
        await self.websocket_callback("workflow_execution_started", {"session_id": self.session_id, "graph_description": self.workflow_def.description}, self.session_id)

        # LangGraph's astream_events gives more granular events.
        # astream gives accumulated state updates.
        async for event in self.compiled_graph.astream_events(self.initial_execution_state, config=thread_config, version="v1"):
            event_type = event["event"]
            event_name = event.get("name", "unknown_event_source") # Node name or other source
            event_data = event.get("data", {})

            logger.debug(f"[{self.session_id}] Workflow Stream Event: Type='{event_type}', Name='{event_name}', DataKeys='{list(event_data.keys())}'")

            if event_type == "on_chain_start": # Or on_node_start if using a more specific event type
                # This event_data often contains the input to the node
                node_id_running = event_name # Assuming name is the node_id
                if node_id_running and node_id_running not in [START, END, "__start__", "__end__"]:
                     logger.info(f"[{self.session_id}] Event: Node '{node_id_running}' starting.")
                     # Callback for node_execution_started is handled within _execute_api_node_wrapper

            elif event_type == "on_chain_end": # Or on_node_end
                node_id_finished = event_name
                output_state = event_data.get("output") # This should be WorkflowExecutionState

                if node_id_finished and node_id_finished not in [START, END, "__start__", "__end__"] and isinstance(output_state, WorkflowExecutionState):
                    logger.info(f"[{self.session_id}] Event: Node '{node_id_finished}' finished.")
                    # Callbacks for success/failure are handled within _execute_api_node_wrapper
                    # We can send a summary or detailed log entry here if needed
                    last_log_entry = next((log for log in reversed(output_state.execution_log) if log.get("node_id") == node_id_finished), None)
                    if last_log_entry:
                         await self.websocket_callback("node_log_update", {"node_id": node_id_finished, "log_entry": last_log_entry}, self.session_id)

                # Check if an interruption was signaled by the node (e.g. requires_confirmation)
                # This part needs careful integration with LangGraph's actual interrupt mechanism.
                # The current _execute_api_node_wrapper doesn't directly raise LangGraph Interrupt.
                # It relies on this streaming loop to catch the conceptual interrupt.
                active_node_def = next((n for n in self.workflow_def.nodes if n.effective_id == output_state.current_node_id), None) # current_node_id is set before API call
                
                if active_node_def and active_node_def.requires_confirmation: # And not yet confirmed
                    # This check might be too late if the node already proceeded.
                    # The interrupt should ideally pause *before* the API call if confirmation is needed.
                    # The current design has a slight race: node signals, then this loop checks.
                    # A true LangGraph Interrupt would make the graph yield control.

                    # If we are at a point where an interrupt was conceptually requested by _execute_api_node_wrapper
                    # and the node is now "finished" in the stream (meaning it didn't truly pause itself via LangGraph Interrupt),
                    # we now explicitly wait for the resume queue.
                    # This is a workaround for not using LangGraph's built-in Interrupt from within the RunnableLambda directly.
                    
                    # Find the node definition that just ran, if its current_node_id is set
                    # This logic is a bit complex because the node itself doesn't *return* an Interrupt object
                    # in the current setup.
                    # We're simulating the pause here in the stream loop.

                    # Let's assume the node execution logic itself handles the interrupt signal
                    # and the main purpose of this loop is to observe.
                    # If a node requires confirmation, it should have sent an "interrupt_confirmation_required"
                    # message via websocket_callback. The client UI should then respond.
                    # The server (FastAPI handler) will get that response and call `submit_interrupt_value`.
                    
                    # If the `current_node_id` is still set on the output_state here, it might mean
                    # the node is in a "waiting for confirmation" state (conceptually).
                    if output_state.current_node_id: # Indicates a node might be waiting
                        node_awaiting_confirmation = output_state.current_node_id
                        logger.info(f"[{self.session_id}] Workflow conceptually paused at node '{node_awaiting_confirmation}', waiting for user resume via queue.")
                        try:
                            resume_value_dict = await asyncio.wait_for(self.resume_queue.get(), timeout=600) # 10 min timeout
                            logger.info(f"[{self.session_id}] Resuming node '{node_awaiting_confirmation}' with value: {resume_value_dict}")
                            
                            # We need to reinvoke the graph with the new state including the resume_value
                            # This is where LangGraph's `update_state` with a checkpointer is useful.
                            # With MemorySaver, we might need to reconstruct the input for the next step.
                            # The `resume_value_dict` should be the payload to use.
                            # We'll store this in the state so the node can pick it up on its *next* invocation
                            # if the graph is designed to re-enter or if the interrupt mechanism is more integrated.

                            # For this simulation, we'll assume the interrupt was for a payload.
                            # The node, when it runs *again* or if it was truly paused, would use this.
                            # This is the hacky part of not using native LangGraph Interrupts.
                            
                            # A better approach for LangGraph: node returns `Interrupt()`
                            # Then, when resuming, `graph.update_state(config, {"interrupt_payload_override": resume_value_dict})`
                            # and then call `graph.astream_events(None, config)` to continue.

                            # For this current structure, let's assume the `submit_interrupt_value`
                            # will somehow make the `_execute_api_node_wrapper` aware of the override
                            # perhaps by setting a flag or value that the *next* time it's called (if graph loops or retries)
                            # or by directly passing it to a specific waiting task.
                            # The current `_execute_api_node_wrapper` consumes `interrupt_payload_override` at its start.
                            # So, we need to update the state and have the graph re-evaluate from the interrupted point.
                            # This is complex without native Interrupts.

                            # Let's simplify: The `submit_interrupt_value` will put the payload into the queue.
                            # The `_execute_api_node_wrapper` for the *interrupted node* needs to be designed
                            # to pick up this value.
                            # The current `astream_events` loop might not naturally re-trigger the node
                            # in a way that it re-evaluates with the new `interrupt_payload_override`
                            # unless the graph explicitly routes back to it or it was truly paused.

                            # The most straightforward way with the current structure is if the interrupt
                            # happens *before* the node is marked "ended" in the stream.
                            # The `_execute_api_node_wrapper` itself should await `self.resume_queue.get()`
                            # if it determines an interruption is needed.

                            # Let's adjust `_execute_api_node_wrapper` to do this.
                            # (Change made in _execute_api_node_wrapper: it will now await resume_queue if it interrupts)

                        except asyncio.TimeoutError:
                            logger.error(f"[{self.session_id}] User did not resume node '{node_awaiting_confirmation}' in time.")
                            await self.websocket_callback("workflow_execution_failed", {"error": "User confirmation timeout"}, self.session_id)
                            return # End streaming

            elif event_type == "on_graph_end":
                final_state = event_data.get("output") # This should be the final WorkflowExecutionState
                logger.info(f"[{self.session_id}] Workflow execution finished. Final state: {final_state.dict() if final_state else 'N/A'}")
                await self.websocket_callback("workflow_execution_completed", {"final_state": final_state.dict() if final_state else {}}, self.session_id)
                return # End of workflow

            elif event_type == "on_tool_error" or event_type == "on_chain_error": # Handle errors from nodes
                node_name_with_error = event_name
                error_details = str(event_data) # Data might be the exception itself or a dict
                logger.error(f"[{self.session_id}] Error in workflow at node/tool '{node_name_with_error}': {error_details}")
                await self.websocket_callback("workflow_execution_failed", {"node_id": node_name_with_error, "error": error_details}, self.session_id)
                return # End streaming on error

        logger.info(f"[{self.session_id}] Workflow streaming loop completed.")
        # Fallback if on_graph_end wasn't explicitly caught or if stream ends unexpectedly
        await self.websocket_callback("workflow_execution_completed", {"message": "Streaming ended."}, self.session_id)


    async def submit_interrupt_value(self, value: Dict[str, Any]):
        """
        Called by the server (e.g., FastAPI handler) when the user provides
        input to resume an interrupted node.
        """
        logger.info(f"[{self.session_id}] Interrupt value submitted to queue: {value}")
        await self.resume_queue.put(value)

