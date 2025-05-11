# workflow_executor.py (Simplified for Debugging RLock)
import asyncio
import logging
import json
from typing import Any, Callable, Awaitable, Dict, Optional, List

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # Using MemorySaver for simplicity
from langchain_core.runnables import RunnableLambda

from models import GraphOutput, Node # Assuming these are your Pydantic models for graph definition

logger = logging.getLogger(__name__)

# --- Simplified State ---
class SimplifiedWorkflowState(BaseModel):
    """A very simplified state for debugging workflow execution."""
    node_visits: List[str] = Field(default_factory=list)
    last_visited_node: Optional[str] = None
    error_message: Optional[str] = None

    class Config:
        extra = 'allow'


class WorkflowExecutor:
    def __init__(
        self,
        workflow_definition: GraphOutput,
        # api_executor_instance: APIExecutor, # Not used in simplified version
        websocket_callback: Callable[[str, Dict, str], Awaitable[None]], # Keep for notifications
        session_id: str,
        # initial_extracted_data: Optional[Dict[str, Any]] = None # Not used
    ):
        if not workflow_definition or not workflow_definition.nodes:
            raise ValueError("Workflow definition (GraphOutput) is empty or has no nodes.")

        self.workflow_def = workflow_definition
        # self.api_executor = api_executor_instance # Not used
        self.websocket_callback = websocket_callback
        self.session_id = session_id
        # self.resume_queue: asyncio.Queue = asyncio.Queue() # Not used in simplified version

        builder = StateGraph(SimplifiedWorkflowState)

        for node_def in self.workflow_def.nodes:
            if node_def.effective_id.upper() not in ["START_NODE", "END_NODE"]:
                node_runner_instance = self._simplified_node_runner_wrapper(node_def)
                builder.add_node(node_def.effective_id, node_runner_instance)

        for edge_def in self.workflow_def.edges:
            if edge_def.from_node.upper() == "START_NODE":
                builder.add_edge(START, edge_def.to_node)
            elif edge_def.to_node.upper() == "END_NODE":
                builder.add_edge(edge_def.from_node, END)
            else:
                builder.add_edge(edge_def.from_node, edge_def.to_node)

        # Using MemorySaver, debug=True for more logs
        self.compiled_graph = builder.compile(checkpointer=MemorySaver(), debug=True)
        logger.info(f"[{self.session_id}] Simplified WorkflowExecutor: Graph compiled with {len(self.workflow_def.nodes)} nodes.")
        self.initial_execution_state = SimplifiedWorkflowState()

    def _simplified_node_runner_wrapper(self, node_def: Node) -> Callable[[SimplifiedWorkflowState], Awaitable[SimplifiedWorkflowState]]:
        """
        Creates a simplified async runner for a node.
        It logs visitation and appends to a list in the state.
        """
        async def _run_node_instance(current_state: SimplifiedWorkflowState) -> SimplifiedWorkflowState:
            node_id = node_def.effective_id
            logger.info(f"[{self.session_id}] Simplified Node Execution: Visiting '{node_id}' (OpID: {node_def.operationId})")
            
            # Simulate some async work
            await asyncio.sleep(0.05) 

            # Update state
            new_node_visits = current_state.node_visits + [node_id]
            
            if self.websocket_callback:
                try:
                    await self.websocket_callback(
                        "node_execution_succeeded", # Simplified event
                        {"node_id": node_id, "message": f"Successfully visited {node_id}"},
                        self.session_id
                    )
                except Exception as e_cb:
                    logger.error(f"[{self.session_id}] Error in websocket_callback during simplified node {node_id}: {e_cb}")


            # Return a new state object or update in place if mutable (Pydantic models are better immutable)
            return SimplifiedWorkflowState(
                node_visits=new_node_visits,
                last_visited_node=node_id,
                error_message=current_state.error_message # Preserve error if any
            )
        return _run_node_instance

    async def run_workflow_streaming(self, thread_config: Dict[str, Any]):
        logger.info(f"[{self.session_id}] Starting SIMPLIFIED workflow streaming. Config: {thread_config}")
        if self.websocket_callback:
            await self.websocket_callback("workflow_execution_started", {"session_id": self.session_id, "message": "Simplified workflow started."}, self.session_id)

        final_state_to_return = self.initial_execution_state
        try:
            # Use astream_events to get more granular updates
            async for event in self.compiled_graph.astream_events(self.initial_execution_state, config=thread_config, version="v1"):
                event_type = event["event"]
                event_name = event.get("name", "unknown_event_source") # Node name or other source
                event_data = event.get("data", {})

                logger.debug(f"[{self.session_id}] Simplified WF Stream Event: Type='{event_type}', Name='{event_name}', DataKeys='{list(event_data.keys())}'")

                # Update local tracking of the state based on events
                if event_type == "on_chain_end": # A node has finished
                    output_state_candidate = event_data.get("output")
                    if isinstance(output_state_candidate, SimplifiedWorkflowState):
                        final_state_to_return = output_state_candidate
                        logger.info(f"[{self.session_id}] Node '{event_name}' finished. Last visited: {final_state_to_return.last_visited_node}. Total visits: {len(final_state_to_return.node_visits)}")

                elif event_type == "on_graph_end":
                    final_output_data = event_data.get("output")
                    if isinstance(final_output_data, SimplifiedWorkflowState):
                        final_state_to_return = final_output_data
                    
                    logger.info(f"[{self.session_id}] Simplified workflow execution finished (on_graph_end). Final state: {final_state_to_return.dict() if final_state_to_return else 'N/A'}")
                    if self.websocket_callback:
                        await self.websocket_callback("workflow_execution_completed", {"final_state": final_state_to_return.dict() if final_state_to_return else {}}, self.session_id)
                    return

                elif event_type in ["on_tool_error", "on_chain_error", "on_node_error"]:
                    error_details_str = str(event_data.get("output", event_data))
                    logger.error(f"[{self.session_id}] Error event '{event_type}' in simplified workflow at '{event_name}': {error_details_str[:500]}")
                    final_state_to_return.error_message = f"Error at {event_name}: {error_details_str[:100]}" # Update state with error
                    if self.websocket_callback:
                        await self.websocket_callback("workflow_execution_failed", {"node_id": event_name, "error_event_type": event_type, "error": error_details_str[:1000]}, self.session_id)
                    return
        
        except asyncio.CancelledError:
            logger.warning(f"[{self.session_id}] Simplified workflow streaming task was cancelled.")
            if self.websocket_callback:
                await self.websocket_callback("workflow_execution_failed", {"error": "Simplified workflow execution was cancelled."}, self.session_id)
            return
        except Exception as e_stream:
            error_str_stream = str(e_stream)
            logger.critical(f"[{self.session_id}] Unhandled exception during simplified workflow streaming: {error_str_stream}", exc_info=True)
            if self.websocket_callback:
                await self.websocket_callback("workflow_execution_failed", {"error": f"Critical simplified streaming error: {error_str_stream[:500]}"}, self.session_id)
            return

        # Fallback if loop finishes without on_graph_end
        logger.info(f"[{self.session_id}] Simplified workflow streaming loop completed (fallback).")
        if self.websocket_callback:
            await self.websocket_callback("workflow_execution_completed", {"message": "Simplified streaming ended (fallback).", "final_state": final_state_to_return.dict() if final_state_to_return else {}}, self.session_id)

    async def submit_interrupt_value(self, value: Dict[str, Any]):
        # This method is not used in the simplified version as interrupt handling is removed.
        logger.warning(f"[{self.session_id}] submit_interrupt_value called on simplified executor, but it has no effect.")
        pass

