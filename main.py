# main.py
import logging
import uuid
import json
import os
import sys
import asyncio
import inspect
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from models import BotState
from graph import build_graph
from llm_config import initialize_llms
from pydantic import ValidationError

# Import the WorkflowExecutor (ensure this points to your simplified or full version)
from workflow_executor import WorkflowExecutor


# Placeholder for APIExecutor if not used by the simplified workflow in this context
class APIExecutor_Placeholder_Main:
    def __init__(self, *args, **kwargs):
        logger.warning("Using placeholder APIExecutor for main.py context.")
    async def close(self):
        logger.warning("Placeholder APIExecutor.close() called.")


from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Check imported WorkflowExecutor (from previous step, kept for sanity) ---
try:
    if WorkflowExecutor:
        init_signature = inspect.signature(WorkflowExecutor.__init__)
        # Simplified WorkflowExecutor __init__ expects: self, workflow_definition, websocket_callback, session_id
        expected_param_count_simplified = 4 
        if len(init_signature.parameters) < expected_param_count_simplified:
            logger.critical(
                "CRITICAL STARTUP WARNING: Imported 'WorkflowExecutor' appears to have an __init__ signature (%s) "
                "with fewer than %d parameters. This might be a placeholder or an incorrect version. "
                "Ensure the correct 'workflow_executor.py' is being imported.",
                str(init_signature), expected_param_count_simplified
            )
except Exception as e_inspect:
    logger.error(f"Error during inspection of WorkflowExecutor: {e_inspect}", exc_info=True)
# --- End Check ---

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logger.info(f"Created static directory at {STATIC_DIR}")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

langgraph_app: Optional[Any] = None
api_executor_instance: Optional[APIExecutor_Placeholder_Main] = None
checkpointer = MemorySaver()
active_workflow_executors: Dict[str, WorkflowExecutor] = {}

@app.on_event("startup")
async def startup_event():
    global langgraph_app, api_executor_instance
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        api_executor_instance = APIExecutor_Placeholder_Main() # Using placeholder for main context

        langgraph_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance, # Pass the placeholder or real one if main graph needs it
            checkpointer
        )
        logger.info("Main LangGraph agent application built and compiled successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize/build graph on startup: {e}", exc_info=True)
        langgraph_app = None
        if api_executor_instance and hasattr(api_executor_instance, 'close'):
            await api_executor_instance.close()
        api_executor_instance = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown initiated.")
    if api_executor_instance and hasattr(api_executor_instance, 'close'):
        try:
            await api_executor_instance.close()
            logger.info("APIExecutor_Placeholder_Main client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing APIExecutor_Placeholder_Main client: {e}", exc_info=True)

    from utils import SCHEMA_CACHE # Assuming SCHEMA_CACHE is in utils
    if SCHEMA_CACHE and hasattr(SCHEMA_CACHE, 'close'):
        try:
            SCHEMA_CACHE.close()
            logger.info("Schema cache closed.")
        except Exception as e:
            logger.error(f"Error closing schema cache: {e}")
    logger.info("FastAPI shutdown complete.")

async def send_ws_message(websocket: WebSocket, msg_type: str, content: Any, session_id: str):
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": msg_type, "content": content})
    except WebSocketDisconnect:
        logger.warning(f"[{session_id}] WS TX fail (Type: {msg_type}): Disconnected.")
    except Exception as e:
        logger.error(f"[{session_id}] WS TX error (Type: {msg_type}): {e}", exc_info=False)

@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket connection accepted: {session_id}")

    if langgraph_app is None:
        await send_ws_message(websocket, "error", "Backend agent not initialized.", session_id)
        await websocket.close(code=1011)
        return

    await send_ws_message(websocket, "info", {"session_id": session_id, "message": "Connection established."}, session_id)
    current_bot_state_obj: Optional[BotState] = None

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX: {user_input_text[:200]}...")

            if not user_input_text:
                await send_ws_message(websocket, "warning", "Empty message received.", session_id)
                continue

            await send_ws_message(websocket, "status", "Processing...", session_id)
            config = {"configurable": {"thread_id": session_id}}

            checkpoint = checkpointer.get(config)
            if checkpoint:
                try:
                    raw_state_values = checkpoint.get("channel_values", checkpoint)
                    dict_to_validate = raw_state_values
                    if not (isinstance(raw_state_values, dict) and "session_id" in raw_state_values):
                        found_state = False
                        if isinstance(raw_state_values, dict):
                            for key, value in raw_state_values.items():
                                if isinstance(value, dict) and "session_id" in value:
                                    dict_to_validate = value; found_state = True; break
                        if not found_state: dict_to_validate = {"session_id": session_id} # Fallback
                    
                    current_bot_state_obj = BotState.model_validate(dict_to_validate)
                    current_bot_state_obj.user_input = user_input_text # Update with new input
                    # Reset transient fields for the new turn
                    current_bot_state_obj.response = None
                    current_bot_state_obj.final_response = ""
                    current_bot_state_obj.next_step = None
                    current_bot_state_obj.intent = None
                    if current_bot_state_obj.scratchpad:
                        current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                        current_bot_state_obj.scratchpad.pop('workflow_executor_instance', None)
                        current_bot_state_obj.scratchpad.pop('pending_resume_payload', None)
                    logger.info(f"[{session_id}] Loaded BotState. WF Status: {current_bot_state_obj.workflow_execution_status}")
                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading BotState from checkpoint: {e}. Initializing new.", exc_info=False)
                    current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)
            else:
                logger.info(f"[{session_id}] No checkpoint. Initializing new BotState.")
                current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)

            # --- Critical check and cleanup of scratchpad before main graph execution ---
            if current_bot_state_obj.scratchpad is None:
                current_bot_state_obj.scratchpad = {} # Ensure scratchpad is a dict

            logger.info(f"[{session_id}] Scratchpad keys BEFORE explicit pop for astream_events: {list(current_bot_state_obj.scratchpad.keys())}")
            
            # Explicitly remove 'workflow_executor_instance' if it exists.
            # This is a safeguard. Ideally, it should not be in BotState processed by the main graph.
            if 'workflow_executor_instance' in current_bot_state_obj.scratchpad:
                removed_item = current_bot_state_obj.scratchpad.pop('workflow_executor_instance')
                logger.warning(
                    f"[{session_id}] Removed 'workflow_executor_instance' (type: {type(removed_item).__name__}) "
                    f"from scratchpad immediately before model_dump for astream_events. "
                    "This indicates it might have been added by a graph node or persisted incorrectly."
                )
            
            logger.info(f"[{session_id}] Scratchpad keys AFTER explicit pop for astream_events: {list(current_bot_state_obj.scratchpad.keys())}")
            # --- End critical check ---

            try:
                # Prepare data for astream_events
                state_dump_for_graph = current_bot_state_obj.model_dump(exclude_none=True)
                
                async for event in langgraph_app.astream_events(state_dump_for_graph, config=config, version="v1"):
                    event_type = event["event"]
                    event_name = event.get("name", "unknown_node")
                    event_data = event.get("data", {})
                    logger.debug(f"[{session_id}] MainAgentStream Event: Type='{event_type}', Name='{event_name}'")

                    if event_type == "on_chain_end" or event_type == "on_tool_end":
                        node_output_value = event_data.get("output")
                        if isinstance(node_output_value, BotState): # If node returns full state
                            current_bot_state_obj = node_output_value
                        elif isinstance(node_output_value, dict) and "session_id" in node_output_value: # If node returns dict
                            try:
                                current_bot_state_obj = BotState.model_validate(node_output_value)
                            except ValidationError as ve:
                                logger.error(f"[{session_id}] Pydantic validation error for node '{event_name}' output: {ve}", exc_info=False)
                        
                        # Process intermediate responses and graph updates from the potentially updated current_bot_state_obj
                        if current_bot_state_obj:
                            if current_bot_state_obj.response:
                                await send_ws_message(websocket, "intermediate", current_bot_state_obj.response, session_id)
                            
                            # Check scratchpad for graph_to_send (nodes might put it there)
                            # Must pop from the current_bot_state_obj that might have been updated by the node
                            graph_json_to_send = current_bot_state_obj.scratchpad.pop('graph_to_send', None) if current_bot_state_obj.scratchpad else None
                            if graph_json_to_send:
                                try:
                                    await send_ws_message(websocket, "graph_update", json.loads(graph_json_to_send), session_id)
                                except json.JSONDecodeError:
                                    logger.error(f"[{session_id}] Failed to parse graph_json_to_send for graph_update from '{event_name}'.")

                    if event_type == "on_graph_end":
                        final_output = event_data.get("output")
                        if final_output and isinstance(final_output, dict):
                             current_bot_state_obj = BotState.model_validate(final_output)
                        break # Exit astream_events loop

                if not current_bot_state_obj:
                    logger.error(f"[{session_id}] Main agent graph execution completed but current_bot_state_obj is not set.")
                    await send_ws_message(websocket, "error", "Critical error: Agent state lost.", session_id)
                    continue

                if current_bot_state_obj.final_response:
                    await send_ws_message(websocket, "final", current_bot_state_obj.final_response, session_id)
                else: # Should ideally always have a final_response from the 'responder' node
                    await send_ws_message(websocket, "info", current_bot_state_obj.response or "Processing complete.", session_id)

            except Exception as e_main_agent:
                logger.critical(f"[{session_id}] Main agent LangGraph execution error: {e_main_agent}", exc_info=True)
                await send_ws_message(websocket, "error", f"Error during main agent processing: {str(e_main_agent)[:200]}", session_id)
                # No checkpointer.put here as it was commented out by user
                continue # To the next user message

            # --- Post Main Agent Graph: Handle Workflow Start or Resume ---
            if current_bot_state_obj: # Ensure current_bot_state_obj is valid
                if current_bot_state_obj.workflow_execution_status == "pending_start":
                    logger.info(f"[{session_id}] Workflow status 'pending_start'. Initiating WorkflowExecutor.")
                    # The execution_graph should be in current_bot_state_obj if models.py is not simplified
                    # If models.py still has execution_graph commented out, this will be None.
                    current_execution_graph = getattr(current_bot_state_obj, 'execution_graph', None)

                    if not current_execution_graph: # Check the potentially simplified state
                        logger.error(f"[{session_id}] Cannot start workflow: execution_graph is missing in BotState (possibly due to simplified model).")
                        await send_ws_message(websocket, "error", "Cannot start workflow: No execution plan found in state.", session_id)
                        current_bot_state_obj.workflow_execution_status = "failed"
                    else:
                        try:
                            wf_executor = WorkflowExecutor(
                                workflow_definition=current_execution_graph, # Use the graph from state
                                websocket_callback=None, 
                                session_id=session_id
                            )
                            active_workflow_executors[session_id] = wf_executor

                            async def wf_websocket_callback_adapter(event_type: str, data: Dict, wf_cb_session_id: str):
                                await send_ws_message(websocket, f"workflow_{event_type}", data, wf_cb_session_id)
                            wf_executor.websocket_callback = wf_websocket_callback_adapter

                            current_bot_state_obj.workflow_execution_status = "running"
                            logger.info(f"[{session_id}] WorkflowExecutor created. Starting run_workflow_streaming task.")
                            asyncio.create_task(wf_executor.run_workflow_streaming(thread_config=config))
                            await send_ws_message(websocket, "info", "Workflow execution process started.", session_id)
                        except TypeError as te: # Catch specific TypeError for instantiation
                            logger.error(f"[{session_id}] TypeError creating WorkflowExecutor: {te}. This often means the wrong class definition is being used (e.g., a placeholder).", exc_info=True)
                            await send_ws_message(websocket, "error", f"Failed to start workflow (TypeError): {str(te)[:100]}", session_id)
                            current_bot_state_obj.workflow_execution_status = "failed"
                        except Exception as e_wf_create:
                            logger.error(f"[{session_id}] Failed to create/start WorkflowExecutor: {e_wf_create}", exc_info=True)
                            await send_ws_message(websocket, "error", f"Failed to start workflow: {str(e_wf_create)[:100]}", session_id)
                            current_bot_state_obj.workflow_execution_status = "failed"

                pending_resume_payload = current_bot_state_obj.scratchpad.pop('pending_resume_payload', None) if current_bot_state_obj.scratchpad else None
                if pending_resume_payload is not None:
                    logger.info(f"[{session_id}] Found 'pending_resume_payload'.")
                    executor_to_resume = active_workflow_executors.get(session_id)
                    if executor_to_resume and isinstance(executor_to_resume, WorkflowExecutor):
                        # This assumes your WorkflowExecutor (even simplified) has submit_interrupt_value
                        await executor_to_resume.submit_interrupt_value(pending_resume_payload)
                        await send_ws_message(websocket, "info", "Interrupt payload submitted to workflow.", session_id)
                    else:
                         await send_ws_message(websocket, "warning", "Pending resume payload found, but no active executor.", session_id)
                
                # checkpointer.put(...) was commented out by user.
                # logger.debug(f"[{session_id}] BotState (not saved by checkpointer.put) after post-agent. WF Status: {current_bot_state_obj.workflow_execution_status}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await send_ws_message(websocket, "error", f"Critical server error: {str(e_outer_loop)[:200]}", session_id)
        except: pass
    finally:
        logger.info(f"[{session_id}] Closing WebSocket connection.")
        if session_id in active_workflow_executors:
            active_workflow_executors.pop(session_id, None) # Use pop with default
            logger.info(f"[{session_id}] Removed WorkflowExecutor instance.")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.close()
        except Exception as e_close:
            logger.warning(f"[{session_id}] Error during WebSocket close: {e_close}", exc_info=False)

@app.get("/", response_class=FileResponse)
async def get_index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"FATAL: index.html not found at {index_path}.")
        return HTMLResponse("HTML Error: Main page not found.", status_code=404)
    return FileResponse(index_path)
    
