# main.py
import logging
import uuid
import json
import os
import sys
import asyncio # Ensure asyncio is imported
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Assuming models.py is in the same directory or accessible in PYTHONPATH
from models import BotState
# Assuming graph.py, llm_config.py, and workflow_executor.py are accessible
from graph import build_graph # build_graph now expects api_executor_instance
from llm_config import initialize_llms
from pydantic import ValidationError

# Import APIExecutor and WorkflowExecutor
try:
    from api_executor import APIExecutor # Assuming api_executor.py is in the same directory
    from workflow_executor import WorkflowExecutor
except ImportError as e:
    logging.critical(f"main.py: Failed to import APIExecutor or WorkflowExecutor: {e}. Ensure api_executor.py and workflow_executor.py are correct and accessible.")
    # Define basic placeholders if import fails, to allow the application to start with warnings.
    class APIExecutor:
        def __init__(self, *args, **kwargs): 
            logger.warning("Using placeholder APIExecutor due to import error.")
        async def close(self): logger.warning("Placeholder APIExecutor.close() called.")
    class WorkflowExecutor:
        def __init__(self, *args, **kwargs):
            logger.warning("Using placeholder WorkflowExecutor due to import error.")
            self.websocket_callback = None # Ensure placeholder has this attribute
        async def run_workflow_streaming(self, *args, **kwargs):
            logger.warning("Placeholder WorkflowExecutor.run_workflow_streaming called.")
            if self.websocket_callback: # Simulate some callback if placeholder
                await self.websocket_callback("execution_completed", {"message": "Placeholder execution finished"}, "placeholder_session")
            await asyncio.sleep(0)
        async def submit_interrupt_value(self, *args, **kwargs):
            logger.warning("Placeholder WorkflowExecutor.submit_interrupt_value called.")
            await asyncio.sleep(0)

# LangGraph checkpointer
from langgraph.checkpoint.memory import MemorySaver # Using MemorySaver for simplicity

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Static files setup
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logger.info(f"Created static directory at {STATIC_DIR}")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Global variables
langgraph_app: Optional[Any] = None
api_executor_instance: Optional[APIExecutor] = None
checkpointer = MemorySaver()
# New global dictionary to manage active WorkflowExecutor instances per session_id
active_workflow_executors: Dict[str, WorkflowExecutor] = {}


@app.on_event("startup")
async def startup_event():
    global langgraph_app, api_executor_instance
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        
        api_executor_instance = APIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL"),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        logger.info("APIExecutor instance created.")

        langgraph_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance,
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
            logger.info("APIExecutor client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing APIExecutor client: {e}", exc_info=True)
            
    # ... (other shutdown logic for checkpointer, SCHEMA_CACHE) ...
    from utils import SCHEMA_CACHE
    if SCHEMA_CACHE and hasattr(SCHEMA_CACHE, 'close'):
        try: SCHEMA_CACHE.close(); logger.info("Schema cache closed.")
        except Exception as e: logger.error(f"Error closing schema cache: {e}")
    logger.info("FastAPI shutdown complete.")


async def send_ws_message(websocket: WebSocket, msg_type: str, content: Any, session_id: str):
    """Helper to send WebSocket messages with logging."""
    try:
        if websocket.client_state == httpx. लेकिन_CLIENT_STATE.OPEN: # Check if socket is open
            await websocket.send_json({"type": msg_type, "content": content})
            log_content_preview = str(content)
            if len(log_content_preview) > 150: log_content_preview = log_content_preview[:150] + "..."
            logger.debug(f"[{session_id}] WS TX: Type='{msg_type}', Content='{log_content_preview}'")
    except WebSocketDisconnect:
        logger.warning(f"[{session_id}] Failed to send WebSocket message (Type: {msg_type}): Connection already closed.")
    except Exception as e:
        logger.error(f"[{session_id}] Error sending WebSocket message (Type: {msg_type}): {e}", exc_info=False)


@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket connection accepted: {session_id}")

    if langgraph_app is None or api_executor_instance is None:
        await send_ws_message(websocket, "error", "Backend agent or API executor not initialized. Please try reconnecting shortly.", session_id)
        await websocket.close(code=1011)
        return

    await send_ws_message(websocket, "info", {"session_id": session_id, "message": "Connection established. Ready for your OpenAPI spec or query."}, session_id)

    current_bot_state_obj: Optional[BotState] = None

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX User Input: {user_input_text[:200]}...")

            if not user_input_text:
                await send_ws_message(websocket, "warning", "Received empty message. Please provide an OpenAPI spec or ask a question.", session_id)
                continue

            await send_ws_message(websocket, "status", "Processing your request...", session_id)
            config = {"configurable": {"thread_id": session_id}}

            # --- State Loading/Initialization ---
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
                        if not found_state: dict_to_validate = {"session_id": session_id}
                    current_bot_state_obj = BotState.model_validate(dict_to_validate)
                    current_bot_state_obj.user_input = user_input_text
                    current_bot_state_obj.response = None; current_bot_state_obj.final_response = ""
                    current_bot_state_obj.next_step = None; current_bot_state_obj.intent = None
                    if current_bot_state_obj.scratchpad:
                        current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                        # Ensure executor instance is NOT loaded from scratchpad into main agent state
                        current_bot_state_obj.scratchpad.pop('workflow_executor_instance', None)
                        # Clear pending resume payload from scratchpad at start of new turn
                        current_bot_state_obj.scratchpad.pop('pending_resume_payload', None)

                    logger.info(f"[{session_id}] Loaded BotState. Workflow status: {current_bot_state_obj.workflow_execution_status}")
                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading BotState from checkpoint: {e}. Initializing new state.", exc_info=False)
                    current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)
            else:
                logger.info(f"[{session_id}] No checkpoint. Initializing new BotState.")
                current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)
            
            # Crucially, ensure scratchpad for the state going into langgraph_app is clean of complex objects
            # that core_logic might have put there in a previous partial run if an error occurred.
            # The 'workflow_executor_instance' should not be in the state passed to the main agent graph.
            if current_bot_state_obj.scratchpad:
                 current_bot_state_obj.scratchpad.pop('workflow_executor_instance', None)


            # --- Main Agent Graph Execution ---
            try:
                # Pass a clean model_dump to astream_events
                async for event in langgraph_app.astream_events(current_bot_state_obj.model_dump(exclude_none=True), config=config, version="v1"):
                    event_type = event["event"]
                    event_name = event.get("name", "unknown_node")
                    event_data = event.get("data", {})
                    logger.debug(f"[{session_id}] MainAgentStream Event: Type='{event_type}', Name='{event_name}'")

                    if event_type == "on_chain_end" or event_type == "on_tool_end":
                        node_output_value = event_data.get("output")
                        if isinstance(node_output_value, BotState):
                            current_bot_state_obj = node_output_value
                        elif isinstance(node_output_value, dict) and "session_id" in node_output_value:
                            try: current_bot_state_obj = BotState.model_validate(node_output_value)
                            except ValidationError as ve: logger.error(f"[{session_id}] Pydantic validation error for node '{event_name}' output: {ve}", exc_info=False)
                        
                        if current_bot_state_obj:
                            if current_bot_state_obj.response:
                                await send_ws_message(websocket, "intermediate", current_bot_state_obj.response, session_id)
                            graph_json_to_send = current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                            if graph_json_to_send:
                                try: await send_ws_message(websocket, "graph_update", json.loads(graph_json_to_send), session_id)
                                except json.JSONDecodeError: logger.error(f"[{session_id}] Failed to parse graph_json_to_send for graph_update from '{event_name}'.")
                    
                    if event_type == "on_graph_end":
                        final_output = event_data.get("output")
                        if final_output and isinstance(final_output, dict):
                             current_bot_state_obj = BotState.model_validate(final_output)
                        break
                
                if not current_bot_state_obj:
                    logger.error(f"[{session_id}] Main agent graph execution completed but current_bot_state_obj is not set.")
                    await send_ws_message(websocket, "error", "Critical error: Agent state lost after processing.", session_id)
                    continue

                if current_bot_state_obj.final_response:
                    await send_ws_message(websocket, "final", current_bot_state_obj.final_response, session_id)
                else:
                    await send_ws_message(websocket, "info", current_bot_state_obj.response or "Main query processing complete.", session_id)

            except Exception as e_main_agent:
                logger.critical(f"[{session_id}] Main agent LangGraph execution error: {e_main_agent}", exc_info=True)
                await send_ws_message(websocket, "error", f"Error during main agent processing: {str(e_main_agent)[:200]}", session_id)
                if current_bot_state_obj: # Try to save a valid state even on error
                    current_bot_state_obj.response = f"Error: {str(e_main_agent)[:150]}"; current_bot_state_obj.final_response = current_bot_state_obj.response
                    checkpointer.put(config, current_bot_state_obj.model_dump(exclude_none=True))
                continue

            # --- Post Main Agent Graph: Handle Workflow Start or Resume ---
            if current_bot_state_obj:
                # 1. Check if a workflow needs to be started
                if current_bot_state_obj.workflow_execution_status == "pending_start":
                    logger.info(f"[{session_id}] Workflow status 'pending_start'. Initiating WorkflowExecutor.")
                    if not current_bot_state_obj.execution_graph:
                        logger.error(f"[{session_id}] Cannot start workflow: execution_graph is missing in BotState.")
                        await send_ws_message(websocket, "error", "Cannot start workflow: No execution plan found.", session_id)
                        current_bot_state_obj.workflow_execution_status = "failed"
                    else:
                        try:
                            # Create and store the WorkflowExecutor instance outside of BotState
                            wf_executor = WorkflowExecutor(
                                workflow_definition=current_bot_state_obj.execution_graph,
                                api_executor_instance=api_executor_instance, # Global instance
                                websocket_callback=None, # Will be set by adapter
                                session_id=session_id,
                                initial_extracted_data=current_bot_state_obj.workflow_extracted_data.copy()
                            )
                            active_workflow_executors[session_id] = wf_executor
                            
                            async def wf_websocket_callback_adapter(event_type: str, data: Dict, wf_cb_session_id: str):
                                await send_ws_message(websocket, f"workflow_{event_type}", data, wf_cb_session_id)
                            
                            wf_executor.websocket_callback = wf_websocket_callback_adapter
                            
                            current_bot_state_obj.workflow_execution_status = "running"
                            logger.info(f"[{session_id}] WorkflowExecutor created and stored. Starting run_workflow_streaming task.")
                            asyncio.create_task(wf_executor.run_workflow_streaming(thread_config=config))
                            await send_ws_message(websocket, "info", "Workflow execution process started.", session_id)
                        except Exception as e_wf_create:
                            logger.error(f"[{session_id}] Failed to create/start WorkflowExecutor: {e_wf_create}", exc_info=True)
                            await send_ws_message(websocket, "error", f"Failed to start workflow: {str(e_wf_create)[:100]}", session_id)
                            current_bot_state_obj.workflow_execution_status = "failed"

                # 2. Check if a workflow needs to be resumed
                pending_resume_payload = current_bot_state_obj.scratchpad.pop('pending_resume_payload', None)
                if pending_resume_payload is not None:
                    logger.info(f"[{session_id}] Found 'pending_resume_payload' in scratchpad. Attempting to resume workflow.")
                    executor_to_resume = active_workflow_executors.get(session_id)
                    if executor_to_resume and isinstance(executor_to_resume, WorkflowExecutor):
                        if current_bot_state_obj.workflow_execution_status == "paused_for_confirmation":
                            try:
                                await executor_to_resume.submit_interrupt_value(pending_resume_payload)
                                current_bot_state_obj.workflow_execution_status = "running" # It will become running
                                await send_ws_message(websocket, "info", "Workflow resumption signal sent.", session_id)
                                logger.info(f"[{session_id}] Submitted interrupt value to executor. Workflow should resume.")
                            except Exception as e_resume:
                                logger.error(f"[{session_id}] Error calling submit_interrupt_value on executor: {e_resume}", exc_info=True)
                                await send_ws_message(websocket, "error", f"Failed to send resume signal: {str(e_resume)[:100]}", session_id)
                                # Status might remain paused or become failed depending on executor's internal state
                        else:
                            logger.warning(f"[{session_id}] 'pending_resume_payload' found, but workflow status is '{current_bot_state_obj.workflow_execution_status}', not 'paused_for_confirmation'.")
                            await send_ws_message(websocket, "warning", "Cannot resume: Workflow is not currently paused for confirmation.", session_id)
                    else:
                        logger.error(f"[{session_id}] 'pending_resume_payload' found, but no active WorkflowExecutor for session.")
                        await send_ws_message(websocket, "error", "Cannot resume: No active workflow found for this session.", session_id)
                        current_bot_state_obj.workflow_execution_status = "failed" # Mark as failed if no executor

                # Save state after potential workflow start/resume attempt
                checkpointer.put(config, current_bot_state_obj.model_dump(exclude_none=True))
                logger.debug(f"[{session_id}] BotState saved after post-agent processing. Workflow status: {current_bot_state_obj.workflow_execution_status}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}. Client closed connection.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
        try:
            if websocket.client_state == httpx. लेकिन_CLIENT_STATE.OPEN:
                await send_ws_message(websocket, "error", f"Critical server error encountered: {str(e_outer_loop)[:200]}", session_id)
        except: pass
    finally:
        logger.info(f"[{session_id}] Closing WebSocket connection and cleaning up session resources.")
        # Remove executor for this session if it exists
        if session_id in active_workflow_executors:
            removed_executor = active_workflow_executors.pop(session_id)
            logger.info(f"[{session_id}] Removed WorkflowExecutor instance from active pool.")
            if hasattr(removed_executor, 'cleanup_session_resources'): # Example
                # await removed_executor.cleanup_session_resources()
                pass
        try:
            if websocket.client_state == httpx. लेकिन_CLIENT_STATE.OPEN: # Check before closing
                 await websocket.close()
        except Exception as e_close:
            logger.warning(f"[{session_id}] Error during WebSocket close: {e_close}", exc_info=False)


# Serve the main HTML page (frontend)
@app.get("/", response_class=FileResponse)
async def get_index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"FATAL: index.html not found at {index_path}. Ensure it's in the '{STATIC_DIR}' directory relative to main.py.")
        return HTMLResponse(
            "<html><body><h1>Error 404: Main page not found</h1><p>Please ensure index.html, style.css, and script.js are in a 'static' subdirectory where the server is running.</p></body></html>",
            status_code=404
        )
    return FileResponse(index_path)

