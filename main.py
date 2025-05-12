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

from models import BotState # Ensure this BotState has execution_graph field active
from graph import build_graph
from llm_config import initialize_llms
from pydantic import ValidationError

# Import the actual APIExecutor
try:
    from api_executor import APIExecutor
except ImportError:
    logger.critical("CRITICAL ERROR: api_executor.py not found or APIExecutor class cannot be imported.")
    # Define a very basic placeholder if the import fails, so the app might partially load for inspection.
    class APIExecutor: # type: ignore
        def __init__(self, *args, **kwargs):
            logger.error("FALLBACK APIExecutor from main.py being used. This is not intended for normal operation.")
        async def close(self): pass

# Import the WorkflowExecutor (ensure this points to your simplified version for now)
from workflow_executor import WorkflowExecutor


from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

langgraph_app: Optional[Any] = None
api_executor_instance: Optional[APIExecutor] = None # Will hold the real APIExecutor
checkpointer = MemorySaver()
active_workflow_executors: Dict[str, WorkflowExecutor] = {} # Manages simplified WorkflowExecutor instances

@app.on_event("startup")
async def startup_event():
    global langgraph_app, api_executor_instance
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        
        # Instantiate the real APIExecutor
        api_executor_instance = APIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL"),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        logger.info("Real APIExecutor instance created.")

        langgraph_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance, # Pass the real APIExecutor to the main graph
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

    if langgraph_app is None or api_executor_instance is None: # Check both
        await send_ws_message(websocket, "error", "Backend agent or API executor not initialized.", session_id)
        await websocket.close(code=1011); return

    await send_ws_message(websocket, "info", {"session_id": session_id, "message": "Connection established."}, session_id)
    current_bot_state_obj: Optional[BotState] = None

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX: {user_input_text[:100]}...")

            if not user_input_text: continue
            await send_ws_message(websocket, "status", "Processing...", session_id)
            config = {"configurable": {"thread_id": session_id}}

            checkpoint = checkpointer.get(config)
            if checkpoint:
                try:
                    raw_state_values = checkpoint.get("channel_values", checkpoint)
                    dict_to_validate = raw_state_values
                    if not (isinstance(raw_state_values, dict) and "session_id" in raw_state_values):
                        dict_to_validate = {"session_id": session_id}
                        logger.warning(f"[{session_id}] Checkpoint structure unexpected, re-initializing BotState partially.")
                    current_bot_state_obj = BotState.model_validate(dict_to_validate)
                    current_bot_state_obj.user_input = user_input_text
                    current_bot_state_obj.response = None; current_bot_state_obj.final_response = ""
                    current_bot_state_obj.next_step = None; current_bot_state_obj.intent = None
                    if current_bot_state_obj.scratchpad:
                        current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                        current_bot_state_obj.scratchpad.pop('workflow_executor_instance', None)
                        current_bot_state_obj.scratchpad.pop('pending_resume_payload', None)
                except Exception as e:
                    logger.warning(f"[{session_id}] Error loading BotState from checkpoint: {e}. Initializing new.", exc_info=False)
                    current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)
            else:
                current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)

            if current_bot_state_obj.scratchpad is None: current_bot_state_obj.scratchpad = {}
            
            logger.debug(f"[{session_id}] Scratchpad keys BEFORE main graph pop: {list(current_bot_state_obj.scratchpad.keys())}")
            if 'workflow_executor_instance' in current_bot_state_obj.scratchpad:
                removed_item_type = type(current_bot_state_obj.scratchpad.pop('workflow_executor_instance')).__name__
                logger.warning(f"[{session_id}] Popped 'workflow_executor_instance' (type: {removed_item_type}) from scratchpad before main graph. This is a safeguard.")
            logger.debug(f"[{session_id}] Scratchpad keys AFTER main graph pop: {list(current_bot_state_obj.scratchpad.keys())}")

            try:
                state_dump_for_graph = current_bot_state_obj.model_dump(exclude_none=True)
                async for event in langgraph_app.astream_events(state_dump_for_graph, config=config, version="v1"):
                    if event["event"] == "on_chain_end" or event["event"] == "on_tool_end":
                        node_output = event["data"].get("output")
                        if isinstance(node_output, BotState): current_bot_state_obj = node_output
                        elif isinstance(node_output, dict):
                            try: current_bot_state_obj = BotState.model_validate(node_output)
                            except Exception as e_val: logger.error(f"[{session_id}] Error validating node output into BotState: {e_val}")
                        
                        if current_bot_state_obj and current_bot_state_obj.response:
                             await send_ws_message(websocket, "intermediate", current_bot_state_obj.response, session_id)
                        if current_bot_state_obj and current_bot_state_obj.scratchpad and 'graph_to_send' in current_bot_state_obj.scratchpad:
                            graph_json = current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                            if graph_json: await send_ws_message(websocket, "graph_update", json.loads(graph_json), session_id)

                    if event["event"] == "on_graph_end":
                        final_output = event["data"].get("output")
                        if final_output and isinstance(final_output, dict):
                             current_bot_state_obj = BotState.model_validate(final_output)
                        break
                
                if not current_bot_state_obj:
                    logger.error(f"[{session_id}] CRITICAL: current_bot_state_obj is None after main graph stream."); continue
                
                if current_bot_state_obj.final_response:
                    await send_ws_message(websocket, "final", current_bot_state_obj.final_response, session_id)
                elif current_bot_state_obj.response:
                    await send_ws_message(websocket, "final", current_bot_state_obj.response, session_id)

            except Exception as e_main_agent:
                logger.critical(f"[{session_id}] Main agent graph error: {e_main_agent}", exc_info=True)
                await send_ws_message(websocket, "error", f"Agent error: {str(e_main_agent)[:150]}", session_id)
                continue

            if current_bot_state_obj:
                if current_bot_state_obj.workflow_execution_status == "pending_start":
                    logger.info(f"[{session_id}] Status 'pending_start'. Initiating WorkflowExecutor (Simplified).")
                    current_execution_graph = getattr(current_bot_state_obj, 'execution_graph', None)
                    if not current_execution_graph:
                        logger.error(f"[{session_id}] Cannot start workflow: execution_graph missing in BotState.")
                        await send_ws_message(websocket, "error", "Cannot start: No execution plan.", session_id)
                        current_bot_state_obj.workflow_execution_status = "failed"
                    else:
                        try:
                            # Using the simplified WorkflowExecutor which does not need api_executor_instance
                            wf_executor = WorkflowExecutor(
                                workflow_definition=current_execution_graph,
                                websocket_callback=None, 
                                session_id=session_id
                            )
                            active_workflow_executors[session_id] = wf_executor
                            async def wf_websocket_callback_adapter(event_type: str, data: Dict, wf_cb_session_id: str):
                                await send_ws_message(websocket, f"workflow_{event_type}", data, wf_cb_session_id)
                            wf_executor.websocket_callback = wf_websocket_callback_adapter
                            current_bot_state_obj.workflow_execution_status = "running"
                            logger.info(f"[{session_id}] Simplified WorkflowExecutor created. Starting run_workflow_streaming.")
                            asyncio.create_task(wf_executor.run_workflow_streaming(thread_config=config))
                            await send_ws_message(websocket, "info", "Simplified workflow execution started.", session_id)
                        except Exception as e_wf_create:
                            logger.error(f"[{session_id}] Failed to create/start Simplified WorkflowExecutor: {e_wf_create}", exc_info=True)
                            await send_ws_message(websocket, "error", f"Failed to start workflow: {str(e_wf_create)[:100]}", session_id)
                            current_bot_state_obj.workflow_execution_status = "failed"

                if current_bot_state_obj.scratchpad and 'pending_resume_payload' in current_bot_state_obj.scratchpad:
                    pending_payload = current_bot_state_obj.scratchpad.pop('pending_resume_payload')
                    logger.info(f"[{session_id}] Found 'pending_resume_payload': {str(pending_payload)[:100]}...")
                    executor_to_resume = active_workflow_executors.get(session_id)
                    if executor_to_resume:
                        logger.info(f"[{session_id}] Active WorkflowExecutor found. Submitting interrupt value.")
                        if current_bot_state_obj.workflow_execution_status == "paused_for_confirmation": # Check state before resuming
                            await executor_to_resume.submit_interrupt_value(pending_payload)
                            await send_ws_message(websocket, "info", "Resume payload submitted to workflow.", session_id)
                        else:
                            logger.warning(f"[{session_id}] Cannot resume: workflow status is '{current_bot_state_obj.workflow_execution_status}', not 'paused_for_confirmation'.")
                            await send_ws_message(websocket, "warning", "Cannot resume: Workflow not paused for confirmation.", session_id)
                    else:
                        logger.error(f"[{session_id}] 'pending_resume_payload' found, but NO active WorkflowExecutor for session.")
                        await send_ws_message(websocket, "error", "Cannot resume: No active workflow.", session_id)
                        current_bot_state_obj.workflow_execution_status = "failed"
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
    finally:
        logger.info(f"[{session_id}] Closing WebSocket connection.")
        active_workflow_executors.pop(session_id, None)
        # try:
        #     if websocket.client_state == WebSocketState.CONNECTED:
        #          await websocket.close()
        # except Exception: pass

@app.get("/", response_class=FileResponse)
async def get_index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("HTML Error: Main page not found.", status_code=404)
    return FileResponse(index_path)
