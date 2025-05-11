# main.py
import logging
import uuid
import json
import os
import sys
import asyncio
from typing import Any, Dict, Optional, Callable, Awaitable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from models import BotState
from graph import build_graph
from llm_config import initialize_llms
from pydantic import ValidationError

# Import the simplified WorkflowExecutor
# Ensure this path is correct based on your project structure.
# If workflow_executor.py is in the same directory:
from workflow_executor import WorkflowExecutor
# If it's workflow_executor_py_simplified.py, adjust the import:
# from workflow_executor_py_simplified import WorkflowExecutor


# Placeholder for APIExecutor if it's still imported elsewhere but not used by simplified workflow
class APIExecutor:
    def __init__(self, *args, **kwargs):
        logger.warning("Using placeholder APIExecutor (not used by simplified workflow).")
    async def close(self):
        logger.warning("Placeholder APIExecutor.close() called.")


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
    logger.info(f"Created static directory at {STATIC_DIR}")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

langgraph_app: Optional[Any] = None
api_executor_instance: Optional[APIExecutor] = None # Kept for main graph, but not passed to simplified workflow
checkpointer = MemorySaver()
active_workflow_executors: Dict[str, WorkflowExecutor] = {}

@app.on_event("startup")
async def startup_event():
    global langgraph_app, api_executor_instance
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()

        # api_executor_instance is still needed for the main graph if it uses it.
        # If your main graph (via core_logic) also doesn't need it when using simplified workflow,
        # you might adjust its initialization too. For now, keeping it for the main graph.
        api_executor_instance = APIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL"),
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        logger.info("APIExecutor instance created (for main graph if needed).")

        langgraph_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance, # Main graph might still use the full APIExecutor
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

    from utils import SCHEMA_CACHE
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

    if langgraph_app is None: # api_executor_instance might not be critical if simplified workflow doesn't use it
        await send_ws_message(websocket, "error", "Backend agent not initialized. Please try reconnecting shortly.", session_id)
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
                await send_ws_message(websocket, "warning", "Received empty message.", session_id)
                continue

            await send_ws_message(websocket, "status", "Processing your request...", session_id)
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
                        if not found_state: dict_to_validate = {"session_id": session_id}
                    current_bot_state_obj = BotState.model_validate(dict_to_validate)
                    current_bot_state_obj.user_input = user_input_text
                    current_bot_state_obj.response = None; current_bot_state_obj.final_response = ""
                    current_bot_state_obj.next_step = None; current_bot_state_obj.intent = None
                    if current_bot_state_obj.scratchpad:
                        current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                        current_bot_state_obj.scratchpad.pop('workflow_executor_instance', None)
                        current_bot_state_obj.scratchpad.pop('pending_resume_payload', None)
                    logger.info(f"[{session_id}] Loaded BotState. Workflow status: {current_bot_state_obj.workflow_execution_status}")
                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading BotState from checkpoint: {e}. Initializing new state.", exc_info=False)
                    current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)
            else:
                logger.info(f"[{session_id}] No checkpoint. Initializing new BotState.")
                current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)

            if current_bot_state_obj.scratchpad:
                 current_bot_state_obj.scratchpad.pop('workflow_executor_instance', None)

            try:
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
                # User commented out the checkpointer.put call.
                # if current_bot_state_obj:
                #     current_bot_state_obj.response = f"Error: {str(e_main_agent)[:150]}"; current_bot_state_obj.final_response = current_bot_state_obj.response
                #     # checkpointer.put(config, current_bot_state_obj.model_dump(exclude_none=True))
                continue

            if current_bot_state_obj:
                if current_bot_state_obj.workflow_execution_status == "pending_start":
                    logger.info(f"[{session_id}] Workflow status 'pending_start'. Initiating SIMPLIFIED WorkflowExecutor.")
                    if not current_bot_state_obj.execution_graph:
                        logger.error(f"[{session_id}] Cannot start workflow: execution_graph is missing in BotState.")
                        await send_ws_message(websocket, "error", "Cannot start workflow: No execution plan found.", session_id)
                        current_bot_state_obj.workflow_execution_status = "failed"
                    else:
                        try:
                            # CORRECTED Instantiation for Simplified WorkflowExecutor
                            wf_executor = WorkflowExecutor(
                                workflow_definition=current_bot_state_obj.execution_graph,
                                # api_executor_instance=api_executor_instance, # REMOVED for simplified version
                                websocket_callback=None, # Will be set by adapter below
                                session_id=session_id
                                # initial_extracted_data=current_bot_state_obj.workflow_extracted_data.copy() # REMOVED for simplified version
                            )
                            active_workflow_executors[session_id] = wf_executor

                            async def wf_websocket_callback_adapter(event_type: str, data: Dict, wf_cb_session_id: str):
                                # Ensure the original websocket object (from the outer scope) is used
                                await send_ws_message(websocket, f"workflow_{event_type}", data, wf_cb_session_id)

                            wf_executor.websocket_callback = wf_websocket_callback_adapter

                            current_bot_state_obj.workflow_execution_status = "running"
                            logger.info(f"[{session_id}] Simplified WorkflowExecutor created. Starting run_workflow_streaming task.")
                            asyncio.create_task(wf_executor.run_workflow_streaming(thread_config=config))
                            await send_ws_message(websocket, "info", "Simplified workflow execution process started.", session_id)
                        except Exception as e_wf_create:
                            logger.error(f"[{session_id}] Failed to create/start Simplified WorkflowExecutor: {e_wf_create}", exc_info=True)
                            await send_ws_message(websocket, "error", f"Failed to start simplified workflow: {str(e_wf_create)[:100]}", session_id)
                            current_bot_state_obj.workflow_execution_status = "failed"

                pending_resume_payload = current_bot_state_obj.scratchpad.pop('pending_resume_payload', None)
                if pending_resume_payload is not None:
                    # Interrupt handling was removed from simplified executor, so this path is less relevant now
                    # but we keep the log for awareness.
                    logger.info(f"[{session_id}] Found 'pending_resume_payload'. Simplified executor doesn't use it directly via queue.")
                    executor_to_resume = active_workflow_executors.get(session_id)
                    if executor_to_resume and isinstance(executor_to_resume, WorkflowExecutor):
                        # The simplified executor's submit_interrupt_value does nothing,
                        # so this call won't actually resume via the old queue mechanism.
                        # This part of the logic would need to be re-thought if interrupts
                        # are re-introduced to the simplified executor.
                        await executor_to_resume.submit_interrupt_value(pending_resume_payload)
                        await send_ws_message(websocket, "info", "Interrupt payload noted (simplified workflow may not use queue).", session_id)
                    else:
                         await send_ws_message(websocket, "warning", "Pending resume payload found, but no active simplified executor.", session_id)


                # logger.debug(f"[{session_id}] BotState (not explicitly saved by checkpointer.put here) after post-agent processing. Workflow status: {current_bot_state_obj.workflow_execution_status}")
                # The checkpointer.put call was previously commented out by user.

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}. Client closed connection.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await send_ws_message(websocket, "error", f"Critical server error encountered: {str(e_outer_loop)[:200]}", session_id)
        except: pass
    finally:
        logger.info(f"[{session_id}] Closing WebSocket connection and cleaning up session resources.")
        if session_id in active_workflow_executors:
            active_workflow_executors.pop(session_id)
            logger.info(f"[{session_id}] Removed WorkflowExecutor instance from active pool.")
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
        return HTMLResponse("HTML Error", status_code=404)
    return FileResponse(index_path)
