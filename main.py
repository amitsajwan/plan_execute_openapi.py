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
        async def run_workflow_streaming(self, *args, **kwargs):
            logger.warning("Placeholder WorkflowExecutor.run_workflow_streaming called.")
            await asyncio.sleep(0)
        async def submit_interrupt_value(self, *args, **kwargs): # Corrected method name
            logger.warning("Placeholder WorkflowExecutor.submit_interrupt_value called.")
            await asyncio.sleep(0)

# LangGraph checkpointer
from langgraph.checkpoint.memory import MemorySaver # Using MemorySaver for simplicity

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Consider logging.DEBUG for development
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


# Global variables for the LangGraph app and API Executor
langgraph_app: Optional[Any] = None # Will hold the compiled LangGraph app for the main agent
api_executor_instance: Optional[APIExecutor] = None # Will hold the APIExecutor instance
checkpointer = MemorySaver() # Using in-memory checkpointer

@app.on_event("startup")
async def startup_event():
    global langgraph_app, api_executor_instance
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        
        # Instantiate your APIExecutor here
        # Example: api_executor_instance = APIExecutor(base_url="https://api.example.com/v1", default_headers={"X-API-Version": "1.0"})
        # For now, using default constructor. Configure as needed from env vars or config files.
        api_executor_instance = APIExecutor(
            base_url=os.getenv("DEFAULT_API_BASE_URL"), # Example: load from environment
            timeout=float(os.getenv("API_TIMEOUT", 30.0))
        )
        logger.info("APIExecutor instance created.")

        # Pass the api_executor_instance to build_graph
        langgraph_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance, # Pass the instance here
            checkpointer
        )
        logger.info("Main LangGraph agent application built and compiled successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize/build graph on startup: {e}", exc_info=True)
        langgraph_app = None
        if api_executor_instance and hasattr(api_executor_instance, 'close'): # Close if partially initialized
            await api_executor_instance.close()
        api_executor_instance = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown initiated.")
    # Clean up resources
    if api_executor_instance and hasattr(api_executor_instance, 'close'):
        try:
            await api_executor_instance.close()
            logger.info("APIExecutor client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing APIExecutor client: {e}", exc_info=True)
            
    if hasattr(checkpointer, 'close'): # If checkpointer has a close method
        try:
            # await checkpointer.close() # Uncomment if your checkpointer needs async close
            logger.info("Checkpointer closed (if applicable).")
        except Exception as e:
            logger.error(f"Error closing checkpointer: {e}")
    
    from utils import SCHEMA_CACHE # Assuming utils.py and SCHEMA_CACHE exist
    if SCHEMA_CACHE and hasattr(SCHEMA_CACHE, 'close'):
        try:
            SCHEMA_CACHE.close()
            logger.info("Schema cache closed.")
        except Exception as e:
            logger.error(f"Error closing schema cache: {e}")
    logger.info("FastAPI shutdown complete.")


async def send_ws_message(websocket: WebSocket, msg_type: str, content: Any, session_id: str):
    """Helper to send WebSocket messages with logging."""
    try:
        await websocket.send_json({"type": msg_type, "content": content})
        # Limit log preview for potentially large content like graph JSON
        log_content_preview = str(content)
        if len(log_content_preview) > 150:
            log_content_preview = log_content_preview[:150] + "..."
        logger.debug(f"[{session_id}] WS TX: Type='{msg_type}', Content='{log_content_preview}'")
    except WebSocketDisconnect:
        logger.warning(f"[{session_id}] Failed to send WebSocket message (Type: {msg_type}): Connection already closed.")
    except Exception as e:
        logger.error(f"[{session_id}] Error sending WebSocket message (Type: {msg_type}): {e}", exc_info=False)


@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4()) # Unique session ID for this WebSocket connection
    logger.info(f"WebSocket connection accepted: {session_id}")

    # Ensure backend components are ready
    if langgraph_app is None or api_executor_instance is None:
        await send_ws_message(websocket, "error", "Backend agent or API executor not initialized. Please try reconnecting shortly.", session_id)
        await websocket.close(code=1011) # Internal error
        return

    await send_ws_message(websocket, "info", {"session_id": session_id, "message": "Connection established. Ready for your OpenAPI spec or query."}, session_id)

    current_bot_state_obj: Optional[BotState] = None # Holds the BotState object for the current turn

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX User Input: {user_input_text[:200]}...") # Log more of user input

            if not user_input_text:
                await send_ws_message(websocket, "warning", "Received empty message. Please provide an OpenAPI spec or ask a question.", session_id)
                continue

            await send_ws_message(websocket, "status", "Processing your request...", session_id)

            # LangGraph configuration for the current session/thread
            config = {"configurable": {"thread_id": session_id}}

            # --- State Loading/Initialization for the main agent graph ---
            checkpoint = checkpointer.get(config) # Get checkpoint for the session
            if checkpoint:
                try:
                    # Attempt to load state from the checkpoint
                    raw_state_values = checkpoint.get("channel_values", checkpoint)
                    dict_to_validate = raw_state_values
                    # Add more robust checkpoint structure detection if needed
                    if not (isinstance(raw_state_values, dict) and "session_id" in raw_state_values):
                        logger.warning(f"[{session_id}] Checkpoint data format for main agent was unexpected. Trying to adapt or starting fresh. Data: {str(raw_state_values)[:200]}")
                        # Attempt to find BotState if nested, otherwise reset
                        found_state = False
                        if isinstance(raw_state_values, dict):
                            for key, value in raw_state_values.items():
                                if isinstance(value, dict) and "session_id" in value:
                                    dict_to_validate = value
                                    found_state = True
                                    break
                        if not found_state:
                            dict_to_validate = {"session_id": session_id} # Fallback to minimal new state

                    current_bot_state_obj = BotState.model_validate(dict_to_validate)
                    current_bot_state_obj.user_input = user_input_text
                    # Reset per-turn fields that should not persist across user messages for the main agent
                    current_bot_state_obj.response = None
                    current_bot_state_obj.final_response = ""
                    current_bot_state_obj.next_step = None
                    current_bot_state_obj.intent = None
                    if current_bot_state_obj.scratchpad: current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                    logger.info(f"[{session_id}] Loaded and updated BotState from checkpoint. Workflow status: {current_bot_state_obj.workflow_execution_status}")
                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading/validating BotState from checkpoint: {e}. Data: {str(checkpoint)[:300]}. Initializing new state.", exc_info=False)
                    current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)
            else:
                logger.info(f"[{session_id}] No checkpoint found for main agent. Initializing new BotState.")
                current_bot_state_obj = BotState(session_id=session_id, user_input=user_input_text)

            # --- Main Agent Graph Execution (for parsing, planning, etc.) ---
            final_main_agent_state_dict: Optional[Dict] = None
            try:
                # Stream events from the main LangGraph agent
                async for event in langgraph_app.astream_events(current_bot_state_obj.model_dump(), config=config, version="v1"):
                    event_type = event["event"]
                    event_name = event.get("name", "unknown_node") # Node that emitted the event
                    event_data = event.get("data", {})

                    logger.debug(f"[{session_id}] MainAgentStream Event: Type='{event_type}', Name='{event_name}'")

                    if event_type == "on_chain_end" or event_type == "on_tool_end": # After a node finishes
                        node_output_value = event_data.get("output")
                        
                        if isinstance(node_output_value, BotState): # If node returns full BotState object
                            current_bot_state_obj = node_output_value
                        elif isinstance(node_output_value, dict) and "session_id" in node_output_value: # If node returns dict
                            try:
                                current_bot_state_obj = BotState.model_validate(node_output_value)
                            except ValidationError as ve:
                                logger.error(f"[{session_id}] Pydantic validation error for node '{event_name}' output dict: {ve}. Data: {str(node_output_value)[:200]}", exc_info=False)
                        else:
                            logger.warning(f"[{session_id}] Node '{event_name}' output was not a BotState object or recognizable dict. State may be stale. Output: {str(node_output_value)[:200]}")
                        
                        if current_bot_state_obj:
                            if current_bot_state_obj.response: # Send intermediate responses
                                await send_ws_message(websocket, "intermediate", current_bot_state_obj.response, session_id)
                            
                            graph_json_to_send = current_bot_state_obj.scratchpad.pop('graph_to_send', None)
                            if graph_json_to_send:
                                try:
                                    await send_ws_message(websocket, "graph_update", json.loads(graph_json_to_send), session_id)
                                except json.JSONDecodeError:
                                    logger.error(f"[{session_id}] Failed to parse graph_json_to_send for graph_update from node '{event_name}'.")
                    
                    if event_type == "on_graph_end":
                        final_main_agent_state_dict = event_data.get("output")
                        if final_main_agent_state_dict and isinstance(final_main_agent_state_dict, dict):
                             current_bot_state_obj = BotState.model_validate(final_main_agent_state_dict)
                        break # Exit main agent's astream_events loop

                if not current_bot_state_obj: # Should be set if graph ran
                    logger.error(f"[{session_id}] Main agent graph execution completed but current_bot_state_obj is not set.")
                    await send_ws_message(websocket, "error", "Critical error: Agent state lost after processing.", session_id)
                    continue

                # Send the final response from the main agent's turn (e.g., graph description, query answer)
                if current_bot_state_obj.final_response:
                    await send_ws_message(websocket, "final", current_bot_state_obj.final_response, session_id)
                else:
                    logger.warning(f"[{session_id}] Main agent turn ended, but no final_response was set. Last intermediate: {current_bot_state_obj.response}")
                    await send_ws_message(websocket, "info", current_bot_state_obj.response or "Main query processing complete.", session_id)

            except Exception as e_main_agent:
                logger.critical(f"[{session_id}] Main agent LangGraph execution error: {e_main_agent}", exc_info=True)
                await send_ws_message(websocket, "error", f"Error during main agent processing: {str(e_main_agent)[:200]}", session_id)
                # Update state to reflect error before potentially continuing or saving checkpoint
                if current_bot_state_obj:
                    current_bot_state_obj.response = f"Error: {str(e_main_agent)[:150]}"
                    current_bot_state_obj.final_response = current_bot_state_obj.response
                    current_bot_state_obj.next_step = "responder" # Ensure it goes to responder
                # checkpointer.put(config, current_bot_state_obj.model_dump() if current_bot_state_obj else {"error_state": True})
                continue # Allow user to try again with a new message

            # --- Workflow Execution Logic (if triggered by the main agent) ---
            if current_bot_state_obj and current_bot_state_obj.workflow_execution_status == "pending_start":
                logger.info(f"[{session_id}] Workflow execution status is 'pending_start'. Attempting to initiate workflow execution.")
                
                workflow_executor_instance = current_bot_state_obj.scratchpad.get('workflow_executor_instance')
                
                if not workflow_executor_instance or not isinstance(workflow_executor_instance, WorkflowExecutor):
                    logger.error(f"[{session_id}] 'pending_start' status but no valid WorkflowExecutor instance found in scratchpad.")
                    await send_ws_message(websocket, "error", "Error: Workflow setup failed internally (executor instance missing or invalid).", session_id)
                    current_bot_state_obj.workflow_execution_status = "failed"
                else:
                    # Define the websocket_callback_adapter for this specific session's workflow execution
                    # This adapter captures the current 'websocket' and 'session_id' in its closure
                    async def wf_websocket_callback_adapter(event_type: str, data: Dict, wf_cb_session_id: str): # wf_cb_session_id is from executor
                        # Prepend "workflow_" to event types to distinguish from main agent messages in frontend if needed
                        await send_ws_message(websocket, f"workflow_{event_type}", data, wf_cb_session_id) # Use session_id from executor for its logs

                    workflow_executor_instance.websocket_callback = wf_websocket_callback_adapter
                    
                    logger.info(f"[{session_id}] Starting WorkflowExecutor.run_workflow_streaming task for session.")
                    current_bot_state_obj.workflow_execution_status = "running" # Update status

                    # Run the workflow execution in a separate asyncio task so it doesn't block the WebSocket loop
                    # The `config` for the workflow executor's LangGraph instance uses the same session_id for its checkpointer
                    asyncio.create_task(
                        workflow_executor_instance.run_workflow_streaming(thread_config=config)
                    )
                    await send_ws_message(websocket, "info", "Workflow execution started. You will receive updates.", session_id)
            
            # Save the final state of BotState after the main agent's turn and potential workflow start
            if current_bot_state_obj:
                checkpointer.put(config, current_bot_state_obj.model_dump())
                logger.debug(f"[{session_id}] BotState saved to checkpoint after main agent turn. Workflow status: {current_bot_state_obj.workflow_execution_status}")


    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}. Client closed connection.")
    except Exception as e_outer_loop:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket main loop: {e_outer_loop}", exc_info=True)
        try:
            await send_ws_message(websocket, "error", f"Critical server error encountered: {str(e_outer_loop)[:200]}", session_id)
        except: # If sending fails, connection is likely already broken
            pass
    finally:
        logger.info(f"[{session_id}] Closing WebSocket connection and cleaning up session resources.")
        # Perform any session-specific cleanup if needed
        if current_bot_state_obj and current_bot_state_obj.scratchpad:
            executor = current_bot_state_obj.scratchpad.pop('workflow_executor_instance', None)
            if executor and hasattr(executor, 'cleanup_session_resources'): # Example cleanup method
                # await executor.cleanup_session_resources()
                logger.info(f"[{session_id}] Called cleanup for workflow executor instance from scratchpad.")
        try:
            if websocket.client_state != httpx. लेकिन_CLIENT_STATE.CLOSED: # Check if websocket is not already closed
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

