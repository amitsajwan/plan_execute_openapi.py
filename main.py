# main.py
import logging
import uuid
import json
import os
import sys
import asyncio # Ensure asyncio is imported
from typing import Any, Dict, Optional, Tuple, List, Callable, Awaitable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Assuming models.py is in the same directory or accessible in PYTHONPATH
from models import BotState
# Assuming graph.py, llm_config.py, and workflow_executor.py are accessible
from graph import build_graph
from llm_config import initialize_llms
from pydantic import ValidationError # Keep for state validation

# Placeholder for APIExecutor - replace with your actual implementation
# Ensure workflow_executor.py defines APIExecutor and WorkflowExecutor
try:
    from workflow_executor import APIExecutor, WorkflowExecutor
except ImportError:
    logging.critical("main.py: Failed to import APIExecutor or WorkflowExecutor from workflow_executor.py.")
    # Define basic placeholders if import fails, to allow the application to start with warnings.
    class APIExecutor:
        def __init__(self, api_key: Optional[str] = None): # Example constructor
            logger.warning("Using placeholder APIExecutor due to import error.")
    class WorkflowExecutor:
        def __init__(self, *args, **kwargs):
            logger.warning("Using placeholder WorkflowExecutor due to import error.")
        async def run_workflow_streaming(self, *args, **kwargs):
            logger.warning("Placeholder WorkflowExecutor.run_workflow_streaming called.")
            await asyncio.sleep(0) # Minimal async operation
        async def submit_interrupt_value(self, *args, **kwargs):
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
langgraph_app: Optional[Any] = None # Will hold the compiled LangGraph app
api_executor_instance: Optional[APIExecutor] = None # Will hold the APIExecutor
checkpointer = MemorySaver() # Using in-memory checkpointer

@app.on_event("startup")
async def startup_event():
    global langgraph_app, api_executor_instance
    logger.info("FastAPI startup: Initializing LLMs, API Executor, and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        
        # Instantiate your APIExecutor here
        # Replace with your actual APIExecutor initialization logic
        # For example, it might take API keys or other config from environment variables
        api_key_from_env = os.getenv("SOME_API_KEY_FOR_EXECUTION") # Example
        api_executor_instance = APIExecutor() # Pass necessary config if any
        logger.info("APIExecutor instance created.")

        # Pass the api_executor_instance to build_graph
        langgraph_app = build_graph(
            router_llm_instance,
            worker_llm_instance,
            api_executor_instance, # Pass the instance here
            checkpointer
        )
        logger.info("LangGraph application built and compiled successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize/build graph on startup: {e}", exc_info=True)
        langgraph_app = None
        api_executor_instance = None # Ensure it's None if setup fails

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown.")
    # Clean up resources if necessary (e.g., close database connections for checkpointer)
    if hasattr(checkpointer, 'close'):
        try:
            # checkpointer.close() # If your checkpointer needs closing
            logger.info("Checkpointer closed (if applicable).")
        except Exception as e:
            logger.error(f"Error closing checkpointer: {e}")
    
    # Clean up schema cache from utils.py
    from utils import SCHEMA_CACHE
    if SCHEMA_CACHE and hasattr(SCHEMA_CACHE, 'close'):
        try:
            SCHEMA_CACHE.close()
            logger.info("Schema cache closed.")
        except Exception as e:
            logger.error(f"Error closing schema cache: {e}")


async def send_ws_message(websocket: WebSocket, msg_type: str, content: Any, session_id: str):
    """Helper to send WebSocket messages with logging."""
    try:
        await websocket.send_json({"type": msg_type, "content": content})
        logger.debug(f"[{session_id}] WS TX: Type='{msg_type}', Content='{str(content)[:100]}...'")
    except Exception as e:
        logger.error(f"[{session_id}] Error sending WebSocket message (Type: {msg_type}): {e}", exc_info=False)


@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4()) # Unique session ID for this connection
    logger.info(f"WebSocket connection accepted: {session_id}")

    if langgraph_app is None or api_executor_instance is None:
        await send_ws_message(websocket, "error", "Backend agent or API executor not initialized. Please try again later.", session_id)
        await websocket.close(code=1011) # Internal error
        return

    await send_ws_message(websocket, "info", {"session_id": session_id, "message": "Connected. Ready to process OpenAPI spec or queries."}, session_id)

    current_bot_state: Optional[BotState] = None

    try:
        while True:
            user_input_text = await websocket.receive_text()
            user_input_text = user_input_text.strip()
            logger.info(f"[{session_id}] WS RX: {user_input_text[:150]}...")

            if not user_input_text:
                await send_ws_message(websocket, "warning", "Received empty message. Please send some input.", session_id)
                continue

            await send_ws_message(websocket, "status", "Processing your request...", session_id)

            # LangGraph configuration for the current session/thread
            config = {"configurable": {"thread_id": session_id}}

            # Retrieve current state from checkpointer or initialize new state
            # This part is crucial for stateful conversation
            checkpoint = checkpointer.get(config)
            if checkpoint:
                try:
                    # LangGraph MemorySaver stores state under "channel_values"
                    # The actual BotState object might be nested if not careful with how state is passed/returned by nodes
                    raw_state_values = checkpoint.get("channel_values", checkpoint) # Fallback to root if no channel_values
                    
                    # Heuristic to find the actual BotState dictionary
                    # LangGraph nodes receive and return the whole state object.
                    # If the state is directly the BotState model, this should work.
                    if isinstance(raw_state_values, dict) and "session_id" in raw_state_values:
                        dict_to_validate = raw_state_values
                    elif isinstance(raw_state_values, dict) and raw_state_values: # If it's a dict but not BotState directly
                        # Try to find it if it's the value of the first key (common in some LangGraph patterns)
                        first_key = next(iter(raw_state_values), None)
                        if first_key and isinstance(raw_state_values[first_key], dict) and "session_id" in raw_state_values[first_key]:
                            dict_to_validate = raw_state_values[first_key]
                        else: # Assume raw_state_values is the BotState dict
                            dict_to_validate = raw_state_values
                    else: # If not a dict or doesn't look like BotState, start fresh
                        logger.warning(f"[{session_id}] Checkpoint data format unexpected. Starting with fresh state. Data: {str(raw_state_values)[:200]}")
                        dict_to_validate = {"session_id": session_id}

                    current_bot_state = BotState.model_validate(dict_to_validate)
                    current_bot_state.user_input = user_input_text # Update with new input
                    # Reset per-turn fields
                    current_bot_state.response = None
                    current_bot_state.final_response = ""
                    current_bot_state.next_step = None
                    current_bot_state.intent = None
                    # Clear graph_to_send from scratchpad for new turn; it's set by nodes if needed
                    if current_bot_state.scratchpad: current_bot_state.scratchpad.pop('graph_to_send', None)

                    logger.debug(f"[{session_id}] Loaded and updated state from checkpoint. Workflow status: {current_bot_state.workflow_execution_status}")

                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading/validating state from checkpoint: {e}. Data: {str(checkpoint)[:300]}. Starting fresh.", exc_info=False)
                    current_bot_state = BotState(session_id=session_id, user_input=user_input_text)
            else:
                logger.debug(f"[{session_id}] No checkpoint found. Initializing new BotState.")
                current_bot_state = BotState(session_id=session_id, user_input=user_input_text)


            # --- Main Agent Graph Execution ---
            final_main_agent_state_dict: Optional[Dict] = None
            try:
                async for event in langgraph_app.astream_events(current_bot_state.model_dump(), config=config, version="v1"):
                    event_type = event["event"]
                    event_name = event.get("name", "unknown_node")
                    event_data = event.get("data", {})

                    logger.debug(f"[{session_id}] MainAgentStream Event: Type='{event_type}', Name='{event_name}'")

                    if event_type == "on_chain_end" or event_type == "on_tool_end": # Or on_node_end
                        node_output_value = event_data.get("output")
                        
                        # Output from a node should be the full BotState dictionary or BotState object
                        if isinstance(node_output_value, BotState):
                            current_bot_state = node_output_value # Update current_bot_state with the output
                        elif isinstance(node_output_value, dict) and "session_id" in node_output_value:
                            try:
                                current_bot_state = BotState.model_validate(node_output_value)
                            except ValidationError as ve:
                                logger.error(f"[{session_id}] Pydantic validation error for node '{event_name}' output dict: {ve}. Data: {str(node_output_value)[:200]}", exc_info=False)
                                # Potentially handle error, for now, we might have stale current_bot_state
                        
                        if current_bot_state:
                            # Send intermediate responses from nodes
                            if current_bot_state.response: # Nodes set this for intermediate feedback
                                await send_ws_message(websocket, "intermediate", current_bot_state.response, session_id)
                            
                            # Send graph updates if available from scratchpad
                            graph_json_to_send = current_bot_state.scratchpad.pop('graph_to_send', None)
                            if graph_json_to_send:
                                try:
                                    await send_ws_message(websocket, "graph_update", json.loads(graph_json_to_send), session_id)
                                except json.JSONDecodeError:
                                    logger.error(f"[{session_id}] Failed to parse graph_json_to_send for graph_update from node '{event_name}'.")
                                    await send_ws_message(websocket, "error", "Internal error: Graph data format issue.", session_id)
                    
                    if event_type == "on_graph_end":
                        final_main_agent_state_dict = event_data.get("output") # This is the final state dict of the graph run
                        if final_main_agent_state_dict and isinstance(final_main_agent_state_dict, dict):
                             current_bot_state = BotState.model_validate(final_main_agent_state_dict) # Update one last time
                        break # Exit astream_events loop

                if not current_bot_state: # Should not happen if graph ran
                    raise RuntimeError("Main agent graph execution did not yield a final state.")

                # Send the final response from the main agent's turn
                if current_bot_state.final_response:
                    await send_ws_message(websocket, "final", current_bot_state.final_response, session_id)
                else: # Should be handled by responder, but as a fallback
                    logger.warning(f"[{session_id}] Main agent turn ended, but no final_response was set in BotState. Last intermediate: {current_bot_state.response}")
                    await send_ws_message(websocket, "info", current_bot_state.response or "Processing of your main query is complete.", session_id)

            except Exception as e_graph_agent:
                logger.critical(f"[{session_id}] Main agent LangGraph execution error: {e_graph_agent}", exc_info=True)
                await send_ws_message(websocket, "error", f"Error during main agent processing: {str(e_graph_agent)[:200]}", session_id)
                continue # Allow user to try again

            # --- Workflow Execution (if triggered by the main agent) ---
            if current_bot_state and current_bot_state.workflow_execution_status == "pending_start":
                logger.info(f"[{session_id}] Workflow execution status is 'pending_start'. Initiating workflow.")
                
                workflow_executor_instance = current_bot_state.scratchpad.get('workflow_executor_instance')
                if not workflow_executor_instance or not isinstance(workflow_executor_instance, WorkflowExecutor):
                    logger.error(f"[{session_id}] 'pending_start' but no WorkflowExecutor found in scratchpad.")
                    await send_ws_message(websocket, "error", "Error: Workflow setup failed internally (executor not found).", session_id)
                    current_bot_state.workflow_execution_status = "failed" # Update status
                    checkpointer.put(config, current_bot_state.model_dump()) # Save updated status
                    continue

                # Define the websocket_callback_adapter for this session's workflow execution
                async def wf_websocket_callback_adapter(event_type: str, data: Dict, wf_session_id: str):
                    # Prepend "workflow_" to event types to distinguish from main agent messages if needed
                    await send_ws_message(websocket, f"workflow_{event_type}", data, wf_session_id)

                workflow_executor_instance.websocket_callback = wf_websocket_callback_adapter
                
                logger.info(f"[{session_id}] Starting WorkflowExecutor.run_workflow_streaming task.")
                current_bot_state.workflow_execution_status = "running" # Update status before starting
                checkpointer.put(config, current_bot_state.model_dump()) # Save updated status

                # Run the workflow execution in a separate task
                asyncio.create_task(
                    workflow_executor_instance.run_workflow_streaming(thread_config=config)
                )
                # No immediate response here; workflow sends its own updates via callback.
                # The main loop continues to listen for new user messages.
                # If the user sends a message while workflow is running, it will be processed by the main agent.
                # This could be used to cancel, query status, or interact if the main agent supports it.

            # Note: If the user sends a message to resume a workflow,
            # the OpenAPIRouter -> interactive_query_planner -> core_logic.resume_workflow_with_payload
            # path will be taken. That logic in core_logic retrieves the executor from scratchpad
            # and calls its `submit_interrupt_value` method.

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e_outer:
        logger.critical(f"[{session_id}] Unhandled error in WebSocket handler: {e_outer}", exc_info=True)
        try:
            # Try to inform the client if possible
            await send_ws_message(websocket, "error", f"Critical server error: {str(e_outer)[:200]}", session_id)
        except:
            pass # Ignore if sending fails (connection might be already gone)
    finally:
        logger.info(f"[{session_id}] Closing WebSocket connection.")
        # Clean up session-specific data if necessary
        # e.g., if workflow_executor_instance needs explicit cleanup for this session
        if current_bot_state and current_bot_state.scratchpad:
            executor = current_bot_state.scratchpad.pop('workflow_executor_instance', None)
            if executor and hasattr(executor, 'cleanup'): # If your executor has a cleanup method
                # await executor.cleanup() # For example
                logger.info(f"[{session_id}] Cleaned up workflow executor instance from scratchpad.")
        try:
            await websocket.close()
        except:
            pass

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

