# main.py
import logging
import uuid
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple, List 

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse # Added FileResponse
from fastapi.staticfiles import StaticFiles # Added StaticFiles

from models import BotState 
from graph import build_graph 
from pydantic import ValidationError 
from llm_config import initialize_llms 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files directory
# Create a directory named 'static' in the same directory as main.py
# and place index.html, style.css, script.js into it.
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
    logger.info(f"Created static directory at {STATIC_DIR}")
    # You would then manually place your index.html, style.css, script.js here

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


langgraph_app: Optional[Any] = None
checkpointer = MemorySaver() 

@app.on_event("startup")
async def startup_event():
    global langgraph_app
    logger.info("FastAPI startup: Initializing LLMs and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms() 
        langgraph_app = build_graph(router_llm_instance, worker_llm_instance, checkpointer)
        logger.info("LangGraph application retrieved (compiled in build_graph).")
    except Exception as e:
        logger.critical(f"Failed to initialize/build graph on startup: {e}", exc_info=True)
        langgraph_app = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown.")
    if hasattr(checkpointer, 'close'):
        try: checkpointer.close(); logger.info("Checkpointer closed.") # type: ignore
        except Exception as e: logger.error(f"Error closing checkpointer: {e}")
    from utils import SCHEMA_CACHE 
    if SCHEMA_CACHE:
        try: SCHEMA_CACHE.close(); logger.info("Schema cache closed.")
        except Exception as e: logger.error(f"Error closing schema cache: {e}")

@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WS accepted: {session_id}")
    if langgraph_app is None:
        await websocket.send_json({"type": "error", "content": "Backend agent not initialized."})
        await websocket.close(code=1011); return
    await websocket.send_json({"type": "info", "session_id": session_id, "content": "Connected. Provide OpenAPI spec or ask."})

    try:
        while True:
            user_input = await websocket.receive_text(); user_input = user_input.strip()
            logger.info(f"[{session_id}] RX: {user_input[:100]}...")
            if not user_input: await websocket.send_json({"type": "warning", "content": "Empty msg."}); continue
            await websocket.send_json({"type": "status", "content": "Processing..."}) 

            config = {"configurable": {"thread_id": session_id}}
            current_checkpoint = checkpointer.get(config)
            initial_state_for_turn: BotState
            if current_checkpoint:
                try:
                    state_values = current_checkpoint.get("channel_values", {})
                    if not isinstance(state_values, dict) or not state_values:
                        state_values = current_checkpoint if isinstance(current_checkpoint, dict) else {}
                    dict_to_validate = {}
                    if "session_id" in state_values: 
                        dict_to_validate = state_values
                    elif state_values: 
                        first_key = next(iter(state_values), None)
                        if first_key and isinstance(state_values[first_key], dict) and "session_id" in state_values[first_key]:
                            dict_to_validate = state_values[first_key]
                        elif "__root__" in state_values and isinstance(state_values["__root__"], dict):
                             dict_to_validate = state_values["__root__"]
                        else: 
                            dict_to_validate = state_values 
                    initial_state_for_turn = BotState.model_validate(dict_to_validate)
                    initial_state_for_turn.user_input = user_input
                    initial_state_for_turn.response = None; initial_state_for_turn.final_response = ""
                    initial_state_for_turn.next_step = None; initial_state_for_turn.intent = None
                    logger.debug(f"[{session_id}] Loaded state from checkpoint for session: {initial_state_for_turn.session_id}")
                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading/validating state from checkpoint: {e}. Data: {str(current_checkpoint)[:300]}. Starting fresh.")
                    initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)
            else: 
                logger.debug(f"[{session_id}] No checkpoint. New state.")
                initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)

            current_turn_final_state_obj: Optional[BotState] = None 
            
            if initial_state_for_turn.scratchpad:
                 initial_state_for_turn.scratchpad.pop('graph_to_send', None)

            try:
                if langgraph_app is None: raise RuntimeError("Agent not available.")
                
                last_processed_state_obj: Optional[BotState] = None

                async for stream_event in langgraph_app.astream_events(initial_state_for_turn, config=config, version="v1"): 
                    event_type = stream_event["event"]
                    event_data = stream_event["data"]
                    event_name = stream_event.get("name", "unknown_node") 
                    
                    if event_type in ("on_chain_end", "on_tool_end"):
                        node_output_value = event_data.get("output")
                        
                        processed_node_state: Optional[BotState] = None
                        if isinstance(node_output_value, BotState):
                            processed_node_state = node_output_value
                        elif isinstance(node_output_value, dict) and "session_id" in node_output_value: 
                            try:
                                processed_node_state = BotState.model_validate(node_output_value)
                            except ValidationError as e_val:
                                logger.error(f"[{session_id}] Pydantic validation error for node '{event_name}' output dict: {e_val}. Data: {str(node_output_value)[:200]}", exc_info=False)
                        
                        if processed_node_state:
                            last_processed_state_obj = processed_node_state 

                            if processed_node_state.response:
                                logger.info(f"[{session_id}] Sending INTERMEDIATE from '{event_name}': {processed_node_state.response[:100]}...")
                                await websocket.send_json({"type": "intermediate", "content": processed_node_state.response})

                            graph_json_to_send = processed_node_state.scratchpad.pop('graph_to_send', None)
                            if graph_json_to_send:
                                logger.info(f"[{session_id}] Sending GRAPH_UPDATE from node '{event_name}'.")
                                try:
                                    await websocket.send_json({"type": "graph_update", "content": json.loads(graph_json_to_send)})
                                except json.JSONDecodeError:
                                    logger.error(f"[{session_id}] Failed to parse graph_json_to_send before sending graph_update.")
                                    await websocket.send_json({"type": "error", "content": "Internal error: Could not send graph update due to format issue."})
                
                current_turn_final_state_obj = last_processed_state_obj
                logger.info(f"[{session_id}] Stream finished. current_turn_final_state_obj available: {current_turn_final_state_obj is not None}")
                if current_turn_final_state_obj: 
                    logger.info(f"[{session_id}] Final state final_response: '{current_turn_final_state_obj.final_response}'")
                    logger.info(f"[{session_id}] Final state response (should be None): '{current_turn_final_state_obj.response}'")

                if current_turn_final_state_obj and current_turn_final_state_obj.final_response:
                    logger.info(f"[{session_id}] Sending FINAL: {current_turn_final_state_obj.final_response[:100]}...")
                    await websocket.send_json({"type": "final", "content": current_turn_final_state_obj.final_response})
                else: 
                    fallback_msg = "Processing complete, but no specific message was generated by the agent."
                    if current_turn_final_state_obj and current_turn_final_state_obj.response: 
                        fallback_msg = current_turn_final_state_obj.response 
                        logger.warning(f"[{session_id}] No final_response, using last intermediate response from snapshot: {fallback_msg[:100]}...")
                    elif initial_state_for_turn.response: 
                        fallback_msg = initial_state_for_turn.response
                        logger.warning(f"[{session_id}] No final_response, using response from initial state for turn: {fallback_msg[:100]}...")
                    else:
                        logger.error(f"[{session_id}] No final_response or any response in final/initial state snapshot for user.")
                    await websocket.send_json({"type": "error", "content": fallback_msg})

            except Exception as e_graph:
                logger.critical(f"[{session_id}] LangGraph execution error: {e_graph}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"Error during agent processing: {str(e_graph)[:200]}"})
    except WebSocketDisconnect: logger.info(f"WS disconnected: {session_id}")
    except Exception as e_outer:
        logger.critical(f"[{session_id}] WS handler error: {e_outer}", exc_info=True)
        try: await websocket.send_json({"type": "error", "content": f"Critical server error: {str(e_outer)[:200]}"})
        except: pass
    finally:
        logger.info(f"[{session_id}] Closing WS connection.")
        try: await websocket.close()
        except: pass

# Serve the main HTML page
@app.get("/", response_class=FileResponse)
async def get_index_page():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        logger.error(f"index.html not found at {index_path}. Make sure it's in the '{STATIC_DIR}' directory.")
        return HTMLResponse("<html><body><h1>Error: index.html not found</h1><p>Please ensure index.html, style.css, and script.js are in a 'static' subdirectory.</p></body></html>", status_code=404)
    return FileResponse(index_path)

