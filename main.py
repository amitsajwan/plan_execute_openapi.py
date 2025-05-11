# main.py
import logging
import uuid
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple, List 

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse 
from langgraph.checkpoint.memory import MemorySaver 

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

            current_turn_final_state: Optional[BotState] = None 
            
            if initial_state_for_turn.scratchpad:
                 initial_state_for_turn.scratchpad.pop('graph_to_send', None)

            try:
                if langgraph_app is None: raise RuntimeError("Agent not available.")
                
                # This will store the BotState object from the last relevant event
                last_processed_state_obj: Optional[BotState] = None

                async for stream_event in langgraph_app.astream_events(initial_state_for_turn, config=config, version="v1"): 
                    event_type, event_data, event_name = stream_event["event"], stream_event["data"], stream_event.get("name", "unknown_node")
                    
                    if event_type in ("on_chain_end", "on_tool_end"): # These events usually contain the full state output of the node
                        node_output_dict = event_data.get("output") 
                        
                        if isinstance(node_output_dict, dict) and "session_id" in node_output_dict: # Check if it's a flat BotState dict
                            try:
                                node_state = BotState.model_validate(node_output_dict)
                                last_processed_state_obj = node_state # Update with the latest full state
                                
                                if node_state.response: 
                                    logger.debug(f"[{session_id}] Node '{event_name}' intermediate: {node_state.response[:100]}...")
                                    await websocket.send_json({"type": "intermediate", "content": node_state.response})

                                graph_json_to_send = node_state.scratchpad.pop('graph_to_send', None)
                                if graph_json_to_send:
                                    logger.info(f"[{session_id}] Sending graph_update from node '{event_name}'.")
                                    await websocket.send_json({"type": "graph_update", "content": json.loads(graph_json_to_send)}) 
                            except ValidationError as e_val: 
                                logger.error(f"[{session_id}] Pydantic state validation error for node '{event_name}': {e_val}. Data tried: {str(node_output_dict)[:300]}", exc_info=False)
                            except Exception as e_proc: 
                                logger.error(f"[{session_id}] Error processing output of node '{event_name}': {e_proc}. Data: {str(node_output_dict)[:300]}", exc_info=True)
                        elif node_output_dict is not None: 
                             logger.warning(f"[{session_id}] Node '{event_name}' output was not None and not a direct BotState dict. Type: {type(node_output_dict)}, Output: {str(node_output_dict)[:300]}")
                
                # After the stream, current_turn_final_state should be the state after the responder has run.
                # We use the last_processed_state_obj which should be the state from the 'responder' node.
                current_turn_final_state = last_processed_state_obj

                logger.info(f"[{session_id}] Stream finished. current_turn_final_state available: {current_turn_final_state is not None}")
                if current_turn_final_state: 
                    logger.info(f"[{session_id}] current_turn_final_state.final_response: '{current_turn_final_state.final_response}'")
                    logger.info(f"[{session_id}] current_turn_final_state.response (after responder): '{current_turn_final_state.response}'") # Should be None

                if current_turn_final_state and current_turn_final_state.final_response:
                    logger.info(f"[{session_id}] Sending final: {current_turn_final_state.final_response[:100]}...")
                    await websocket.send_json({"type": "final", "content": current_turn_final_state.final_response})
                else: 
                    fallback_msg = "Processing complete, but no specific message was generated by the agent."
                    # Check initial_state_for_turn.response if current_turn_final_state is None or has no response
                    # This can happen if an error occurred very early, before any node properly set the state.
                    if initial_state_for_turn.response and not (current_turn_final_state and current_turn_final_state.final_response):
                        fallback_msg = initial_state_for_turn.response
                        logger.warning(f"[{session_id}] No final_response from graph, using response from initial state for turn: {fallback_msg[:100]}...")
                    elif current_turn_final_state and current_turn_final_state.response : # Should be None after responder
                        fallback_msg = current_turn_final_state.response 
                        logger.warning(f"[{session_id}] No final_response, using last intermediate response from snapshot: {fallback_msg[:100]}...")
                    else:
                        logger.error(f"[{session_id}] No final_response or any response in final/initial state snapshot for user.")
                    await websocket.send_json({"type": "error", "content": fallback_msg})

            except Exception as e_graph:
                logger.critical(f"[{session_id}] LangGraph error: {e_graph}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"Error: {str(e_graph)[:200]}"})
    except WebSocketDisconnect: logger.info(f"WS disconnected: {session_id}")
    except Exception as e_outer:
        logger.critical(f"[{session_id}] WS handler error: {e_outer}", exc_info=True)
        try: await websocket.send_json({"type": "error", "content": f"Server error: {str(e_outer)[:200]}"})
        except: pass
    finally:
        logger.info(f"[{session_id}] Closing WS connection.")
        try: await websocket.close()
        except: pass

HTML_TEST_PAGE = """
<!DOCTYPE html>
<html>
<head><title>OpenAPI Agent (Gemini)</title><style>body{font-family:sans-serif;margin:0;background-color:#f0f2f5;color:#333;display:flex;flex-direction:column;height:100vh}.header{background-color:#4A90E2;color:#fff;padding:15px 20px;text-align:center;font-size:1.2em;box-shadow:0 2px 4px rgba(0,0,0,.1)}.main-container{display:flex;flex-grow:1;overflow:hidden;padding:10px}.graph-view-container{width:35%;min-width:300px;background-color:#2d3748;color:#e2e8f0;padding:15px;margin-right:10px;border-radius:8px;display:flex;flex-direction:column;overflow-y:auto;font-family:'Courier New',Courier,monospace;font-size:.85em}.graph-view-container h2{margin-top:0;color:#90cdf4;border-bottom:1px solid #4a5568;padding-bottom:10px}#graphJsonView{flex-grow:1;white-space:pre-wrap;word-wrap:break-word;overflow-y:auto}#graphJsonView pre{margin:0;background-color:transparent!important;color:#e2e8f0!important;padding:0!important}.chat-container{display:flex;flex-direction:column;flex-grow:1;background-color:#fff;border-radius:8px;box-shadow:0 0 15px rgba(0,0,0,.1);overflow:hidden}#messages{flex-grow:1;padding:20px;overflow-y:scroll;border-bottom:1px solid #e0e0e0}.message{padding:10px 15px;margin-bottom:10px;border-radius:18px;max-width:75%;word-wrap:break-word;line-height:1.4}.user{background-color:#007bff;color:#fff;align-self:flex-end;margin-left:25%;border-bottom-right-radius:5px}.agent{background-color:#e9ecef;color:#495057;align-self:flex-start;margin-right:25%;border-bottom-left-radius:5px}.error{background-color:#ffebee;color:#c62828;border:1px solid #ef9a9a;padding:10px}.info{background-color:#e8f5e9;color:#2e7d32;font-style:italic;text-align:center;padding:6px;font-size:.9em;border-radius:8px}.status{background-color:#fffde7;color:#f57f17;font-style:italic;text-align:center;padding:6px;font-size:.9em;border-radius:8px}.intermediate{background-color:#e3f2fd;color:#1565c0;font-style:italic;font-size:.9em;margin-right:25%;align-self:flex-start;border-bottom-left-radius:5px}#inputArea{display:flex;padding:15px;border-top:1px solid #e0e0e0;background-color:#f8f9fa}textarea{flex-grow:1;padding:12px;border-radius:20px;border:1px solid #ced4da;resize:none;min-height:24px;max-height:100px;overflow-y:auto;font-size:1em;line-height:1.4}button{padding:12px 20px;margin-left:10px;border:none;background-color:#007bff;color:#fff;border-radius:20px;cursor:pointer;font-size:1em;display:flex;align-items:center;justify-content:center}button:hover{background-color:#0056b3}.thinking-indicator{width:20px;height:20px;border:3px solid rgba(0,123,255,.2);border-top-color:#007bff;border-radius:50%;animation:spin 1s linear infinite;display:none;margin-left:10px}@keyframes spin{to{transform:rotate(360deg)}}button span{margin-right:5px}pre{background-color:#282c34;color:#abb2bf;padding:1em;border-radius:4px;overflow-x:auto;white-space:pre-wrap;word-wrap:break-word}</style></head>
<body><div class=header>OpenAPI Multi-view Agent (Gemini LLM)</div><div class=main-container><div class=graph-view-container><h2>Execution Graph (JSON)</h2><div id=graphJsonView><pre>No graph loaded yet.</pre></div></div><div class=chat-container><div id=messages></div><div id=inputArea><textarea id=messageInput placeholder="Paste OpenAPI spec (JSON/YAML) or ask a question..."rows=1></textarea><button id=sendButton onclick=sendMessage()><span>Send</span><div class=thinking-indicator id=thinkingIndicator></div></button></div></div></div>
<script>
const messagesDiv=document.getElementById("messages"),messageInput=document.getElementById("messageInput"),graphJsonView=document.getElementById("graphJsonView").querySelector("pre"),sendButton=document.getElementById("sendButton"),thinkingIndicator=document.getElementById("thinkingIndicator");let ws;
function showThinking(e){thinkingIndicator.style.display=e?"inline-block":"none",sendButton.disabled=e,messageInput.disabled=e}
function connect(){const e=location.protocol==="https:"?"wss:":"ws:",t=e+"//"+location.host+"/ws/openapi_agent";addChatMessage("Connecting to: "+t,"info"),ws=new WebSocket(t),ws.onopen=()=>{addChatMessage("WebSocket connected.","info"),showThinking(!1)},ws.onmessage=e=>{const t=JSON.parse(e.data);let o=t.content;if("graph_update"===t.type)return graphJsonView.textContent=JSON.stringify(o,null,2),addChatMessage("Execution graph has been updated.","info"),void console.log("Graph Update Received:",o);if("status"===t.type&&o&&o.toLowerCase().includes("processing"))showThinking(!0);else if("final"===t.type||"error"===t.type||"info"===t.type||"warning"===t.type)showThinking(!1);else"intermediate"===t.type&&o&&o.toLowerCase().includes("processing");if("object"==typeof o)o="<pre>"+JSON.stringify(o,null,2)+"</pre>";else{o=String(o).replace(/</g,"&lt;").replace(/>/g,"&gt;");const e=t=>"<pre>"+t.replace(/</g,"&lt;").replace(/>/g,"&gt;")+"</pre>",n=t=>"<pre><code>"+t.replace(/</g,"&lt;").replace(/>/g,"&gt;")+"</code></pre>";o=o.replace(/```json\\n([\s\S]*?)\\n```/g,(t,o)=>e(o)),o=o.replace(/```(\w*?)\\n([\s\S]*?)\\n```/g,(t,o,e)=>n(e)),o=o.replace(/```\\n([\s\S]*?)\\n```/g,(t,o)=>e(o))}addChatMessage(`Agent (${t.type||"message"}): ${o}`,t.type||"agent")},ws.onerror=e=>{addChatMessage("WebSocket error. Check console. If page HTTPS, WS must be WSS.","error"),console.error("WebSocket error object:",e),showThinking(!1)},ws.onclose=e=>{let t="";e.code&&(t+=`Code: ${e.code} `),e.reason&&(t+=`Reason: ${e.reason} `),t+=e.wasClean?"(Clean close) ":"(Unclean close) ",addChatMessage("WebSocket disconnected. "+t+"Attempting to reconnect in 5s...","info"),console.log("WebSocket close event:",e),showThinking(!1),setTimeout(connect,5e3)}}
function addChatMessage(e,t){const o=document.createElement("div");o.innerHTML=e,o.className="message "+t,messagesDiv.appendChild(o),messagesDiv.scrollTop=messagesDiv.scrollHeight}
function sendMessage(){if(ws&&ws.readyState===WebSocket.OPEN){const e=messageInput.value;e.trim()&&(addChatMessage("You: "+e.replace(/</g,"&lt;").replace(/>/g,"&gt;"),"user"),ws.send(e),messageInput.value="",showThinking(!0))}else addChatMessage("WebSocket is not connected.","error")}
messageInput.addEventListener("input",function(){this.style.height="auto",this.style.height=this.scrollHeight+"px"}),messageInput.addEventListener("keypress",function(e){"Enter"===e.key&&!e.shiftKey&&(e.preventDefault(),sendMessage())}),connect();
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    return HTML_TEST_PAGE
