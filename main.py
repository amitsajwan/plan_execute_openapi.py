# main.py
import logging
import uuid
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple 

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse 
from langgraph.checkpoint.memory import MemorySaver 

from models import BotState 
from graph import build_graph 
from pydantic import ValidationError 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- LLM Initialization ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(".env file loaded if present.")
except ImportError:
    logger.warning("python-dotenv not installed. Cannot load .env file.")

class PlaceholderLLM:
    def __init__(self, name="PlaceholderLLM", delay=0.01): 
        self.name = name
        self.delay = delay
        logger.warning(f"Initialized {self.name}. This is a fallback if real LLMs fail.")

    def _generate_simulated_response(self, prompt_str: str) -> str:
        if "Summarize this OpenAPI specification" in prompt_str:
            return "This is a simulated API summary from PlaceholderLLM."
        if "design an API execution graph" in prompt_str.lower():
            if "Cannot generate graph: No APIs identified." in prompt_str: 
                 return json.dumps({ "nodes": [], "edges": [], "description": "Placeholder: Cannot generate graph, no APIs."})
            return json.dumps({
                "nodes": [{"operationId": "createItem", "summary": "Create an item", "description":"Initial step", "payload_description":"Req: name. Resp: id", "input_mappings":[]}],
                "edges": [], "description": "Simulated graph from PlaceholderLLM: Create an item."})
        if "Critique and refine" in prompt_str:
             return json.dumps({
                "nodes": [{"operationId": "createItem", "summary": "Create item (refined by Placeholder)", "payload_description":"Req: name. Resp: id", "input_mappings":[]}],
                "edges": [], "description": "Refined graph by Placeholder.", "refinement_summary": "Placeholder refined details."})
        if "Classify the user's intent" in prompt_str: 
            if "focus on" in prompt_str.lower() or "what if" in prompt_str.lower(): return "interactive_query_planner"
            return "answer_openapi_query"
        if "plans internal actions" in prompt_str: 
            return json.dumps({
                "user_query_understanding": "User wants info on 'createItem' (Placeholder).",
                "interactive_action_plan": [{"action_name": "answer_query_directly", "action_params": {"query_for_synthesizer": "About createItem"}, "description":"Answer about createItem"}]})
        if "Determine the user's high-level intent" in prompt_str:
            if "graph" in prompt_str.lower(): return "describe_graph"
            return "answer_openapi_query"
        return f"Simulated response from {self.name} for: {prompt_str[:50]}..."

    def invoke(self, prompt: Any, **kwargs) -> Any:
        prompt_str = str(prompt); logger.debug(f"{self.name} prompt: {prompt_str[:100]}...")
        class ContentWrapper: def __init__(self, text): self.content = text
        return ContentWrapper(self._generate_simulated_response(prompt_str))
    async def ainvoke(self, prompt: Any, **kwargs) -> Any:
        import asyncio; await asyncio.sleep(self.delay); return self.invoke(prompt, **kwargs)

def initialize_llms() -> Tuple[Any, Any]:
    logger.info("Initializing LLMs (Google Gemini)...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    router_llm, worker_llm = None, None
    if google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            router_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=google_api_key, convert_system_message_to_human=True)
            worker_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1, google_api_key=google_api_key, convert_system_message_to_human=True)
            logger.info("Gemini LLMs initialized: Router (Flash), Worker (Pro).")
        except Exception as e: logger.error(f"Failed to init Gemini: {e}", exc_info=True)
    else: logger.warning("GOOGLE_API_KEY not set.")
    if not router_llm: router_llm = PlaceholderLLM("RouterLLM_Fallback"); logger.warning("Using Placeholder Router LLM.")
    if not worker_llm: worker_llm = PlaceholderLLM("WorkerLLM_Fallback"); logger.warning("Using Placeholder Worker LLM.")
    return router_llm, worker_llm

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
                    # Attempt to extract the state dictionary more robustly
                    state_dict = current_checkpoint.get("channel_values", {}) 
                    if not isinstance(state_dict, dict) or not state_dict : # If channel_values is empty or not a dict, try current_checkpoint itself
                        state_dict = current_checkpoint if isinstance(current_checkpoint, dict) else {}
                    
                    # LangGraph with Pydantic state often nests it under "__root__" if not using Annotated state
                    # Or it might be directly the fields if the StateGraph was defined with the Pydantic model directly
                    # This part needs to be robust to how MemorySaver stores the Pydantic model's dict.
                    if "__root__" in state_dict and isinstance(state_dict["__root__"], dict):
                        state_dict_to_validate = state_dict["__root__"]
                    elif "session_id" in state_dict: # Check if it's already a flat state dict
                         state_dict_to_validate = state_dict
                    else: # Could be nested under the last active node name if something went wrong
                        # This is a less common case for the *input* to the graph run, but defensive.
                        active_node_key = next(iter(state_dict), None) if state_dict else None
                        if active_node_key and isinstance(state_dict.get(active_node_key), dict):
                            state_dict_to_validate = state_dict.get(active_node_key, {})
                            logger.warning(f"[{session_id}] Loaded state was nested under key '{active_node_key}'.")
                        else:
                            state_dict_to_validate = state_dict # Fallback to whatever state_dict is

                    initial_state_for_turn = BotState.model_validate(state_dict_to_validate)
                    initial_state_for_turn.user_input = user_input
                    initial_state_for_turn.response = None; initial_state_for_turn.final_response = ""
                    initial_state_for_turn.next_step = None; initial_state_for_turn.intent = None
                    logger.debug(f"[{session_id}] Loaded state from checkpoint: {initial_state_for_turn.session_id}")
                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading state from checkpoint: {e}. Data: {str(current_checkpoint)[:300]}. Starting fresh.")
                    initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)
            else: 
                logger.debug(f"[{session_id}] No checkpoint. New state.")
                initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)

            final_state_snapshot: Optional[BotState] = None
            try:
                if langgraph_app is None: raise RuntimeError("Agent not available.")
                async for stream_event in langgraph_app.astream_events(initial_state_for_turn, config=config, version="v1"): 
                    event_type, event_data, event_name = stream_event["event"], stream_event["data"], stream_event.get("name", "unknown_node")
                    if event_type in ("on_chain_end", "on_tool_end"): # These events usually contain the full state output of the node
                        output_val = event_data.get("output", event_data.get("outputs"))
                        logger.debug(f"[{session_id}] Node '{event_name}' raw output_val type: {type(output_val)}, content: {str(output_val)[:300]}...")

                        dict_to_validate = None
                        if isinstance(output_val, dict):
                            # For StateGraph(BotState), output_val should be the dictionary representation of BotState.
                            if "session_id" in output_val: # Check if it's directly the state
                                dict_to_validate = output_val
                            # Check if nested under node name (less common for direct node output but defensive)
                            elif event_name in output_val and isinstance(output_val[event_name], dict) and "session_id" in output_val[event_name]:
                                dict_to_validate = output_val[event_name]
                                logger.info(f"[{session_id}] Extracted state from output nested under node name '{event_name}'.")
                            # Check if nested under "__root__" (common for Pydantic models in some LangGraph versions/setups)
                            elif "__root__" in output_val and isinstance(output_val["__root__"], dict) and "session_id" in output_val["__root__"]:
                                dict_to_validate = output_val["__root__"]
                                logger.info(f"[{session_id}] Extracted state from output nested under '__root__'.")
                            else:
                                logger.warning(f"[{session_id}] Node '{event_name}' output dict structure not directly BotState or common nesting. Will attempt to validate directly. Output: {str(output_val)[:200]}")
                                dict_to_validate = output_val # Attempt direct validation as last resort for dicts
                        
                        if dict_to_validate and isinstance(dict_to_validate, dict):
                            try:
                                node_state = BotState.model_validate(dict_to_validate)
                                final_state_snapshot = node_state 
                                if node_state.response: 
                                    logger.debug(f"[{session_id}] Node '{event_name}' intermediate: {node_state.response[:100]}...")
                                    await websocket.send_json({"type": "intermediate", "content": node_state.response})
                            except ValidationError as e_val: 
                                logger.error(f"[{session_id}] Pydantic state validation error for node '{event_name}': {e_val}. Data tried: {str(dict_to_validate)[:300]}", exc_info=False)
                            except Exception as e_proc: 
                                logger.error(f"[{session_id}] Error processing output of node '{event_name}': {e_proc}. Data: {str(dict_to_validate)[:300]}", exc_info=True)
                        elif output_val is not None : # If not a dict but not None
                             logger.warning(f"[{session_id}] Node '{event_name}' output was not a dictionary. Type: {type(output_val)}, Output: {str(output_val)[:300]}")
                
                logger.info(f"[{session_id}] Stream finished. Snapshot valid: {final_state_snapshot is not None}")
                if final_state_snapshot: logger.info(f"[{session_id}] Snapshot final_response: '{final_state_snapshot.final_response}' (responder should have set this)")

                if final_state_snapshot and final_state_snapshot.final_response:
                    logger.info(f"[{session_id}] Sending final: {final_state_snapshot.final_response[:100]}...")
                    await websocket.send_json({"type": "final", "content": final_state_snapshot.final_response})
                else: # Fallback if no final_response
                    error_message_to_user = "Processing complete, but no specific message was generated by the agent."
                    if final_state_snapshot and final_state_snapshot.response: # Should be None after responder
                        error_message_to_user = final_state_snapshot.response 
                        logger.warning(f"[{session_id}] No final_response, using last intermediate response from snapshot: {error_message_to_user[:100]}...")
                    elif initial_state_for_turn.response: # If an error was set early and graph didn't run far
                        error_message_to_user = initial_state_for_turn.response
                        logger.warning(f"[{session_id}] No final_response, using response from initial state for turn: {error_message_to_user[:100]}...")
                    else:
                        logger.error(f"[{session_id}] No final_response or any response in final/initial state snapshot for user.")
                    await websocket.send_json({"type": "error", "content": error_message_to_user})

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
<head><title>OpenAPI Agent (Gemini)</title><style>body{font-family:sans-serif;margin:20px;background-color:#f4f4f4;color:#333}#messages{width:100%;max-width:800px;margin:20px auto;background-color:#fff;border-radius:8px;box-shadow:0 0 10px rgba(0,0,0,.1);padding:20px;height:400px;overflow-y:scroll;border:1px solid #ddd}.message{padding:8px;margin-bottom:8px;border-radius:4px}.user{background-color:#e1f5fe;text-align:right;margin-left:50px}.agent{background-color:#f0f0f0;margin-right:50px}.error{background-color:#ffcdd2;color:#c62828}.info{background-color:#c8e6c9;color:#2e7d32;font-style:italic}.status{background-color:#fff9c4;font-style:italic;text-align:center}#inputArea{display:flex;max-width:800px;margin:20px auto}textarea{flex-grow:1;padding:10px;border-radius:4px;border:1px solid #ccc;resize:vertical;min-height:80px}button{padding:10px 15px;margin-left:10px;border:none;background-color:#007bff;color:#fff;border-radius:4px;cursor:pointer}button:hover{background-color:#0056b3}pre{background-color:#282c34;color:#abb2bf;padding:1em;border-radius:4px;overflow-x:auto;white-space:pre-wrap;word-wrap:break-word}</style></head>
<body><h1>OpenAPI Agent Test (Gemini LLM)</h1><div id="messages"></div><div id="inputArea"><textarea id="messageInput" placeholder="Paste OpenAPI spec (JSON/YAML) or ask a question..."></textarea><button onclick="sendMessage()">Send</button></div>
<script>
const messagesDiv=document.getElementById("messages"),messageInput=document.getElementById("messageInput");let ws;
function connect(){const t=location.protocol==="https:"?"wss:":"ws:",o=t+"//"+location.host+"/ws/openapi_agent";addMessage("Connecting to: "+o,"info"),ws=new WebSocket(o),ws.onopen=()=>{addMessage("WebSocket connected.","info")},ws.onmessage=t=>{const o=JSON.parse(t.data);let e=o.content;"object"==typeof e?e="<pre>"+JSON.stringify(e,null,2)+"</pre>":(e=String(e).replace(/</g,"&lt;").replace(/>/g,"&gt;"),e=e.replace(/```json\\n([\s\S]*?)\\n```/g,(t,o)=>"<pre>"+o.replace(/</g,"&lt;").replace(/>/g,"&gt;")+"</pre>"),e=e.replace(/```([\s\S]*?)\\n([\s\S]*?)\\n```/g,(t,o,n)=>"<pre><code>"+n.replace(/</g,"&lt;").replace(/>/g,"&gt;")+"</code></pre>"),e=e.replace(/```\\n([\s\S]*?)\\n```/g,(t,o)=>"<pre>"+o.replace(/</g,"&lt;").replace(/>/g,"&gt;")+"</pre>")),addMessage(`Agent (${o.type||"message"}): ${e}`,o.type||"agent")},ws.onerror=t=>{addMessage("WebSocket error. Check console. If page HTTPS, WS must be WSS.","error"),console.error("WebSocket error object:",t)},ws.onclose=t=>{let o="";t.code&&(o+=`Code: ${t.code} `),t.reason&&(o+=`Reason: ${t.reason} `),o+=t.wasClean?"(Clean close) ":"(Unclean close) ",addMessage("WebSocket disconnected. "+o+"Attempting to reconnect in 5s...","info"),console.log("WebSocket close event:",t),setTimeout(connect,5e3)}}
function addMessage(t,o){const e=document.createElement("p");e.innerHTML=t,e.className="message "+o,messagesDiv.appendChild(e),messagesDiv.scrollTop=messagesDiv.scrollHeight}
function sendMessage(){if(ws&&ws.readyState===WebSocket.OPEN){const t=messageInput.value;t.trim()&&(addMessage("You: "+t.replace(/</g,"&lt;").replace(/>/g,"&gt;"),"user"),ws.send(t),messageInput.value="")}else addMessage("WebSocket is not connected.","error")}
messageInput.addEventListener("keypress",function(t){"Enter"===t.key&&!t.shiftKey&&(t.preventDefault(),sendMessage())}),connect();
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    return HTML_TEST_PAGE
