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
            return "This is a simulated API summary from PlaceholderLLM. It details several interesting endpoints for resource management."
        if "design an API execution graph" in prompt_str.lower():
            if "Cannot generate graph: No identified APIs." in prompt_str or "No identified APIs." in prompt_str:
                 return json.dumps({ 
                    "nodes": [],
                    "edges": [],
                    "description": "Placeholder: Cannot generate graph as no APIs were identified to form a workflow."
                })
            return json.dumps({
                "nodes": [{"operationId": "createItem", "summary": "Create an item", "description":"Initial step", "payload_description":"Req: name. Resp: id", "input_mappings":[]}],
                "edges": [],
                "description": "Simulated graph from PlaceholderLLM: Create an item."
            })
        if "Critique and refine the following API execution graph" in prompt_str:
             return json.dumps({
                "nodes": [{"operationId": "createItem", "summary": "Create an item (refined by PlaceholderLLM)", "description":"Initial step (refined)", "payload_description":"Req: name (string). Resp: id (string)", "input_mappings":[]}],
                "edges": [],
                "description": "Refined graph from PlaceholderLLM: Create an item with more detail.",
                "refinement_summary": "PlaceholderLLM added more detail to payload description."
            })
        if "Classify the user's intent" in prompt_str: 
            if "focus on" in prompt_str.lower() or "what if" in prompt_str.lower() or "change the plan" in prompt_str.lower():
                return "interactive_query_planner"
            return "answer_openapi_query"
        if "plans internal actions to respond to a user's query" in prompt_str: 
            return json.dumps({
                "user_query_understanding": "User wants to know more about 'createItem' (PlaceholderLLM).",
                "interactive_action_plan": [{"action_name": "answer_query_directly", "action_params": {"query_for_synthesizer": "Tell me about createItem API"}, "description":"Answer about createItem"}]
            })
        
        if "Determine the user's high-level intent from the list" in prompt_str:
            if "graph" in prompt_str.lower(): return "describe_graph"
            if "plan" in prompt_str.lower(): return "_generate_execution_graph"
            return "answer_openapi_query"

        return f"Simulated response from {self.name} for prompt starting with: {prompt_str[:60]}..."

    def invoke(self, prompt: Any, **kwargs) -> Any:
        prompt_str = str(prompt)
        logger.debug(f"{self.name} received prompt: {prompt_str[:200]}...")
        class ContentWrapper:
            def __init__(self, text): self.content = text
        return ContentWrapper(self._generate_simulated_response(prompt_str))

    async def ainvoke(self, prompt: Any, **kwargs) -> Any:
        import asyncio
        await asyncio.sleep(self.delay)
        return self.invoke(prompt, **kwargs)


def initialize_llms() -> Tuple[Any, Any]:
    logger.info("Initializing LLMs with Google Gemini...")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    router_llm = None
    worker_llm = None

    if google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            router_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0, google_api_key=google_api_key,
                convert_system_message_to_human=True )
            logger.info("Router LLM (Gemini Flash) initialized.")
            worker_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash", temperature=0.1, google_api_key=google_api_key,
                convert_system_message_to_human=True )
            logger.info("Worker LLM (Gemini 1.5 Pro) initialized.")
        except ImportError: logger.error("langchain-google-genai not installed.")
        except Exception as e: logger.error(f"Failed to initialize Google Gemini LLMs: {e}", exc_info=True)
    else: logger.warning("GOOGLE_API_KEY environment variable not set.")

    if router_llm is None:
        logger.warning("Router LLM (Gemini) failed. Falling back to PlaceholderLLM.")
        router_llm = PlaceholderLLM(name="RouterLLM_Placeholder_Fallback")
    if worker_llm is None:
        logger.warning("Worker LLM (Gemini) failed. Falling back to PlaceholderLLM.")
        worker_llm = PlaceholderLLM(name="WorkerLLM_Placeholder_Fallback")
    
    if not hasattr(router_llm, 'invoke') or not hasattr(worker_llm, 'invoke'):
        raise TypeError("Initialized LLMs must have 'invoke' method.")
    if not hasattr(router_llm, 'ainvoke') or not hasattr(worker_llm, 'ainvoke'):
        logger.warning("LLMs missing 'ainvoke'. Adding dummy ainvoke.")
        if not hasattr(router_llm, 'ainvoke'): router_llm.ainvoke = lambda p, **k: router_llm.invoke(p, **k) # type: ignore
        if not hasattr(worker_llm, 'ainvoke'): worker_llm.ainvoke = lambda p, **k: worker_llm.invoke(p, **k) # type: ignore
    logger.info("LLM clients initialization attempt finished.")
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
        langgraph_app = build_graph(
            router_llm=router_llm_instance, worker_llm=worker_llm_instance,
            checkpointer=checkpointer )
        logger.info("LangGraph application retrieved (compiled in build_graph).")
    except Exception as e:
        logger.critical(f"Failed to initialize/build graph on startup: {e}", exc_info=True)
        langgraph_app = None

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown event.")
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
    logger.info(f"WebSocket accepted for session: {session_id}")

    if langgraph_app is None:
        await websocket.send_json({"type": "error", "content": "Backend agent not initialized."})
        await websocket.close(code=1011); return

    await websocket.send_json({"type": "info", "session_id": session_id, "content": "Connected. Provide OpenAPI spec or ask."})

    try:
        while True:
            user_input = await websocket.receive_text()
            user_input = user_input.strip()
            logger.info(f"[{session_id}] Received: {user_input[:200]}...")
            if not user_input:
                await websocket.send_json({"type": "warning", "content": "Empty message."}); continue
            await websocket.send_json({"type": "status", "content": "Processing..."})

            config = {"configurable": {"thread_id": session_id}}
            current_checkpoint = checkpointer.get(config)
            initial_state_for_turn: BotState
            if current_checkpoint:
                try:
                    state_dict = current_checkpoint.get("channel_values", current_checkpoint if isinstance(current_checkpoint, dict) else {})
                    if "__root__" in state_dict: state_dict = state_dict["__root__"] # type: ignore
                    initial_state_for_turn = BotState.model_validate(state_dict)
                    initial_state_for_turn.user_input = user_input
                    initial_state_for_turn.response = None
                    initial_state_for_turn.final_response = ""
                    initial_state_for_turn.next_step = None
                    initial_state_for_turn.intent = None
                    logger.debug(f"[{session_id}] Loaded state from checkpoint.")
                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Error loading state: {e}. Starting fresh.")
                    initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)
            else:
                logger.debug(f"[{session_id}] No checkpoint. New state.")
                initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)

            final_state_snapshot: Optional[BotState] = None
            try:
                if langgraph_app is None:
                    logger.error(f"[{session_id}] langgraph_app is None."); 
                    await websocket.send_json({"type": "error", "content": "Agent not available."}); continue

                async for stream_event in langgraph_app.astream_events(initial_state_for_turn, config=config, version="v1"): 
                    event_type, event_data = stream_event["event"], stream_event["data"]
                    if event_type in ("on_chain_end", "on_tool_end"):
                        output_dict = event_data.get("output", event_data.get("outputs"))
                        if output_dict and isinstance(output_dict, dict):
                            try:
                                node_state = BotState.model_validate(output_dict)
                                final_state_snapshot = node_state # Keep latest state
                                if node_state.response:
                                    logger.debug(f"[{session_id}] Intermediate: {node_state.response[:100]}...")
                                    await websocket.send_json({"type": "intermediate", "content": node_state.response})
                            except Exception as e_val: logger.error(f"[{session_id}] State validation error: {e_val}. Data: {output_dict}", exc_info=True)
                
                logger.info(f"[{session_id}] Stream finished. Snapshot: {bool(final_state_snapshot)}")
                if final_state_snapshot: logger.info(f"[{session_id}] Snapshot final_response: '{final_state_snapshot.final_response}'")

                if final_state_snapshot and final_state_snapshot.final_response:
                    logger.info(f"[{session_id}] Sending final: {final_state_snapshot.final_response[:100]}...")
                    await websocket.send_json({"type": "final", "content": final_state_snapshot.final_response})
                else:
                    logger.error(f"[{session_id}] No final_response in snapshot.")
                    await websocket.send_json({"type": "error", "content": "No final message generated."})

            except Exception as e_graph:
                logger.critical(f"[{session_id}] LangGraph error: {e_graph}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"Error: {e_graph}"})

    except WebSocketDisconnect: logger.info(f"WS disconnected: {session_id}")
    except Exception as e_outer:
        logger.critical(f"[{session_id}] WS handler error: {e_outer}", exc_info=True)
        try: await websocket.send_json({"type": "error", "content": f"Server error: {e_outer}"})
        except: pass
    finally:
        logger.info(f"[{session_id}] Closing WS connection.")
        try: await websocket.close()
        except: pass

HTML_TEST_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenAPI Agent Test (Gemini)</title>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        #messages { width: 100%; max-width: 800px; margin: 20px auto; background-color: #fff; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); padding: 20px; height: 400px; overflow-y: scroll; border: 1px solid #ddd;}
        .message { padding: 8px; margin-bottom: 8px; border-radius: 4px; }
        .user { background-color: #e1f5fe; text-align: right; margin-left: 50px;}
        .agent { background-color: #f0f0f0; margin-right: 50px;}
        .error { background-color: #ffcdd2; color: #c62828; }
        .info { background-color: #c8e6c9; color: #2e7d32; font-style: italic;}
        .status { background-color: #fff9c4; font-style: italic; text-align: center;}
        #inputArea { display: flex; max-width: 800px; margin: 20px auto; }
        textarea { flex-grow: 1; padding: 10px; border-radius: 4px; border: 1px solid #ccc; resize: vertical; min-height: 80px;}
        button { padding: 10px 15px; margin-left: 10px; border: none; background-color: #007bff; color: white; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        pre { background-color: #282c34; color: #abb2bf; padding: 1em; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>OpenAPI Agent Test (Gemini LLM)</h1>
    <div id="messages"></div>
    <div id="inputArea">
        <textarea id="messageInput" placeholder="Paste OpenAPI spec (JSON/YAML) or ask a question..."></textarea>
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        let ws;

        function connect() {
            const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
            const wsUrl = wsProtocol + "//" + location.host + "/ws/openapi_agent";
            addMessage("Connecting to: " + wsUrl, "info");
            ws = new WebSocket(wsUrl);

            ws.onopen = () => { addMessage("WebSocket connected.", "info"); };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                let content = data.content;
                if (typeof content === 'object') { 
                    content = '<pre>' + JSON.stringify(content, null, 2) + '</pre>';
                } else { 
                    content = String(content).replace(/</g, "&lt;").replace(/>/g, "&gt;");
                    content = content.replace(/```json\\n([\s\S]*?)\\n```/g, (match, p1) => '<pre>' + p1.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</pre>');
                    content = content.replace(/```([\s\S]*?)\\n([\s\S]*?)\\n```/g, (match, lang, code) => '<pre><code>' + code.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</code></pre>');
                    content = content.replace(/```\\n([\s\S]*?)\\n```/g, (match, p1) => '<pre>' + p1.replace(/</g, "&lt;").replace(/>/g, "&gt;") + '</pre>');
                }
                addMessage(`Agent (${data.type || 'message'}): ${content}`, data.type || "agent");
            };
            ws.onerror = (error) => { 
                addMessage("WebSocket error. Check browser console for details. If page is HTTPS, WS must be WSS.", "error"); 
                console.error("WebSocket error object:", error);
            };
            ws.onclose = (event) => { 
                let reason = "";
                if (event.code) reason += `Code: ${event.code} `;
                if (event.reason) reason += `Reason: ${event.reason} `;
                if (event.wasClean) reason += `(Clean close) `; else reason += `(Unclean close) `;
                addMessage("WebSocket disconnected. " + reason + "Attempting to reconnect in 5s...", "info");
                console.log("WebSocket close event:", event);
                setTimeout(connect, 5000);
            };
        }
        function addMessage(message, type) {
            const p = document.createElement('p');
            p.innerHTML = message; 
            p.className = 'message ' + type;
            messagesDiv.appendChild(p);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        function sendMessage() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const message = messageInput.value;
                if (message.trim() === "") return;
                addMessage("You: " + message.replace(/</g, "&lt;").replace(/>/g, "&gt;"), "user");
                ws.send(message);
                messageInput.value = '';
            } else { addMessage("WebSocket is not connected.", "error"); }
        }
        messageInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
        connect();
    </script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    return HTML_TEST_PAGE
