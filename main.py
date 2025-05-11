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

HTML_TEST_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenAPI Agent (Gemini)</title>
    <style>
        body { 
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; 
            margin: 0; 
            background-color: #f0f2f5; 
            color: #333; 
            display: flex; 
            flex-direction: column; 
            height: 100vh; 
            line-height: 1.6;
        }
        .header { 
            background-color: #4A90E2; 
            color: white; 
            padding: 15px 20px; 
            text-align: center; 
            font-size: 1.2em; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        }
        .main-container { 
            display: flex; 
            flex-grow: 1; 
            overflow: hidden; 
            padding: 10px; 
            gap: 10px; 
        }
        .graph-view-container { 
            width: 35%; 
            min-width: 350px; 
            max-width: 40%; 
            background-color: #2d3748; 
            color: #e2e8f0; 
            padding: 15px; 
            margin-right: 10px; 
            border-radius: 8px; 
            display: flex; 
            flex-direction: column; 
            overflow-y: auto; 
            font-family: Menlo, Monaco, 'Courier New', monospace; 
            font-size: 0.85em; 
        }
        .graph-view-container h2 { 
            margin-top: 0; 
            color: #90cdf4; 
            border-bottom: 1px solid #4a5568; 
            padding-bottom: 10px; 
        }
        #graphJsonView { 
            flex-grow: 1; 
            white-space: pre-wrap; 
            word-wrap: break-word; 
            overflow-y: auto; 
            padding: 10px; 
            background-color: #1a202c; 
            border-radius: 4px; 
        }
        #graphJsonView pre { 
            margin: 0; 
            background-color: transparent !important; 
            color: #e2e8f0 !important; 
            padding: 0 !important; 
        }
        .chat-container { 
            display: flex; 
            flex-direction: column; 
            flex-grow: 1; 
            background-color: #fff; 
            border-radius: 8px; 
            box-shadow: 0 0 15px rgba(0,0,0,0.1); 
            overflow: hidden; 
        }
        #messages { 
            flex-grow: 1; 
            padding: 20px; 
            overflow-y: auto; 
            border-bottom: 1px solid #e0e0e0;
        }
        .message { 
            padding: 10px 15px; 
            margin-bottom: 10px; 
            border-radius: 18px; 
            max-width: 85%; 
            word-wrap: break-word; 
            line-height: 1.5; 
            font-size: 0.95em; 
        }
        .user { 
            background-color: #007bff; 
            color: white; 
            align-self: flex-end; 
            margin-left: 15%; 
            border-bottom-right-radius: 5px; 
        }
        .agent { 
            background-color: #e9ecef; 
            color: #343a40; 
            align-self: flex-start; 
            margin-right: 15%; 
            border-bottom-left-radius: 5px; 
        }
        .agent h4 { margin-top: 0; margin-bottom: 8px; color: #00529B; font-size: 1.1em; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
        .agent h5 { margin-top: 0; margin-bottom: 6px; color: #333; font-size: 1.05em; }
        .agent h6 { margin-top: 0; margin-bottom: 4px; color: #555; font-size: 1em; }
        .agent ul { padding-left: 20px; margin-top: 5px; margin-bottom: 10px; list-style-type: disc; }
        .agent li { margin-bottom: 4px; }
        .agent p { margin-top: 0; margin-bottom: 10px; }
        .agent pre { margin-bottom: 10px; }
        .error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 10px; }
        .info { background-color: #d4edda; color: #155724; font-style: italic; text-align: center; padding: 6px; font-size: 0.9em; border-radius: 8px; }
        .status { background-color: #fff3cd; color: #856404; font-style: italic; text-align: center; padding: 6px; font-size: 0.9em; border-radius: 8px; }
        .intermediate { background-color: #e0e0e0; color: #555; font-style: italic; font-size: 0.85em; margin-right: 30%; align-self: flex-start; border-radius: 12px; border-bottom-left-radius: 5px; }
        #inputArea { display: flex; padding: 15px; border-top: 1px solid #e0e0e0; background-color: #f8f9fa; }
        textarea { 
            flex-grow: 1; 
            padding: 12px 15px; 
            border-radius: 20px; 
            border: 1px solid #ced4da; 
            resize: none; 
            min-height: 24px; /* Start small */
            max-height: 120px; /* Max height before scroll */
            overflow-y: auto; 
            font-size: 1em; 
            line-height: 1.5;
        }
        button { 
            padding: 0 20px; /* Adjusted padding for height */
            margin-left: 10px; 
            border: none; 
            background-color: #007bff; 
            color: white; 
            border-radius: 20px; 
            cursor: pointer; 
            font-size: 1em; 
            display: flex; 
            align-items: center; 
            justify-content: center;
            height: 48px; /* Match textarea initial height + padding */
        }
        button:hover { background-color: #0056b3; }
        .thinking-indicator { 
            width: 18px; height: 18px; 
            border: 3px solid hsla(210, 100%, 50%, 0.2); 
            border-top-color: #007bff; 
            border-radius: 50%; 
            animation: spin 0.8s linear infinite; 
            display: none; /* Hidden by default */
            margin-left: 8px; 
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        button span { margin-right: 5px; }
        pre { 
            background-color: #282c34; 
            color: #e2e8f0; /* Changed from abb2bf for better contrast on dark */
            padding: 1em; 
            border-radius: 6px; 
            overflow-x: auto; 
            white-space: pre-wrap; 
            word-wrap: break-word;
            font-family: Menlo, Monaco, 'Courier New', monospace;
            font-size: 0.9em;
        }
        code { font-family: Menlo, Monaco, 'Courier New', monospace; }
    </style>
</head>
<body>
    <div class="header">OpenAPI Multi-view Agent (Gemini LLM)</div>
    <div class="main-container">
        <div class="graph-view-container">
            <h2>Execution Graph (JSON)</h2>
            <div id="graphJsonView"><pre>No graph loaded yet.</pre></div>
        </div>
        <div class="chat-container">
            <div id="messages"></div>
            <div id="inputArea">
                <textarea id="messageInput" placeholder="Paste OpenAPI spec (JSON/YAML) or ask a question..." rows="1"></textarea>
                <button id="sendButton" onclick="sendMessage()">
                    <span>Send</span>
                    <div class="thinking-indicator" id="thinkingIndicator"></div>
                </button>
            </div>
        </div>
    </div>

<script>
    const messagesDiv = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const graphJsonViewPre = document.getElementById('graphJsonView').querySelector('pre');
    const sendButton = document.getElementById('sendButton');
    const thinkingIndicator = document.getElementById('thinkingIndicator');
    let ws;

    function showThinking(show) {
        thinkingIndicator.style.display = show ? 'inline-block' : 'none';
        sendButton.disabled = show;
        messageInput.disabled = show;
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
            return JSON.stringify(unsafe, null, 2); // Fallback for non-strings
        }
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function formatMessageContent(contentStr, type) {
        if (typeof contentStr !== 'string') {
            return '<pre>' + escapeHtml(JSON.stringify(contentStr, null, 2)) + '</pre>';
        }

        // Apply rich formatting only for 'final' or 'agent' (default) message types
        if (type !== 'agent' && type !== 'final') {
            return escapeHtml(contentStr); // Basic escaping for other types
        }
        
        let html = '';
        const codeBlockPlaceholders = [];
        
        // Pre-process to handle fenced code blocks
        let processedContentStr = contentStr.replace(/```(\w*)\n([\s\S]*?)\n```/g, (match, lang, code) => {
            const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
            const langClass = lang ? `language-${escapeHtml(lang)}` : 'language-plaintext';
            codeBlockPlaceholders.push(`<pre><code class="${langClass}">${escapeHtml(code)}</code></pre>`);
            return placeholder;
        });
        // For code blocks without language identifier
        processedContentStr = processedContentStr.replace(/```\n([\s\S]*?)\n```/g, (match, code) => {
            const placeholder = `__CODEBLOCK_${codeBlockPlaceholders.length}__`;
            codeBlockPlaceholders.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
            return placeholder;
        });

        const lines = processedContentStr.split('\\n'); // Split by actual escaped newline from Python string
        let inList = false;
        let paragraphBuffer = [];

        function flushParagraph() {
            if (paragraphBuffer.length > 0) {
                html += `<p>${paragraphBuffer.join('<br>')}</p>`; // Use <br> for multi-line paragraphs
                paragraphBuffer = [];
            }
        }

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i]; 

            if (line.startsWith("### ")) {
                flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
                html += `<h4>${escapeHtml(line.substring(4))}</h4>`;
            } else if (line.startsWith("## ")) {
                flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
                html += `<h5>${escapeHtml(line.substring(3))}</h5>`;
            } else if (line.startsWith("# ")) {
                flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
                html += `<h6>${escapeHtml(line.substring(2))}</h6>`;
            } else if (line.startsWith("- ") || line.startsWith("* ")) {
                flushParagraph();
                if (!inList) {
                    html += '<ul>';
                    inList = true;
                }
                html += `<li>${escapeHtml(line.substring(line.startsWith("- ") ? 2 : 1).trim())}</li>`;
            } else if (line.startsWith("__CODEBLOCK_")) { 
                flushParagraph(); if (inList) { html += '</ul>'; inList = false; }
                const placeholderIndex = parseInt(line.substring("__CODEBLOCK_".length, line.lastIndexOf("__")));
                if (placeholderIndex >= 0 && placeholderIndex < codeBlockPlaceholders.length) {
                    html += codeBlockPlaceholders[placeholderIndex];
                } else {
                    html += escapeHtml(line); // Fallback if placeholder is weird
                }
            } else { 
                if (inList && line.trim() !== "" && !line.trim().startsWith("- ") && !line.trim().startsWith("* ")) {
                    // Line after a list item that isn't another list item, ends the list.
                    html += '</ul>'; inList = false;
                }
                
                if (line.trim() === "") {
                    flushParagraph(); 
                } else {
                    paragraphBuffer.push(escapeHtml(line)); // Don't trim here, let <p> handle spacing
                }
            }
        }
        flushParagraph(); 
        if (inList) html += '</ul>'; 
        
        return html || "<p>" + escapeHtml(contentStr) + "</p>"; // Fallback if all lines were e.g. placeholders
    }

    function connect() {
        const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = wsProtocol + "//" + location.host + "/ws/openapi_agent";
        addChatMessage("Connecting to: " + wsUrl, "info");
        ws = new WebSocket(wsUrl);

        ws.onopen = () => { addChatMessage("WebSocket connected.", "info"); showThinking(false); };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            let content = data.content;
            const type = data.type || "agent";

            if (type === "graph_update") {
                graphJsonViewPre.textContent = JSON.stringify(content, null, 2);
                addChatMessage("Execution graph has been updated.", "info");
                console.log("Graph Update Received:", content);
                return; 
            }

            if (type === "status" && content && content.toLowerCase().includes("processing")) {
                showThinking(true);
            } else if (type === "final" || type === "error" || type === "info" || type === "warning") {
                showThinking(false);
            }
            // Intermediate messages might also stop thinking if they are substantial,
            // but for now, only the above types explicitly stop it.

            content = formatMessageContent(content, type);
            addChatMessage(`Agent (${type}): ${content}`, type);
        };

        ws.onerror = (error) => { 
            addChatMessage("WebSocket error. Check console. If page HTTPS, WS must be WSS.", "error"); 
            console.error("WebSocket error object:", error);
            showThinking(false);
        };
        ws.onclose = (event) => { 
            let reason = "";
            if (event.code) reason += `Code: ${event.code} `;
            if (event.reason) reason += `Reason: ${event.reason} `;
            if (event.wasClean) reason += `(Clean close) `; else reason += `(Unclean close) `;
            addChatMessage("WebSocket disconnected. " + reason + "Attempting to reconnect in 5s...", "info");
            console.log("WebSocket close event:", event);
            showThinking(false);
            setTimeout(connect, 5000);
        };
    }

    function addChatMessage(message, type) {
        const messageElement = document.createElement('div');
        messageElement.innerHTML = message; 
        messageElement.className = 'message ' + type;
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function sendMessage() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            const messageText = messageInput.value;
            if (messageText.trim() === "") return;
            addChatMessage("You: " + escapeHtml(messageText), "user");
            ws.send(messageText);
            messageInput.value = '';
            messageInput.style.height = 'auto'; // Reset height
            showThinking(true);
        } else {
            addChatMessage("WebSocket is not connected.", "error");
        }
    }
    
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    messageInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { 
            e.preventDefault(); 
            sendMessage(); 
        }
    });
    connect();
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    return HTML_TEST_PAGE
