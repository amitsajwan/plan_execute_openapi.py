# main.py
import logging
import uuid
import json
import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse # For a simple test page
from langgraph.checkpoint.memory import MemorySaver # In-memory checkpointer

from models import BotState # Pydantic model for state
from graph import build_graph # Graph building function
from pydantic import ValidationError # For validating loaded state

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- LLM Initialization ---
# Placeholder: Replace with your actual LLM client setup (e.g., OpenAI, Anthropic)
# Ensure API keys are handled securely via environment variables.
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info(".env file loaded if present.")
except ImportError:
    logger.warning("python-dotenv not installed. Cannot load .env file.")

# Dummy LLM for placeholder - REPLACE THIS
class PlaceholderLLM:
    def __init__(self, name="PlaceholderLLM", delay=0.1):
        self.name = name
        self.delay = delay
        logger.warning(f"Initialized {self.name}. REPLACE with a real LLM.")

    def _generate_simulated_response(self, prompt_str: str) -> str:
        # Simulate responses based on keywords in prompt
        if "Summarize this OpenAPI specification" in prompt_str:
            return "This is a simulated API summary. It details several interesting endpoints for resource management."
        if "Identify all API operation details" in prompt_str: # Not directly called by LLM, but for completeness
            return json.dumps([{"operationId": "get_items", "summary": "Get all items"}])
        if "Generate a concise, natural language description of" in prompt_str and "Request:" in prompt_str: # Payload desc
             return "Request: Requires an 'item_id' (string). Response: Returns item details including 'name' and 'price'."
        if "design an API execution graph" in prompt_str.lower(): # Graph generation
            return json.dumps({
                "nodes": [{"operationId": "createItem", "summary": "Create an item", "description":"Initial step", "payload_description":"Req: name. Resp: id", "input_mappings":[]}],
                "edges": [],
                "description": "Simulated graph: Create an item."
            })
        if "Critique and refine the following API execution graph" in prompt_str: # Graph refinement
             return json.dumps({
                "nodes": [{"operationId": "createItem", "summary": "Create an item (refined)", "description":"Initial step (refined)", "payload_description":"Req: name (string). Resp: id (string)", "input_mappings":[]}],
                "edges": [],
                "description": "Refined graph: Create an item with more detail.",
                "refinement_summary": "Added more detail to payload description."
            })
        if "Describe the following API execution graph" in prompt_str:
            return "This workflow involves creating an item and then perhaps listing items."
        if "Answer the user's question based on the provided API information" in prompt_str: # General Q&A
            if "list all" in prompt_str.lower(): return "Available APIs: createItem, getItem, listItems."
            return "This is a simulated answer to your query about the API."
        if "Classify the user's intent" in prompt_str: # Router classification
            if "focus on apple" in prompt_str.lower() or "what if" in prompt_str.lower(): return "interactive_query_planner"
            return "answer_openapi_query" # Default for router
        if "plans internal actions to respond to a user's query" in prompt_str: # Interactive planner
            return json.dumps({
                "user_query_understanding": "User wants to know more about 'createItem'.",
                "interactive_action_plan": [{"action_name": "answer_query_directly", "action_params": {"query_for_synthesizer": "Tell me about createItem API"}, "description":"Answer about createItem"}]
            })
        if "synthesize a final answer for the user" in prompt_str: # Interactive synthesizer
            return "Final synthesized answer: The 'createItem' API is used to add new items to the system."
        
        # Fallback for router's general intent prompt
        if "Determine the user's high-level intent from the list" in prompt_str:
            if "graph" in prompt_str.lower(): return "describe_graph"
            if "plan" in prompt_str.lower(): return "_generate_execution_graph" # Assuming a goal is implied
            return "answer_openapi_query"

        return f"Simulated response from {self.name} for prompt starting with: {prompt_str[:60]}..."

    def invoke(self, prompt: Any, **kwargs) -> Any:
        prompt_str = str(prompt) # Handle various prompt types by converting to string
        logger.debug(f"{self.name} received prompt: {prompt_str[:200]}...")
        # Simulate LLM processing delay
        # import time; time.sleep(self.delay) 
        
        # For LangChain compatibility, often an object with a 'content' attribute is expected
        class ContentWrapper:
            def __init__(self, text): self.content = text
        
        return ContentWrapper(self._generate_simulated_response(prompt_str))

    async def ainvoke(self, prompt: Any, **kwargs) -> Any: # For astream
        # Simulate async behavior for astream
        import asyncio
        await asyncio.sleep(self.delay / 2) # Shorter delay for async
        return self.invoke(prompt, **kwargs)


def initialize_llms() -> Tuple[Any, Any]:
    """Initializes and returns router and worker LLMs."""
    logger.info("Initializing LLMs...")
    # Replace with your actual LLM client instantiation
    # Example: from langchain_openai import ChatOpenAI
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # if not openai_api_key:
    #     logger.error("OPENAI_API_KEY not found in environment variables.")
    #     raise ValueError("OPENAI_API_KEY is required.")
    # router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    # worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=openai_api_key) # Or gpt-4-turbo

    router_llm = PlaceholderLLM(name="RouterLLM_Placeholder")
    worker_llm = PlaceholderLLM(name="WorkerLLM_Placeholder")
    
    logger.info("LLM clients initialized (using placeholders).")
    return router_llm, worker_llm

# --- FastAPI App and LangGraph Setup ---
app = FastAPI()
langgraph_app: Optional[Any] = None # Compiled LangGraph application
checkpointer = MemorySaver() # In-memory state persistence per session

@app.on_event("startup")
async def startup_event():
    global langgraph_app
    logger.info("FastAPI startup: Initializing LLMs and building LangGraph app...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        # Pass checkpointer during compilation for state persistence
        langgraph_app = build_graph(
            router_llm=router_llm_instance, 
            worker_llm=worker_llm_instance
        ).compile(checkpointer=checkpointer)
        logger.info("LangGraph application compiled successfully with MemorySaver.")
    except Exception as e:
        logger.critical(f"Failed to initialize/build graph on startup: {e}", exc_info=True)
        langgraph_app = None # Ensure app is None if build fails

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI shutdown event.")
    if hasattr(checkpointer, 'close'): # If checkpointer needs explicit closing
        try:
            checkpointer.close() # type: ignore
            logger.info("Checkpointer closed.")
        except Exception as e:
            logger.error(f"Error closing checkpointer: {e}")
    # Close schema cache if it was initialized
    from utils import SCHEMA_CACHE
    if SCHEMA_CACHE:
        try:
            SCHEMA_CACHE.close()
            logger.info("Schema cache closed.")
        except Exception as e:
            logger.error(f"Error closing schema cache: {e}")


# --- WebSocket Endpoint ---
@app.websocket("/ws/openapi_agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    logger.info(f"WebSocket accepted for session: {session_id}")

    if langgraph_app is None:
        await websocket.send_json({"type": "error", "content": "Backend agent not initialized. Please check server logs."})
        await websocket.close(code=1011)
        return

    await websocket.send_json({"type": "info", "session_id": session_id, "content": "Connected. Please provide OpenAPI spec (JSON/YAML) or ask a question."})

    try:
        while True:
            user_input = await websocket.receive_text()
            user_input = user_input.strip()
            logger.info(f"[{session_id}] Received: {user_input[:200]}...")

            if not user_input:
                await websocket.send_json({"type": "warning", "content": "Empty message received."})
                continue

            await websocket.send_json({"type": "status", "content": "Processing your request..."})

            # Config for LangGraph stream, including thread_id for state persistence
            config = {"configurable": {"thread_id": session_id}}
            
            # Load existing state or create new
            # LangGraph's MemorySaver checkpointer handles loading state internally based on thread_id
            # The input to astream should be the initial values for the BotState for this turn.
            # We need to construct the initial BotState for the graph's input.
            # The checkpointer.get(config) would give us the *full checkpoint object*,
            # from which we'd extract the BotState values if needed, but LangGraph
            # handles this internally when `input` is provided to `astream`.
            # So, we just need to prepare the *input for the first node* (router).
            # The router expects a BotState object.
            
            # Get current state from checkpointer to update it for the new turn
            current_checkpoint = checkpointer.get(config)
            initial_state_for_turn: BotState
            if current_checkpoint:
                try:
                    # The actual BotState is typically under 'channel_values' if using default LangGraph config
                    # or directly if the checkpoint is just the BotState.
                    # For MemorySaver, it's often {'channel_values': {'__root__': BotState_dict}}
                    # or {'channel_values': BotState_dict}
                    # This depends on how StateGraph is set up and if it's a "single channel" graph.
                    # Assuming BotState is the root of the state dictionary stored by MemorySaver.
                    # Let's assume the checkpointed value *is* the BotState dictionary.
                    # This might need adjustment based on exact checkpointer behavior with StateGraph(BotState).
                    
                    # A common pattern is that the state is under a key like '__root__' or the state class name
                    # if the graph's state is a single Pydantic model.
                    # For StateGraph(BotState), the values are usually directly the BotState fields.
                    
                    # Let's try to load state assuming it's a dict of BotState fields
                    state_dict_from_checkpoint = current_checkpoint.get("channel_values", {})
                    if not state_dict_from_checkpoint and isinstance(current_checkpoint, dict): # If channel_values is empty, maybe it's the dict itself
                        state_dict_from_checkpoint = current_checkpoint

                    if "__root__" in state_dict_from_checkpoint: # Handle if Pydantic model was saved with a root key
                        state_dict_from_checkpoint = state_dict_from_checkpoint["__root__"]

                    initial_state_for_turn = BotState.model_validate(state_dict_from_checkpoint)
                    initial_state_for_turn.user_input = user_input
                    # Clear previous turn's intermediate response and next_step
                    initial_state_for_turn.response = None
                    initial_state_for_turn.final_response = "" # Clear previous final response
                    initial_state_for_turn.next_step = None
                    initial_state_for_turn.intent = None # Router will set this
                    logger.debug(f"[{session_id}] Loaded and updated state from checkpoint.")

                except (ValidationError, TypeError, AttributeError) as e:
                    logger.warning(f"[{session_id}] Failed to validate/load state from checkpoint: {e}. Starting fresh. Checkpoint was: {current_checkpoint}")
                    initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)
            else:
                logger.debug(f"[{session_id}] No checkpoint found. Starting with new state.")
                initial_state_for_turn = BotState(session_id=session_id, user_input=user_input)


            final_state_snapshot: Optional[BotState] = None
            try:
                # Stream mode "values" yields the full state object (or its dict representation) after each node
                async for stream_event in langgraph_app.astream_events(initial_state_for_turn, config=config, version="v1"):
                    event_type = stream_event["event"]
                    event_data = stream_event["data"]
                    # event_tags = stream_event.get("tags", [])
                    # event_name = stream_event.get("name", "unknown_node")

                    if event_type == "on_chat_model_stream" and "chunk" in event_data: # LLM token streaming
                        # token = event_data["chunk"].content if hasattr(event_data["chunk"], 'content') else str(event_data["chunk"])
                        # await websocket.send_json({"type": "token", "content": token})
                        pass # For now, we send full intermediate messages, not live tokens

                    elif event_type == "on_chain_end" or event_type == "on_tool_end": # After a node (chain/tool) finishes
                        # The 'outputs' in event_data for on_chain_end (if it's a node)
                        # should contain the state object returned by that node.
                        node_output_state_dict = event_data.get("output") # LangGraph v0.1+ uses 'output'
                        if not node_output_state_dict and "outputs" in event_data : # Older LangGraph might use 'outputs'
                             node_output_state_dict = event_data.get("outputs")


                        if node_output_state_dict and isinstance(node_output_state_dict, dict):
                            try:
                                # The output might be the state directly or nested.
                                # If StateGraph(BotState), output is usually BotState dict.
                                current_node_state = BotState.model_validate(node_output_state_dict)
                                final_state_snapshot = current_node_state # Keep track of latest valid state

                                # Send intermediate response if set by the node
                                if current_node_state.response:
                                    logger.debug(f"[{session_id}] Sending intermediate: {current_node_state.response[:100]}...")
                                    await websocket.send_json({"type": "intermediate", "content": current_node_state.response})
                                    # To avoid resending, the node should clear its own .response after setting it,
                                    # or the responder should be the only one setting final_response.
                                    # For now, we rely on nodes to set state.response for intermediate,
                                    # and responder to set state.final_response.
                            except ValidationError as e:
                                logger.error(f"[{session_id}] Failed to validate state from node output: {e}. Data: {node_output_state_dict}")
                            except Exception as e_inner:
                                logger.error(f"[{session_id}] Error processing node output: {e_inner}. Data: {node_output_state_dict}", exc_info=True)
                
                # After stream is fully processed, the final state is in `final_state_snapshot`
                # (which should be the state after the 'responder' node)
                if final_state_snapshot and final_state_snapshot.final_response:
                    logger.info(f"[{session_id}] Sending final response: {final_state_snapshot.final_response[:100]}...")
                    await websocket.send_json({"type": "final", "content": final_state_snapshot.final_response})
                elif final_state_snapshot and final_state_snapshot.response : # Fallback if responder didn't set final_response but last node set response
                     logger.warning(f"[{session_id}] No final_response from responder, using last intermediate response.")
                     await websocket.send_json({"type": "final", "content": final_state_snapshot.response })
                else:
                    logger.error(f"[{session_id}] Graph execution finished but no final_response or response in the last state.")
                    await websocket.send_json({"type": "error", "content": "Processing finished, but no final message was generated."})

            except Exception as e_graph:
                logger.critical(f"[{session_id}] Error during LangGraph execution: {e_graph}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"An error occurred: {str(e_graph)}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e_outer:
        logger.critical(f"[{session_id}] Unexpected error in WebSocket handler: {e_outer}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": f"A critical server error occurred: {str(e_outer)}"})
        except: pass # Ignore if cannot send error
    finally:
        logger.info(f"[{session_id}] Closing WebSocket connection.")
        try:
            await websocket.close()
        except: pass

# --- Simple HTML Test Page (Optional) ---
HTML_TEST_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>OpenAPI Agent Test</title>
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
    <h1>OpenAPI Agent Test</h1>
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
            ws = new WebSocket("ws://" + location.host + "/ws/openapi_agent");

            ws.onopen = () => {
                addMessage("WebSocket connected.", "info");
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                let content = data.content;
                if (typeof content === 'object') { // Pretty print if content is object (e.g. graph JSON)
                    content = '<pre>' + JSON.stringify(content, null, 2) + '</pre>';
                } else { // Escape HTML in string content
                    content = content.replace(/</g, "&lt;").replace(/>/g, "&gt;");
                     // Basic markdown for code blocks
                    content = content.replace(/```json\\n([\s\S]*?)\\n```/g, '<pre>$1</pre>');
                    content = content.replace(/```\\n([\s\S]*?)\\n```/g, '<pre>$1</pre>');
                }
                addMessage(`Agent (${data.type || 'message'}): ${content}`, data.type || "agent");
            };

            ws.onerror = (error) => {
                addMessage("WebSocket error: " + error, "error");
                console.error("WebSocket error:", error);
            };

            ws.onclose = () => {
                addMessage("WebSocket disconnected. Attempting to reconnect in 5s...", "info");
                setTimeout(connect, 5000); // Try to reconnect
            };
        }

        function addMessage(message, type) {
            const p = document.createElement('p');
            p.innerHTML = message; // Use innerHTML to render pre tags if any
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
            } else {
                addMessage("WebSocket is not connected.", "error");
            }
        }
        
        messageInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent new line
                sendMessage();
            }
        });

        connect(); // Initial connection
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    return HTML_TEST_PAGE


# --- How to Run (Example using Uvicorn) ---
# 1. Ensure all Python files (models.py, utils.py, core_logic.py, router.py, graph.py, main.py) are in the same directory.
# 2. Create a 'requirements.txt' file (see immersive document for its content).
# 3. Install dependencies: pip install -r requirements.txt
# 4. Create a '.env' file for API keys if using real LLMs (e.g., OPENAI_API_KEY=your_key).
# 5. Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
# 6. Open a browser to http://localhost:8000 for the test UI.
