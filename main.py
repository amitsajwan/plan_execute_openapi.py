# filename: main.py

import logging
import uuid
import json
import os
import sys # Import sys to check Python version
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, File, UploadFile, Form # Added UploadFile, Form for potential file upload
from fastapi.responses import HTMLResponse, JSONResponse # Added JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.checkpoint.memory import MemorySaver # Assuming you are still using MemorySaver
# If using a different checkpointer (e.g., SQL), import it here

# Import necessary components from your project files
from graph import build_graph # Imports build_graph function
from models import BotState # Imports BotState model
# Import utils for cache closing on shutdown if needed
# try:
#     import utils
# except ImportError:
#     utils = None
#     logging.warning("Could not import utils. Cache will not be explicitly closed on shutdown.")


# --- Basic Logging Setup ---
# Configure logging to output to console
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM Initialization (REPLACE PLACEHOLDER) ---
# Use python-dotenv to load environment variables from a .env file
try:
    from dotenv import load_dotenv
    load_dotenv() # Load environment variables from .env file
    logger.info(".env file loaded.")
except ImportError:
    logger.warning("python-dotenv not installed. Cannot load environment variables from .env file.")
except Exception as e:
    logger.warning(f"Error loading .env file: {e}")


def initialize_llms():
    """
    Initializes and returns the router and worker LLM instances.
    Replace this with your actual LLM setup using your preferred LLM provider.
    Ensure API keys are handled securely (e.g., environment variables).
    """
    logger.info("Initializing LLMs...")

    # Example using environment variables for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # google_api_key = os.getenv("GOOGLE_API_KEY") # Example for Google

    router_llm = None
    worker_llm = None

    # --- START OF REAL LLM EXAMPLE (OpenAI) ---
    # UNCOMMENT AND REPLACE WITH YOUR ACTUAL LLM INSTANTIATION
    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            logger.info("Attempting to initialize ChatOpenAI.")
            # Ensure model names are appropriate for your OpenAI account/tier
            # gpt-3.5-turbo is usually sufficient and cheaper for routing/simple tasks
            router_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
            # gpt-4o-mini or gpt-4-turbo might be better for complex tasks like planning/simulating
            worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=openai_api_key)
            logger.info("ChatOpenAI LLMs initialized.")
        except ImportError:
             logger.error("langchain-openai not installed. Cannot initialize ChatOpenAI.")
        except Exception as e:
             logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
    else:
         logger.warning("OPENAI_API_KEY environment variable not set. Cannot initialize ChatOpenAI.")
    # --- END REAL LLM EXAMPLE ---


    # --- START OF PLACEHOLDER FALLBACK ---
    # This block provides dummy LLMs if real ones fail to initialize.
    # REMOVE THIS BLOCK IN YOUR ACTUAL PRODUCTION IMPLEMENTATION if you require real LLMs.
    if router_llm is None or worker_llm is None:
        logger.warning("Using PlaceholderLLMs due to missing API key or initialization failure.")
        # Define a simple placeholder class that mimics the LLM interface expected by utils.llm_call_helper
        class PlaceholderLLM:
            def __init__(self, name="PlaceholderLLM"): self.name = name
            # Simulate invoke returning an object with .content or a string directly
            def invoke(self, prompt: Any, **kwargs) -> Any:
                logger.warning(f"Using {self.name}. This needs replacement with a real LLM.")
                prompt_str = str(prompt)

                # Simple simulation based on prompt keywords
                if "Determine the user's high-level intent" in prompt_str:
                    # Simulate router response (string intent) - needs to match router's logic
                    if any(sig in prompt_str for sig in ['"openapi":', 'swagger:', '{', '-']):
                         return "parse_openapi_spec"
                    elif "list apis" in prompt_str.lower() or "endpoints" in prompt_str.lower():
                         return "answer_openapi_query"
                    elif "describe graph" in prompt_str.lower() or "show graph" in prompt_str.lower():
                         return "describe_graph"
                    elif "get graph json" in prompt_str.lower() or "show graph json" in prompt_str.lower():
                         return "get_graph_json"
                    elif "generate graph" in prompt_str.lower():
                         return "generate_execution_graph"
                    elif "plan execution" in prompt_str.lower() or "create plan" in prompt_str.lower() or "workflow for" in prompt_str.lower():
                         return "plan_execution"
                    elif "add edge" in prompt_str.lower() or "delete edge" in prompt_str.lower():
                         return "update_graph"
                    else:
                         return "handle_unknown"

                # Simulate core_logic responses (text or JSON string)
                if "Create a concise but comprehensive summary" in prompt_str:
                    return "Simulating schema summary: This is a placeholder summary for the API."
                if "Identify relevant API operations" in prompt_str:
                     # Simulate JSON output for identified APIs
                     return json.dumps([{"operationId": "getPlaceholder", "method": "GET", "path": "/placeholder", "summary": "Get placeholder data"}])
                if "Generate a clear, natural language description of an EXAMPLE payload" in prompt_str:
                     # Simulate payload description - try to extract operationId from prompt
                     match = re.search(r"Operation ID: (\w+)", prompt_str)
                     op_id = match.group(1) if match else "an API"
                     return f"Simulating payload description for {op_id}: Requires an optional 'id' parameter."
                if "Generate a description of an API execution workflow graph" in prompt_str:
                     # Simulate valid GraphOutput JSON
                     simulated_graph = {
                         "nodes": [
                             {"operationId": "createPlaceholder", "display_name": "create_step", "summary": "Create placeholder", "payload_description": "Requires data in body.", "input_mappings": []},
                             {"operationId": "getPlaceholder", "display_name": "get_step", "summary": "Get placeholder", "payload_description": "Requires ID.", "input_mappings": [{"source_operation_id": "create_step", "source_data_path": "$.id", "target_parameter_name": "id", "target_parameter_in": "path"}]}
                         ],
                         "edges": [
                              {"from_node": "create_step", "to_node": "get_step", "description": "Uses ID from creation."}
                         ],
                         "description": "Simulated workflow to create and then get a placeholder."
                     }
                     return json.dumps(simulated_graph)
                if "Provide a concise, natural language description of the following API execution workflow graph" in prompt_str:
                     # Simulate graph description based on prompt content
                     return "Simulating graph description: This workflow creates a placeholder resource and then retrieves it using the ID obtained from the creation step."
                if "Answer the user's question based on the available information" in prompt_str:
                    # Simulate answering a query - needs to *look* at context in prompt
                    if "list apis" in prompt_str or "endpoints" in prompt_str:
                        return "Simulating answer: Available APIs include getPlaceholder, createPlaceholder."
                    else:
                        return "Simulating answer: I can answer questions about the loaded API or the simulated graph/plan."
                if "create a simple, sequential plan" in prompt_str:
                     # Simulate plan JSON
                     return json.dumps({"plan": ["createPlaceholder", "getPlaceholder"], "description": "Simulated plan to create and get placeholder."})
                if "Simulate the response of an API call" in prompt_str:
                     # Simulate API response JSON - try to extract operationId
                     match = re.search(r"operationId: (\w+)", prompt_str)
                     op_id = match.group(1) if match else "unknown_op"
                     if op_id == "createPlaceholder":
                         return json.dumps({"id": "sim-123", "status": "created"})
                     elif op_id == "getPlaceholder":
                         # Simulate using previous results if available in prompt context
                         if "sim-123" in prompt_str: # Crude check for previous ID
                              return json.dumps({"id": "sim-123", "data": "simulated data for 123"})
                         else:
                              return json.dumps({"id": "sim-abc", "data": "simulated data"})
                     else:
                         return json.dumps({"status": "simulated success", "operation": op_id})


                # Default response for other LLM calls
                return "Simulating LLM response for an unspecified task."

            # For LangChain integration, ensure other required methods (_call, ainvoke, etc.) are present if needed
            # For async streaming, ainvoke is necessary.
            # Since LangGraph astream is used, ainvoke is important for non-blocking behavior.
            async def ainvoke(self, prompt: Any, **kwargs) -> Any:
                 # Simulate async behavior
                 import asyncio
                 await asyncio.sleep(0.05) # Small delay
                 # Call the sync invoke method
                 sync_result = self.invoke(prompt, **kwargs)
                 return sync_result # Return the simulated LLM text output


        # Instantiate PlaceholderLLMs if real ones couldn't be initialized
        if router_llm is None:
             router_llm = PlaceholderLLM("RouterLLM")
        if worker_llm is None:
             worker_llm = PlaceholderLLM("WorkerLLM")

    # --- END OF PLACEHOLDER FALLBACK ---


    # Final check to ensure LLMs were initialized (either real or placeholder)
    if router_llm is None or worker_llm is None:
         raise RuntimeError("Failed to initialize both real and placeholder LLMs.")

    # Validate that the LLMs have the required methods for the graph (invoke/ainvoke)
    if not hasattr(router_llm, 'invoke') or not hasattr(worker_llm, 'invoke'):
        raise TypeError("Initialized LLMs must have an 'invoke' method.")
    # If using astream, they must also have ainvoke
    if not hasattr(router_llm, 'ainvoke') or not hasattr(worker_llm, 'ainvoke'):
         logger.warning("LLMs do not have 'ainvoke' method. Adding dummy ainvoke for compatibility with astream.")
         # Add dummy ainvoke if missing to prevent errors with astream
         if not hasattr(router_llm, 'ainvoke'): router_llm.ainvoke = lambda p, **k: router_llm.invoke(p, **k)
         if not hasattr(worker_llm, 'ainvoke'): worker_llm.ainvoke = lambda p, **k: worker_llm.invoke(p, **k)


    logger.info("LLM clients initialized.")
    return router_llm, worker_llm

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Serve Static Files and Templates (Optional) ---
# Serve files from a 'static' directory (e.g., for HTML, CSS, JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# Set up templates directory (e.g., for index.html)
# templates = Jinja2Templates(directory="templates")

# --- Global LangGraph Instance ---
# Initialize LLMs and build the graph once on startup
router_llm_instance: Any = None
worker_llm_instance: Any = None
langgraph_app: Any = None
# Use MemorySaver for in-memory state persistence per session.
# Replace with a persistent checkpointer (e.g., SQL) for production.
checkpointer = MemorySaver()

@app.on_event("startup")
async def startup_event():
    """Initializes LLMs and builds the LangGraph application on FastAPI startup."""
    global router_llm_instance, worker_llm_instance, langgraph_app
    logger.info("FastAPI startup event: Initializing LLMs and building graph...")
    try:
        router_llm_instance, worker_llm_instance = initialize_llms()
        # Pass the checkpointer when compiling the app to enable state persistence
        langgraph_app = build_graph(router_llm=router_llm_instance, worker_llm=worker_llm_instance).compile(checkpointer=checkpointer)
        logger.info("LangGraph application compiled successfully with MemorySaver checkpointer.")
    except Exception as e:
        logger.critical(f"Failed to initialize LLMs or build graph on startup: {e}", exc_info=True)
        # Depending on your needs, you might want to raise the exception
        # or set a flag to indicate the service is not fully operational.
        # For this example, we'll just log and continue, but requests might fail.
        langgraph_app = None # Ensure app is None if build fails

@app.on_event("shutdown")
async def shutdown_event():
    """Cleans up resources on FastAPI shutdown (if needed)."""
    logger.info("FastAPI shutdown event.")
    # If using diskcache, consider explicitly closing it if needed (MemorySaver doesn't need close)
    # if 'utils' in sys.modules and hasattr(utils, 'SCHEMA_CACHE') and utils.SCHEMA_CACHE:
    #      try:
    #          utils.SCHEMA_CACHE.close()
    #          logger.info("Schema cache closed.")
    #      except Exception as e:
    #          logger.error(f"Error closing schema cache: {e}")


# --- WebSocket Endpoint ---
@app.websocket("/ws/submit_openapi")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections for OpenAPI analysis."""
    await websocket.accept()
    logger.info(f"WebSocket accepted connection from {websocket.client.host}:{websocket.client.port}")

    # Generate a unique session ID for this WebSocket connection
    # This will be used as the thread_id for LangGraph's checkpointer
    # Can also try to get a user identifier if available in a real auth scenario
    session_id = str(uuid.uuid4())
    logger.info(f"Assigned session ID: {session_id}")

    # Provide initial welcome message
    await websocket.send_json({"type": "info", "content": f"Connected. Session ID: {session_id}. Please provide an OpenAPI spec (JSON/YAML) or ask a question."})

    if langgraph_app is None:
         error_msg = "Backend initialization failed. Cannot process requests."
         logger.error(error_msg)
         await websocket.send_json({"type": "error", "content": error_msg})
         await websocket.close(code=1011) # Internal Error
         return # Exit the handler if app failed to build

    try:
        while True:
            # Receive message from client (assuming text for simplicity for now)
            data = await websocket.receive_text()
            user_input = data.strip()

            if not user_input:
                logger.warning("Received empty message from client.")
                await websocket.send_json({"type": "warning", "content": "Received empty message. Please provide input."})
                continue

            logger.info(f"Received message for session {session_id}: '{user_input[:200]}...'") # Log snippet
            await websocket.send_json({"type": "info", "content": "Processing your request..."}) # Acknowledge receipt

            # Prepare input for LangGraph
            # The initial input to astream is the state object itself for nodes that expect state
            config = {"configurable": {"thread_id": session_id}}

            # Load the latest state for this thread_id from the checkpointer
            # The checkpointer's get method returns a dictionary representing the state, or {} if none exists
            thread_state_dict = checkpointer.get(config.get("configurable", {})).get("channel_values", {}) if checkpointer else {}

            # Create or load BotState
            loaded_state = None
            if thread_state_dict:
                # Try to load from checkpoint state dictionary using Pydantic model_validate
                try:
                    loaded_state = BotState.model_validate(thread_state_dict)
                    loaded_state.user_input = user_input # Update with current user input
                    # Clear previous intermediate response and next_step for the new turn
                    loaded_state.response = None
                    loaded_state.next_step = None
                    # Clear last sent intermediate message tracker from scratchpad for the new turn
                    if loaded_state.scratchpad and "last_sent_intermediate_response" in loaded_state.scratchpad:
                         del loaded_state.scratchpad["last_sent_intermediate_response"]

                    logger.debug(f"Loaded state for session {session_id} from checkpoint.")
                except ValidationError as e:
                    logger.error(f"Failed to validate loaded state for session {session_id}: {e}. Starting with new state.", exc_info=True)
                    # If validation fails, start fresh
                    loaded_state = BotState(session_id=session_id, user_input=user_input)
                except Exception as e:
                     logger.error(f"Unexpected error loading state for session {session_id}: {e}. Starting with new state.", exc_info=True)
                     loaded_state = BotState(session_id=session_id, user_input=user_input)
            else:
                # No existing state found for this thread_id, create new
                loaded_state = BotState(session_id=session_id, user_input=user_input)
                logger.debug(f"No checkpoint state found for session {session_id}. Starting with new state.")


            # The input to `astream` is the initial state object for the first node (router)
            current_state_input = loaded_state

            # Use astream for asynchronous streaming in FastAPI
            # stream_mode="values" yields the full state object (or dict representation) after each node
            # The graph nodes now return BotState objects, which astream yields.
            final_state_snapshot: Optional[BotState] = None # Keep track of the latest state object yielded

            try:
                # astream yields the state after each node completes
                async for intermediate_state_dict in langgraph_app.astream(current_state_input, config=config, stream_mode="values"):
                     # intermediate_state_dict is the dictionary representation of the BotState object
                     # returned by the node. We need to validate it back to a BotState model.
                     try:
                         intermediate_state = BotState.model_validate(intermediate_state_dict)
                         final_state_snapshot = intermediate_state # Keep track of the latest valid state

                         # Check for intermediate response messages set by nodes
                         # Only send if the response field has been updated and is not None
                         response_message = intermediate_state.response
                         # Use a scratchpad key to track the last sent intermediate response
                         # Note: This tracking in scratchpad might not be perfectly reliable with streaming
                         # as the scratchpad update might not be flushed immediately.
                         last_sent_response = intermediate_state.scratchpad.get("last_sent_intermediate_response_sent") # Use a different key

                         # Simple check to avoid sending the same message repeatedly within one turn
                         # This is imperfect with streaming but better than nothing.
                         # A more robust approach might involve tracking response hashes or sequence numbers.
                         if response_message and response_message != last_sent_response:
                             logger.debug(f"Sending intermediate message for session {session_id}: {response_message[:200]}...")
                             await websocket.send_json({"type": "intermediate", "content": response_message})
                             # Update scratchpad to mark this message as sent (imperfectly)
                             # This update might not be saved until the end of the turn.
                             intermediate_state.scratchpad["last_sent_intermediate_response_sent"] = response_message
                             # Note: The state object is passed by reference, so modifying intermediate_state
                             # here *does* modify the state object that will be potentially checkpointed.


                     except ValidationError as e:
                         logger.error(f"Failed to validate intermediate state from stream for session {session_id}: {e}. Skipping this state update.", exc_info=True)
                         await websocket.send_json({"type": "warning", "content": f"Received invalid state update from backend: {e}"})
                     except Exception as e:
                         logger.error(f"Unexpected error processing intermediate state for session {session_id}: {e}", exc_info=True)
                         await websocket.send_json({"type": "warning", "content": f"Unexpected error processing state update: {e}"})


                # After the stream finishes, process the final state snapshot
                if final_state_snapshot and isinstance(final_state_snapshot, BotState):
                     # The responder node should have put the final user-facing message here
                     final_response = final_state_snapshot.final_response

                     # Send the final response message
                     if final_response:
                         logger.info(f"Sending final response for session {session_id}: {final_response[:200]}...")
                         await websocket.send_json({"type": "final", "content": final_response})
                     else:
                         # Fallback if final_response isn't set (e.g., error before responder)
                         logger.warning(f"Graph execution finished for session {session_id}, but 'final_response' was empty.")
                         # Check if there was an intermediate response that wasn't marked final
                         fallback_response = final_state_snapshot.response if final_state_snapshot.response else "Processing finished, but no specific final result message was generated."
                         await websocket.send_json({"type": "warning", "content": fallback_response})

                     # Optionally send other final state info if needed, e.g., the graph JSON
                     # if final_state_snapshot.execution_graph:
                     #      try:
                     #          graph_json = final_state_snapshot.execution_graph.model_dump_json(indent=2)
                     #          await websocket.send_json({"type": "graph_json", "content": graph_json})
                     #      except Exception as e:
                     #          logger.error(f"Error sending final graph JSON for session {session_id}: {e}")
                     #          await websocket.send_json({"type": "warning", "content": f"Could not serialize final graph to JSON: {e}"})

                     # Clear the last sent intermediate response tracker from scratchpad for the next turn
                     if final_state_snapshot.scratchpad and "last_sent_intermediate_response_sent" in final_state_snapshot.scratchpad:
                          del final_state_snapshot.scratchpad["last_sent_intermediate_response_sent"]


                else:
                     logger.error(f"Graph execution finished for session {session_id} without a valid final state object.")
                     await websocket.send_json({"type": "error", "content": "Internal error: Failed to get final processing state."})


            except Exception as e:
                # Catch exceptions during graph execution (e.g., LLM call errors)
                logger.critical(f"Error during LangGraph execution for session {session_id}: {e}", exc_info=True)
                await websocket.send_json({"type": "error", "content": f"An error occurred during processing: {e}"})
                # Decide if you want to close the connection on error or continue
                # await websocket.close(code=1011) # Example: Close on internal error
                # break # Example: Break loop on error

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        # Catch any other exceptions in the WebSocket loop itself
        logger.critical(f"Unexpected error in WebSocket loop for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "content": f"An unexpected error occurred: {e}"})
        except:
            pass # Ignore errors sending error message
    finally:
        # Ensure WebSocket is closed if not already
        try:
            await websocket.close()
        except:
            pass # Ignore errors closing


# --- How to Run ---
# 1. Save the files: models.py, utils.py, core_logic.py, router.py, graph.py, main.py, requirements.txt
# 2. Install dependencies: pip install -r requirements.txt
# 3. Set your LLM API key in a .env file in the same directory (e.g., OPENAI_API_KEY=your_key_here)
# 4. Run the FastAPI application: uvicorn main:app --reload
# 5. Connect to the WebSocket endpoint ws://localhost:8000/ws/submit_openapi from a client.
#    You can use a simple HTML page with JavaScript or a WebSocket testing tool.
