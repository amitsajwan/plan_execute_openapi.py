# llm_config.py
import logging
import os
import json
from typing import Any, Tuple

# Attempt to import LLM provider libraries
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None # type: ignore 
    logging.warning("llm_config.py: langchain-google-genai not installed. Real Gemini LLMs will not be available.")

# Attempt to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("llm_config.py: .env file loaded if present.")
except ImportError:
    logging.warning("llm_config.py: python-dotenv not installed. Cannot load .env file for API keys.")

logger = logging.getLogger(__name__)

class PlaceholderLLM:
    """
    A dummy LLM class for fallback or testing when real LLMs are not available/configured.
    It simulates responses based on keywords in the prompt.
    """
    def __init__(self, name="PlaceholderLLM", delay=0.01): 
        self.name = name
        self.delay = delay
        logger.warning(f"Initialized PlaceholderLLM: {self.name}. This is a fallback or for testing.")

    def _generate_simulated_response(self, prompt_str: str) -> str:
        """Generates a simulated LLM response based on prompt content."""
        prompt_lower = prompt_str.lower()

        if "summarize this openapi specification" in prompt_lower:
            return "This is a simulated API summary from PlaceholderLLM."
        
        if "design an api execution graph" in prompt_lower or \
           "critique and refine the following api execution graph" in prompt_lower:
            graph_data = {
                "nodes": [], "edges": [], 
                "description": "Placeholder graph: Default workflow by PlaceholderLLM.",
                "refinement_summary": "Initial version by PlaceholderLLM."
            }
            if "design an api execution graph" in prompt_lower and \
               "cannot generate graph: no apis identified" not in prompt_lower:
                graph_data["nodes"].append(
                    {"operationId": "placeholderOp1", "summary": "Placeholder Op 1", 
                     "description":"First placeholder step", 
                     "payload_description":"Req: data. Resp: result", "input_mappings":[]}
                )
            if "critique and refine" in prompt_lower:
                 graph_data["description"] = "Refined placeholder graph."
                 graph_data["refinement_summary"] = "Placeholder made some refinements."
                 if not graph_data["nodes"]: 
                     graph_data["nodes"].append(
                         {"operationId": "placeholderOp1Refined", "summary": "Refined Placeholder Op 1", 
                          "description":"Refined first step", 
                          "payload_description":"Req: data. Resp: result", "input_mappings":[]}
                     )
            return json.dumps(graph_data)

        if "classify the user's intent" in prompt_lower: 
            if "focus on" in prompt_lower or "what if" in prompt_lower: return "interactive_query_planner"
            return "answer_openapi_query"
        if "plans internal actions" in prompt_lower: 
            return json.dumps({
                "user_query_understanding": "User wants info on 'someItem' (Placeholder).",
                "interactive_action_plan": [{"action_name": "answer_query_directly", "action_params": {"query_for_synthesizer": "About someItem"}, "description":"Answer about someItem"}]})
        if "determine the user's high-level intent" in prompt_lower: # Fallback router prompt
            if "graph" in prompt_lower: return "describe_graph"
            return "answer_openapi_query"
            
        return f"Simulated response from {self.name} for: {prompt_str[:50]}..."

    def invoke(self, prompt: Any, **kwargs) -> Any:
        """Simulates the LLM's invoke method."""
        prompt_str = str(prompt); 
        # logger.debug(f"{self.name} received prompt (first 100 chars): {prompt_str[:100]}...")
        response_text = self._generate_simulated_response(prompt_str)
        # logger.debug(f"{self.name} generated response (first 100 chars): {response_text[:100]}...")
        
        # LangChain models typically return a message object with a 'content' attribute.
        class ContentWrapper: 
            def __init__(self, text): self.content = text
        return ContentWrapper(response_text)

    async def ainvoke(self, prompt: Any, **kwargs) -> Any:
        """Simulates the LLM's asynchronous invoke method."""
        import asyncio # Local import for async behavior
        await asyncio.sleep(self.delay) 
        return self.invoke(prompt, **kwargs)


def initialize_llms() -> Tuple[Any, Any]:
    """
    Initializes and returns the router and worker LLMs.
    Attempts to use Google Gemini, falls back to PlaceholderLLM if not configured.
    """
    logger.info("Attempting to initialize LLMs (Google Gemini)...")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    router_llm = None
    worker_llm = None

    if google_api_key and ChatGoogleGenerativeAI:
        try:
            router_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest", 
                temperature=0,
                google_api_key=google_api_key,
                convert_system_message_to_human=True 
            )
            logger.info("Router LLM (Gemini Flash) initialized successfully.")

            worker_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest", 
                temperature=0.1, 
                google_api_key=google_api_key,
                convert_system_message_to_human=True
            )
            logger.info("Worker LLM (Gemini 1.5 Pro) initialized successfully.")

        except Exception as e: 
            logger.error(f"Failed to initialize Google Gemini LLMs: {e}", exc_info=True)
            # Ensure fallback if one initializes but the other fails
            router_llm = None 
            worker_llm = None
    elif not ChatGoogleGenerativeAI:
        logger.warning("langchain-google-genai library not found. Cannot initialize Gemini LLMs.")
    else: # API key not found
        logger.warning("GOOGLE_API_KEY environment variable not set. Cannot initialize Gemini LLMs.")

    # Fallback to PlaceholderLLMs if real ones couldn't be initialized
    if router_llm is None:
        logger.warning("Using PlaceholderLLM for Router LLM.")
        router_llm = PlaceholderLLM(name="RouterLLM_Placeholder")
    if worker_llm is None:
        logger.warning("Using PlaceholderLLM for Worker LLM.")
        worker_llm = PlaceholderLLM(name="WorkerLLM_Placeholder")
    
    # Final check for necessary methods (invoke/ainvoke)
    # This is more for ensuring the LLM objects are compatible with LangChain/LangGraph
    if not hasattr(router_llm, 'invoke') or not hasattr(worker_llm, 'invoke'):
        # This should ideally not happen if using LangChain models or the PlaceholderLLM
        raise TypeError("One or both initialized LLMs are missing the 'invoke' method.")
    if not hasattr(router_llm, 'ainvoke') or not hasattr(worker_llm, 'ainvoke'):
        logger.warning("One or both LLMs do not have 'ainvoke'. Adding dummy ainvoke for LangGraph compatibility if needed.")
        # LangChain base models usually have ainvoke if invoke is present.
        # PlaceholderLLM has it.
        if not hasattr(router_llm, 'ainvoke'): # Should not be hit with current PlaceholderLLM
            setattr(router_llm, 'ainvoke', router_llm.invoke) # type: ignore
        if not hasattr(worker_llm, 'ainvoke'): # Should not be hit
            setattr(worker_llm, 'ainvoke', worker_llm.invoke) # type: ignore

    logger.info("LLM client initialization process finished.")
    return router_llm, worker_llm
