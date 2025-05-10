# filename: utils.py
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, ValidationError
import diskcache # Import diskcache

# Assuming models.py defines GraphOutput, Node, Edge, BaseModel
# Need to import BaseModel if expected_model type hint is used directly
# from pydantic import BaseModel
# Assuming models.py is available
try:
    # Import necessary models and BaseModel from your models.py
    from models import GraphOutput, Node, Edge, BaseModel
except ImportError:
    # Define dummy classes if models.py is not found, to allow utils.py to load
    # This is primarily for linting/basic checks if utils is run standalone
    logger.warning("Could not import models from models.py. Using dummy classes for basic type hinting.")
    class GraphOutput:
        def __init__(self, nodes=None, edges=None):
            self.nodes = nodes or []
            self.edges = edges or []
    class Node:
        def __init__(self, operationId="dummy", display_name=None):
            self.operationId = operationId
            self.display_name = display_name
        @property
        def effective_id(self):
            return self.display_name if self.display_name else self.operationId
    class Edge:
         def __init__(self, from_node, to_node):
             self.from_node = from_node
             self.to_node = to_node
         def __hash__(self): return hash((self.from_node, self.to_node))
         def __eq__(self, other):
             if not isinstance(other, Edge): return False
             return self.from_node == other.from_node and self.to_node == other.to_node
    class BaseModel: # Dummy for type hint if needed
         @classmethod
         def model_validate(cls, data): return data
         def model_dump(self): return self.__dict__
         def model_dump_json(self, indent=None): return json.dumps(self.__dict__, indent=indent)


# Module-level logger
# Configure basic logging if not already configured by the main app
if not logging.getLogger('').handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Persistent Caching Setup ---
# Use a dedicated cache directory in the current working directory or a temporary one
# Ensure the directory exists
CACHE_DIR = os.path.join(os.getcwd(), ".openapi_cache")
os.makedirs(CACHE_DIR, exist_ok=True) # Create directory if it doesn't exist

SCHEMA_CACHE = None
try:
    # Use a context manager or explicit close in a real application
    SCHEMA_CACHE = diskcache.Cache(CACHE_DIR)
    logger.info(f"Initialized persistent schema cache at: {CACHE_DIR}")
except Exception as e:
    logger.error(f"Failed to initialize disk cache at {CACHE_DIR}: {e}. Caching will not work.", exc_info=True)
    SCHEMA_CACHE = None # Disable cache if initialization fails

def get_cache_key(spec_text: str) -> str:
    """Generates a cache key based on the hash of the spec text."""
    # Use a more robust hash like SHA-256 for longer keys less prone to collisions
    return hashlib.sha256(spec_text.encode('utf-8')).hexdigest()

def load_cached_schema(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads a parsed schema from the persistent cache if it exists."""
    if SCHEMA_CACHE is None:
        return None
    try:
        # Cache stores (key, value, expire) - get returns value or default
        schema = SCHEMA_CACHE.get(cache_key, default=None)
        if schema:
            logger.debug(f"Cache hit for key: {cache_key}")
            return schema
        else:
            return None
    except Exception as e:
        logger.error(f"Error loading schema from cache (key: {cache_key}): {e}", exc_info=True)
        return None

def save_schema_to_cache(cache_key: str, schema: Dict[str, Any]):
    """Saves a parsed schema to the persistent cache."""
    if SCHEMA_CACHE is None:
        return
    try:
        # Set with no expiration by default
        SCHEMA_CACHE.set(cache_key, schema)
        logger.debug(f"Saved schema to cache with key: {cache_key}")
    except Exception as e:
        logger.error(f"Error saving schema to cache (key: {cache_key}): {e}", exc_info=True)


# --- Graph Utilities ---

def check_for_cycles(graph: GraphOutput) -> Tuple[bool, str]:
    """
    Checks if the given execution graph is a Directed Acyclic Graph (DAG).
    Uses the effective_id of nodes for checks.
    Returns a tuple: (is_dag, cycle_message).
    """
    # Ensure graph and graph.nodes are valid before proceeding
    if not isinstance(graph, GraphOutput) or not isinstance(graph.nodes, list):
         logger.warning("Invalid graph object passed to check_for_cycles.")
         return False, "Invalid graph structure provided." # Treat invalid graph as potentially cyclic

    # Build adjacency list using effective_id from valid Node objects
    adj: Dict[str, List[str]] = {}
    node_ids = set()

    for node in graph.nodes:
         if isinstance(node, Node):
              node_id = node.effective_id
              node_ids.add(node_id)
              adj[node_id] = [] # Initialize adjacency list for this node

    if not node_ids:
        return True, "Graph is empty or has no valid nodes."


    if not isinstance(graph.edges, list):
         logger.warning("Graph edges attribute is not a list in check_for_cycles.")
         return False, "Invalid graph edges structure." # Treat invalid graph as potentially cyclic

    for edge in graph.edges:
        if isinstance(edge, Edge) and hasattr(edge, 'from_node') and hasattr(edge, 'to_node'):
            # Edge nodes must exist as effective_ids in the adjacency list keys
            # (which ensures they were valid Node objects)
            if edge.from_node in adj and edge.to_node in node_ids:
                 # Prevent adding duplicate edges if they exist in input
                 if edge.to_node not in adj[edge.from_node]:
                      adj[edge.from_node].append(edge.to_node)
            else:
                 logger.warning(f"Skipping edge in cycle check: Source or target node not found as a valid node effective_id: {edge.from_node} -> {edge.to_node}")
        else:
             logger.warning(f"Skipping invalid edge object in cycle check: {edge}")


    visited: Dict[str, bool] = {node_id: False for node_id in node_ids}
    recursion_stack: Dict[str, bool] = {node_id: False for node_id in node_ids}

    # Store the detected cycle path if found
    cycle_path: List[str] = []

    def dfs_cycle_check_recursive(node_id: str) -> bool:
        """Recursive helper for DFS cycle detection."""
        visited[node_id] = True
        recursion_stack[node_id] = True
        cycle_path.append(node_id) # Add to current path

        # Check neighbors
        for neighbor_id in adj.get(node_id, []):
            if not visited[neighbor_id]:
                if dfs_cycle_check_recursive(neighbor_id):
                    return True # Cycle found in deeper recursion
            elif recursion_stack[neighbor_id]:
                # Cycle detected: neighbor is already in the current recursion stack
                logger.error(f"Cycle detected: Edge from '{node_id}' to '{neighbor_id}' completes a cycle.")
                # Complete the cycle path
                try:
                    cycle_start_index = cycle_path.index(neighbor_id)
                    full_cycle = cycle_path[cycle_start_index:]
                    cycle_path[:] = full_cycle # Trim path to just the cycle
                except ValueError:
                    # Should not happen if neighbor_id is in recursion_stack, but as safeguard
                    logger.error(f"Error finding cycle start index for neighbor {neighbor_id} in path {cycle_path}")
                    cycle_path[:] = [f"Cycle involving {neighbor_id}"] # Fallback message
                return True

        # Remove node from recursion stack as we backtrack if no cycle found from this path
        recursion_stack[node_id] = False
        if cycle_path and cycle_path[-1] == node_id: # Only pop if this is the current path's end
             cycle_path.pop()
        return False

    # Iterate through all nodes to start DFS
    for node_id in node_ids:
        if not visited[node_id]:
            if dfs_cycle_check_recursive(node_id):
                cycle_message = f"Cycle detected: {' -> '.join(cycle_path)}"
                return False, cycle_message

    return True, "No cycles detected."


# --- LLM Call Helper ---

def llm_call_helper(llm: Any, prompt: Any) -> str:
    """
    Helper function to make an LLM call with basic logging and error handling.
    Accepts string or structured prompts (like lists of messages).

    Args:
        llm: The LLM instance with an 'invoke' method.
        prompt: The prompt (string or structured) to send to the LLM.

    Returns:
        The response content (usually text) from the LLM.

    Raises:
        Exception: Re-raises exceptions from the llm.invoke call.
    """
    # Limit prompt logging for privacy/verbosity
    prompt_repr = str(prompt)[:1000] + '...' if len(str(prompt)) > 1000 else str(prompt)
    logger.debug(f"Calling LLM with prompt: {prompt_repr}")
    try:
        # Assuming the LLM's invoke method returns an object with a 'content' attribute
        # (like AIMessage or ChatCompletion). Adjust if your LLM client returns differently.
        response_obj = llm.invoke(prompt)

        # Extract content based on typical LangChain patterns
        if hasattr(response_obj, 'content'):
             response_content = response_obj.content
        elif isinstance(response_obj, str):
             response_content = response_obj # Handle cases where invoke directly returns a string
        else:
             # Handle cases where response_obj might be a Pydantic model or other type
             # Attempt to convert to string or dump to JSON if it's a known model/dict
             try:
                 if isinstance(response_obj, BaseModel):
                     response_content = response_obj.model_dump_json(indent=2)
                 elif isinstance(response_obj, dict) or isinstance(response_obj, list):
                      response_content = json.dumps(response_obj, indent=2)
                 else:
                    logger.warning(f"LLM response object type ({type(response_obj)}) has no 'content' attribute and is not a string/dict/list/BaseModel. Returning raw object representation.")
                    response_content = str(response_obj)
             except Exception as inner_e:
                  logger.warning(f"Could not convert LLM response object to string/JSON: {inner_e}")
                  response_content = str(response_obj)


        # Ensure response is a string
        if not isinstance(response_content, str):
             logger.warning(f"LLM response content is not a string ({type(response_content)}). Converting to string.")
             response_content = str(response_content)

        # Limit response logging for privacy/verbosity
        response_repr = response_content[:1000] + '...' if len(response_content) > 1000 else response_content
        logger.debug(f"LLM call successful. Response: {response_repr}")
        return response_content
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        # Include prompt in error for debugging
        logger.error(f"Failed LLM prompt: {prompt_repr}")
        raise # Re-raise the exception so the calling node can handle it

# --- JSON Parsing Helper ---

def parse_llm_json_output_with_model(llm_output: str, expected_model: Optional[Type[BaseModel]] = None) -> Any:
    """
    Parses JSON output from an LLM string, handling potential markdown formatting
    and optional Pydantic model validation.

    Args:
        llm_output: The raw string output from the LLM.
        expected_model: An optional Pydantic model class (type) to validate against.

    Returns:
        The parsed JSON object (dict, list, or Pydantic model instance),
        or None if parsing or validation fails.
    """
    if not isinstance(llm_output, str):
        logger.error(f"Cannot parse non-string LLM output as JSON. Type: {type(llm_output)}")
        return None

    # Limit logging of raw LLM output for privacy/verbosity
    llm_output_repr = llm_output[:1000] + '...' if len(llm_output) > 1000 else llm_output
    logger.debug(f"Attempting to parse LLM output as JSON: {llm_output_repr}")

    json_block = llm_output.strip()

    # Attempt to extract JSON block if LLM wrapped it in markdown code fences
    # Robustly handle various markdown fences and optional language identifier
    if json_block.startswith('```'):
        end_fence_pos = json_block.rfind('```')
        if end_fence_pos > 0: # Ensure closing fence exists and is not at the start
            first_newline_pos = json_block.find('\n')
            if first_newline_pos != -1 and first_newline_pos < end_fence_pos:
                 # Content starts after the first newline following the opening fence
                 json_block = json_block[first_newline_pos + 1:end_fence_pos].strip()
                 logger.debug("Extracted content between markdown fences.")
            else:
                 # No newline after ``` or newline is after the closing fence,
                 # maybe just ``` without language or newline? Unlikely but handle.
                 # Or closing fence is missing/malformed.
                 logger.warning("Could not find standard markdown fence structure. Attempting to parse full block.")
                 # Fallback: remove only the opening fence and any leading/trailing whitespace
                 json_block = json_block[3:].strip()
                 if json_block.endswith('```'): # If it had a closing fence but no newline
                      json_block = json_block[:-3].strip()


    # Sometimes LLMs might just output the JSON without fences
    # Basic check if it looks like JSON (more permissive than before)
    looks_like_json = (json_block.startswith('{') and json_block.endswith('}')) or \
                      (json_block.startswith('[') and json_block.endswith(']')) or \
                      (json_block.strip().startswith('{') and json_block.strip().endswith('}')) or \
                      (json_block.strip().startswith('[') and json_block.strip().endswith(']'))

    if not looks_like_json:
        logger.warning("LLM output doesn't look like a standard JSON object or array (missing typical start/end chars).")
        # Continue attempting to parse, as it might be valid JSON without typical start/end

    try:
        # Attempt to parse the JSON
        parsed_data = json.loads(json_block)
        logger.debug("Successfully parsed JSON.")

        # If an expected Pydantic model class is provided, validate the data
        # Check if expected_model is actually a subclass of BaseModel and is not the dummy class
        # Use hasattr to check for model_validate as a more robust check than __name__
        if expected_model and issubclass(expected_model, BaseModel) and hasattr(expected_model, 'model_validate'):
            logger.debug(f"Validating parsed JSON against model: {expected_model.__name__}")
            try:
                # Use model_validate for data validation, including type coercion
                validated_data = expected_model.model_validate(parsed_data)
                logger.debug("JSON validated successfully against model.")
                return validated_data # Return the validated model instance
            except ValidationError as e:
                logger.error(f"Pydantic validation failed against model {expected_model.__name__}: {e}")
                # Log the problematic data on debug level
                logger.debug(f"Data that failed validation: {parsed_data}")
                return None # Validation failed
        else:
            # No validation requested or invalid/dummy model provided, return raw parsed data
            if expected_model:
                 logger.warning(f"expected_model ({expected_model}) is not a valid Pydantic model type. Skipping validation.")
            return parsed_data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}. Problematic text snippet: '{json_block[max(0, e.pos-20):min(len(json_block), e.pos+20)]}'", exc_info=False) # Log context around error
        logger.debug(f"Full text attempted parsing: {json_block}") # Log full text on debug level
        return None
    except Exception as e:
        # Catch any other unexpected errors (e.g., during validation if not ValidationError)
        logger.error(f"An unexpected error occurred during JSON parsing/validation: {e}", exc_info=True)
        return None

