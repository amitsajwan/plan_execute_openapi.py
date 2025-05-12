# core_logic.py

# ... (other imports like json, logging, asyncio, yaml, models, utils, ValidationError) ...

# --- Corrected Imports (Key Change 1) ---
# Import APIExecutor and WorkflowExecutor from their definitive locations.
# This removes the problematic placeholder definitions from the main scope of this file.
try:
    from api_executor import APIExecutor
except ImportError:
    logger.critical(
        "core_logic.py: Failed to import APIExecutor from api_executor.py. "
        "Ensure api_executor.py exists and is in the Python path. "
        "A placeholder will be used, which may cause runtime errors if API execution is attempted."
    )
    class APIExecutor: # type: ignore # Fallback placeholder
        def __init__(self, *args, **kwargs):
            logger.error("CRITICAL: Using CORE_LOGIC's FALLBACK APIExecutor due to import error. API calls will NOT work.")
        async def execute_api(self, *args, **kwargs) -> Dict[str, Any]:
            logger.error("FALLBACK APIExecutor execute_api called. This should not happen in a normal run.")
            return {"status_code": 500, "error": "Fallback APIExecutor used, real execution failed."}

try:
    from workflow_executor import WorkflowExecutor # This should be your actual/simplified WorkflowExecutor
except ImportError:
    logger.critical(
        "core_logic.py: Failed to import WorkflowExecutor from workflow_executor.py. "
        "Ensure workflow_executor.py exists and is in the Python path. "
        "A placeholder will be used, which WILL cause runtime errors if workflow execution is attempted."
    )
    class WorkflowExecutor: # type: ignore # Fallback placeholder
        def __init__(self, *args, **kwargs): 
            logger.error(
                "CRITICAL: Using CORE_LOGIC's FALLBACK WorkflowExecutor due to import error. "
                "Workflow execution will NOT work correctly. Args received: %s, %s", args, kwargs
            )
        async def run_workflow_streaming(self, *args, **kwargs):
            logger.error("FALLBACK WorkflowExecutor run_workflow_streaming called.")
        async def submit_interrupt_value(self, *args, **kwargs):
            logger.error("FALLBACK WorkflowExecutor submit_interrupt_value called.")

logger = logging.getLogger(__name__)

class OpenAPICoreLogic:
    # ... (__init__ and other methods like parse_openapi_spec, _generate_llm_schema_summary, etc.) ...
    # Ensure __init__ uses the imported APIExecutor:
    # def __init__(self, worker_llm: Any, api_executor_instance: APIExecutor):
    #     ...
    #     self.api_executor = api_executor_instance
    #     ...

    # Method to setup workflow execution (Key Change 2)
    def setup_workflow_execution(self, state: BotState) -> BotState:
        tool_name = "setup_workflow_execution"
        logger.info(f"[{state.session_id}] Setting up workflow execution.")
        state.update_scratchpad_reason(tool_name, "Preparing for workflow execution.")

        if not getattr(state, 'execution_graph', None): # Safely access execution_graph
            state.response = "No execution graph to run. Please generate one first."
            state.workflow_execution_status = "failed"
            state.next_step = "responder"
            return state

        if state.workflow_execution_status in ["running", "paused_for_confirmation"]:
            state.response = "A workflow is already running or paused."
            state.next_step = "responder"
            return state

        try:
            # Ensure scratchpad exists and remove any old executor instance from it.
            # The BotState itself should not carry the executor instance.
            if state.scratchpad is None:
                state.scratchpad = {}
            state.scratchpad.pop('workflow_executor_instance', None) 
            
            state.workflow_execution_status = "pending_start" 
            state.response = (
                "Workflow execution initialized. The system will attempt to start. "
                "Updates will follow."
            )
            logger.info(f"[{state.session_id}] Workflow status set to 'pending_start'. Main.py will create and manage executor.")

        except Exception as e:
            logger.error(f"[{state.session_id}] Error preparing for workflow execution: {e}", exc_info=True)
            state.response = f"Critical error setting up workflow: {str(e)[:150]}"
            state.workflow_execution_status = "failed"

        state.next_step = "responder" # The main graph responds; main.py will handle the actual start
        return state

    # Method to resume workflow (Key Change 3)
    def resume_workflow_with_payload(self, state: BotState, confirmed_payload: Dict[str, Any]) -> BotState:
        tool_name = "resume_workflow_with_payload"
        logger.info(f"[{state.session_id}] Attempting to resume workflow with confirmed_payload.")
        state.update_scratchpad_reason(tool_name, f"Received payload for resumption: {str(confirmed_payload)[:100]}...")

        # The actual executor is managed in main.py's active_workflow_executors
        # This node signals main.py by putting payload into scratchpad.
        if state.workflow_execution_status != "paused_for_confirmation":
            state.response = (
                f"Workflow not paused for confirmation (status: {state.workflow_execution_status}). "
                "Cannot resume."
            )
            state.next_step = "responder"
            return state

        try:
            # Ensure scratchpad exists
            if state.scratchpad is None:
                state.scratchpad = {}
            # Store the payload in scratchpad; main.py will pick it up 
            # and call executor.submit_interrupt_value on the externally managed instance.
            state.scratchpad['pending_resume_payload'] = confirmed_payload
            
            # Tentatively set to running; main.py confirms and manages the actual executor's state.
            state.workflow_execution_status = "running" 
            state.response = "Confirmation payload received. Workflow will attempt to resume via main handler."
            logger.info(f"[{state.session_id}] Confirmed payload stored in scratchpad for main.py to process. Workflow status tentatively 'running'.")

        except Exception as e: # Should be rare if just setting scratchpad
            logger.error(f"[{state.session_id}] Error preparing payload for workflow resumption: {e}", exc_info=True)
            state.response = f"Error resuming workflow: {str(e)[:150]}"
            # Consider if status should be reverted or set to failed here.

        state.next_step = "responder"
        return state
    
    # ... (other methods like handle_unknown, handle_loop, etc.) ...            
