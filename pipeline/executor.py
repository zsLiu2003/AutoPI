# pipeline/executor.py

from typing import List, Dict, Any
from data.inputs import InputData
from data.outputs import UserAgnosticOutput
from utils.logger import get_logger
from utils.llm_provider import get_llm_provider
from config.parser import load_config
import json

logger = get_logger(__name__)

class CommandExecutor:
    """Executes prompt optimization commands and generates outputs"""
    
    def __init__(self):
        self.config = load_config()
        
    def execute_user_agnostic(self, input_data: InputData, num_variants: int = 2) -> UserAgnosticOutput:
        """Execute user-agnostic prompt optimization"""
        logger.info("Executing user-agnostic command pipeline...")
        
        try:
            # Load the user-agnostic system prompt
            with open(f"{self.config.get('data_path', './data')}/user_agnostic_instruction_prompt.txt", 'r') as f:
                system_prompt = f.read().strip()
            
            # Create the prompt for the LLM
            user_prompt = self._format_user_agnostic_prompt(input_data, num_variants)
            
            # Get LLM provider (default to gpt-4)
            provider = get_llm_provider("gpt-5")
            
            # Generate response
            response = provider.generate_response(user_prompt, system_prompt)
            
            # Parse JSON response to extract tool descriptions
            tool_descriptions = json.loads(response)
            if not isinstance(tool_descriptions, list):
                raise ValueError("Expected JSON array response")
            
            logger.info(f"Generated {len(tool_descriptions)} variants")
            return UserAgnosticOutput.from_variants(tool_descriptions)
            
        except Exception as e:
            logger.error(f"User-agnostic execution failed: {e}")
            # Return fallback variants
            fallback_variants = [f"Variant {i+1}: {input_data.user_prompt}" for i in range(num_variants)]
            return UserAgnosticOutput.from_variants(fallback_variants)
    
    def execute(self, input_data: InputData, execution_type: str = "user_agnostic", **kwargs) -> Any:
        """Main execution method that routes to appropriate executor"""
        logger.info(f"Executing {execution_type} pipeline...")
        
        if execution_type == "user_agnostic":
            return self.execute_user_agnostic(input_data, kwargs.get('num_variants', 2))
        else:
            raise ValueError(f"Unknown execution type: {execution_type}")
    
    def _format_user_agnostic_prompt(self, input_data: InputData, num_variants: int) -> str:
        """Format the user prompt for user-agnostic optimization"""
        return f"""
        TARGET_SYSTEM_PROMPT: {input_data.system_prompt}
        
        MALICIOUS_COMMAND: {input_data.target_command}
        
        BASE_TOOL_DESCRIPTION: {input_data.seed_tool_des}
        
        USER_PROMPT: {input_data.user_prompt}
        
        NUM_VARIANTS: {num_variants}
        
        HISTORY: []
        """
    
