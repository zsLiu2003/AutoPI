# pipeline/executor.py

from typing import List, Dict, Any
from data.inputs import InputData
from utils.logger import get_logger
from utils.llm_provider import get_llm_provider
from config.parser import load_config
import json

logger = get_logger(__name__)

class CommandExecutor:
    """Executes prompt optimization commands and generates outputs"""

    def __init__(self):
        self.config = load_config()

    def execute(self, input_data: InputData, execution_type: str = "user_specific", **kwargs) -> Any:
        """Main execution method"""
        logger.info(f"Executing {execution_type} pipeline...")

        if execution_type == "user_specific":
            logger.info("User-specific execution - no special handling needed")
            return {"status": "executed", "type": execution_type}
        else:
            raise ValueError(f"Unknown execution type: {execution_type}")
