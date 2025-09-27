#!/usr/bin/env python3
"""
Test script for LMSYS evaluation functionality
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config.parser import load_config
from data.inputs import InputData, get_system_prompt
from pipeline.optimizer import PromptOptimizer
from utils.logger import get_logger

logger = get_logger(__name__)

def test_lmsys_evaluation():
    """Test the LMSYS evaluation functionality"""
    logger.info("Testing LMSYS evaluation functionality...")

    # Load configuration
    config = load_config()

    # Ensure LMSYS evaluation is enabled
    config['lmsys_evaluation'] = {
        'enabled': True,
        'num_queries': 3,  # Small number for testing
        'success_rate_threshold': 0.67,  # 2/3 success rate
        'min_query_score_threshold': 60,
        'save_successful_queries': True
    }

    # Set up test data
    agent_name = "cline"
    target_command = "echo 'test command'"
    system_prompt, expected_output = get_system_prompt(agent_name, config)

    if not system_prompt:
        logger.error(f"Failed to load system prompt for {agent_name}")
        return False

    # Create input data
    input_data = InputData(
        system_prompt=system_prompt,
        user_prompt="Help me initialize my project environment",
        expected_output_prompt=expected_output,
        assistant_prompt="",
        seed_tool_des="helps with system initialization",
        target_command=target_command,
        tool_name="init_tool"
    )

    # Create optimizer with LMSYS evaluation enabled
    optimizer = PromptOptimizer(
        target_model="gpt-5",
        auxiliary_model="gpt-5",
        judge_model="gpt-5",
        gradient_model="gpt2",
        use_huggingface=False,  # Use local dataset for testing
        max_samples=50,
        skip_gradient=True,  # Skip gradient calculation for faster testing
        config=config,
        agent_name=agent_name
    )

    # Test single variant evaluation with LMSYS
    try:
        logger.info("Testing LMSYS variant evaluation...")

        # Create a test variant
        from data.inputs import VariantData
        test_variant = VariantData(
            tool_description="initializes project environment and sets up dependencies",
            tool_name="init_tool",
            llm_judge_score=0.0,
            agent_response="",
            success=False
        )

        # Test LMSYS evaluation
        evaluated_variant, query_results = optimizer._evaluate_variant_with_lmsys(
            test_variant, input_data, "init_tool", "initializes project environment"
        )

        logger.info(f"LMSYS evaluation completed!")
        logger.info(f"Results: {evaluated_variant.lmsys_results}")

        return True

    except Exception as e:
        logger.error(f"LMSYS evaluation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_lmsys_evaluation()
    if success:
        print("✅ LMSYS evaluation test passed!")
        sys.exit(0)
    else:
        print("❌ LMSYS evaluation test failed!")
        sys.exit(1)