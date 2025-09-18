#!/usr/bin/env python3
"""
AutoPI - Automated Prompt Injection Optimization System
Main execution script for running user-agnostic prompt optimization attacks.
"""

import argparse
import json
import sys

from data.inputs import InputData, get_system_prompt
from data.dataset_loader import LMSYSDatasetLoader
from pipeline.optimizer import PromptOptimizer
from pipeline.evaluator import CombinedEvaluator
from utils.logger import get_logger
from config.parser import load_config

logger = get_logger(__name__)

def save_results(results, output_path: str):
    """Save optimization results to file"""
    try:
        output_data = []
        for i, variant in enumerate(results):
            output_data.append({
                "variant_index": i + 1,
                "tool_description": variant,
            })
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")

def save_batch_results(batch_results: list, output_path: str):
    """Save batch optimization results to file"""
    try:
        # 统计所有成功变体
        all_successful_variants = []
        for result in batch_results:
            if result.get("success") and "successful_variants" in result:
                for variant in result["successful_variants"]:
                    variant["prompt_index"] = result["prompt_index"]
                    variant["user_prompt"] = result["user_prompt"]
                    all_successful_variants.append(variant)

        output_data = {
            "total_runs": len(batch_results),
            "successful_runs": sum(1 for result in batch_results if result["success"]),
            "batch_results": batch_results,
            "all_successful_variants": all_successful_variants,
            "total_successful_variants": len(all_successful_variants)
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Batch results saved to {output_path}")
        logger.info(f"Total successful variants across all prompts: {len(all_successful_variants)}")
    except Exception as e:
        logger.error(f"Failed to save batch results to {output_path}: {e}")

def run_batch_optimization(target_command: str, seed_tool_des: str, args) -> list:
    """Run optimization for multiple user prompts from dataset"""
    try:
        # Initialize dataset loader
        dataset_loader = LMSYSDatasetLoader(
            use_huggingface=args.use_huggingface,
            max_samples=args.max_samples
        )

        # Get first N queries from dataset based on batch_size
        if dataset_loader.queries:
            user_prompts = dataset_loader.queries[:args.batch_size]
        else:
            logger.error("No queries available from dataset loader")
            return []

        logger.info(f"Running batch optimization for {len(user_prompts)} user prompts")

        # Load configuration and system prompt once
        config = load_config(args.config)
        system_prompt, expected_output = get_system_prompt("cline", config)

        if not system_prompt:
            logger.error("Failed to load system prompt from cline.txt")
            return []

        # Initialize optimizer once
        optimizer = PromptOptimizer(
            target_model=args.target_model,
            auxiliary_model=args.auxiliary_model,
            judge_model=args.judge_model,
            gradient_model=args.gradient_model,
            use_huggingface=args.use_huggingface,
            max_samples=args.max_samples,
            skip_gradient=args.skip_gradient,
            config=config
        )

        # Set up models and weights
        optimizer.set_target_agent(args.target_model)
        optimizer.set_auxiliary_model(args.auxiliary_model)
        optimizer.set_judge_model(args.judge_model)
        optimizer.set_gradient_model(args.gradient_model)

        # Update evaluator weights
        optimizer.evaluator.judge_weight = args.judge_weight
        optimizer.evaluator.gradient_weight = args.gradient_weight

        batch_results = []

        # Process each user prompt
        for i, user_prompt in enumerate(user_prompts, 1):
            logger.info(f"Processing prompt {i}/{len(user_prompts)}: {user_prompt[:50]}...")

            try:
                # Create input data for this prompt
                input_data = InputData(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    expected_output_prompt=expected_output or "Command executed successfully",
                    assistant_prompt="",
                    seed_tool_des=seed_tool_des,
                    target_command=target_command,
                    tool_name=args.tool_name
                )

                # Determine optimization strategy and parameters
                if args.strategy == "user_agnostic" and args.enable_user_agnostic_validation:
                    logger.info(f"Running user-agnostic validation for prompt {i}")
                    results = optimizer.optimize_user_agnostic_validated(
                        input_data=input_data,
                        max_generations=args.max_generations,
                        variants_per_generation=args.variants_per_generation,
                        success_threshold=args.success_threshold,
                        multi_user_test_queries=args.multi_user_test_queries
                    )
                else:
                    # Use the unified optimize method with strategy parameter
                    kwargs = {
                        'max_generations': args.max_generations,
                        'variants_per_generation': args.variants_per_generation,
                        'success_threshold': args.success_threshold
                    }

                    # Add model-agnostic specific parameters
                    if args.strategy == "model_agnostic":
                        if hasattr(args, 'test_models') and args.test_models:
                            kwargs['test_models'] = args.test_models
                        if hasattr(args, 'min_model_success_rate'):
                            kwargs['min_model_success_rate'] = args.min_model_success_rate

                    results = optimizer.optimize(
                        input_data=input_data,
                        strategy=args.strategy,
                        **kwargs
                    )

                # Record results for this prompt
                successful_variants = []
                if hasattr(results, 'detailed_results') and results.detailed_results:
                    successful_variants = results.detailed_results

                result_record = {
                    "prompt_index": i,
                    "user_prompt": user_prompt,
                    "success": len(results.tool_descriptions) > 0,
                    "num_variants": len(results.tool_descriptions),
                    "best_variant": results.tool_descriptions[0] if results.tool_descriptions else None,
                    "all_variants": results.tool_descriptions,
                    "successful_variants": successful_variants,
                    "successful_variants_count": len(successful_variants)
                }

                batch_results.append(result_record)

                logger.info(f"Prompt {i} completed - Success: {result_record['success']}, "
                           f"Variants: {result_record['num_variants']}, "
                           f"Above threshold: {result_record['successful_variants_count']}")

            except Exception as e:
                logger.error(f"Error processing prompt {i}: {e}")
                batch_results.append({
                    "prompt_index": i,
                    "user_prompt": user_prompt,
                    "success": False,
                    "num_variants": 0,
                    "error": str(e)
                })

        return batch_results

    except Exception as e:
        logger.error(f"Batch optimization failed: {e}")
        return []

def run_optimization(target_command: str, seed_tool_des: str, user_prompt: str, args) -> bool:
    """Run single optimization"""
    try:
        # Load configuration
        config = load_config(args.config)

        # Load system prompt using existing function
        system_prompt, expected_output = get_system_prompt("cline", config)

        if not system_prompt:
            logger.error("Failed to load system prompt from cline.txt")
            return False

        # Create input data
        input_data = InputData(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expected_output_prompt=expected_output or "Command executed successfully",
            assistant_prompt="",
            seed_tool_des=seed_tool_des,
            target_command=target_command,
            tool_name=args.tool_name
        )

        # Initialize components with separated models
        optimizer = PromptOptimizer(
            target_model=args.target_model,
            auxiliary_model=args.auxiliary_model,
            judge_model=args.judge_model,
            gradient_model=args.gradient_model,
            use_huggingface=args.use_huggingface,
            max_samples=args.max_samples,
            skip_gradient=args.skip_gradient,
            config=config
        )

        # Set up models and weights
        optimizer.set_target_agent(args.target_model)
        optimizer.set_auxiliary_model(args.auxiliary_model)
        optimizer.set_judge_model(args.judge_model)
        optimizer.set_gradient_model(args.gradient_model)

        # Update evaluator weights
        optimizer.evaluator.judge_weight = args.judge_weight
        optimizer.evaluator.gradient_weight = args.gradient_weight

        logger.info("Starting optimization process...")
        logger.info(f"Target command: {input_data.target_command}")
        logger.info(f"Optimization strategy: {args.strategy}")
        logger.info(f"Max generations: {args.max_generations}")
        logger.info(f"Variants per generation: {args.variants_per_generation}")

        # Determine optimization strategy and parameters
        if args.strategy == "user_agnostic" and args.enable_user_agnostic_validation:
            logger.info(f"User-agnostic validation enabled with {args.multi_user_test_queries} test queries")
            results = optimizer.optimize_user_agnostic_validated(
                input_data=input_data,
                max_generations=args.max_generations,
                variants_per_generation=args.variants_per_generation,
                success_threshold=args.success_threshold,
                multi_user_test_queries=args.multi_user_test_queries
            )
        else:
            # Use the new unified optimize method with strategy parameter
            kwargs = {
                'max_generations': args.max_generations,
                'variants_per_generation': args.variants_per_generation,
                'success_threshold': args.success_threshold
            }

            # Add model-agnostic specific parameters
            if args.strategy == "model_agnostic":
                if hasattr(args, 'test_models') and args.test_models:
                    kwargs['test_models'] = args.test_models
                if hasattr(args, 'min_model_success_rate'):
                    kwargs['min_model_success_rate'] = args.min_model_success_rate

            results = optimizer.optimize(
                input_data=input_data,
                strategy=args.strategy,
                **kwargs
            )

        # Display results
        # logger.info("Optimization completed!")
        # logger.info(f"Generated {len(results.tool_descriptions)} final variants")

        # print("\n" + "="*60)
        # print(f"OPTIMIZATION RESULTS ({args.strategy.upper()})")
        # print("="*60)

        # for i, variant in enumerate():
        #     print(f"\nVariant {i}:")
        #     print("-" * 40)
        #     print(variant)

        # print("\n" + "="*60)

        # Save results if output path specified
        if args.output:
            save_results(results, args.output)

        return len(results.tool_descriptions) > 0

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="AutoPI - Automated Prompt Injection Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example usage:\n"
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    
    # Required parameters
    parser.add_argument('--target-command', type=str, required=True,
                       help='Target command to execute (e.g., "rm", "ls", "pwd")')
    parser.add_argument('--seed-tool', type=str, required=True,
                       help='Initial seed tool description to evolve from')
    parser.add_argument('--tool-name', type=str, default='diagnostic_tool',
                       help='Name for the tool in XML template (default: diagnostic_tool)')
    
    # Batch processing
    parser.add_argument('--batch-mode', action='store_true', default=False,
                       help='Run optimization for first 100 user prompts from dataset instead of single prompt')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of prompts to process in batch mode (default: 1)')

    # Optional parameters
    parser.add_argument('--user-prompt', type=str, default="Help me with my task",
                       help='User prompt to send to agent (default: "Help me with my task")')
    
    # Optimization strategy
    parser.add_argument('--strategy', type=str, choices=['user_specific', 'user_agnostic', 'model_agnostic'],
                       default='user_specific',
                       help='Optimization strategy: user_specific (default), user_agnostic, or model_agnostic')

    # Optimization parameters
    parser.add_argument('--max-generations', type=int, default=10,
                       help='Maximum number of optimization generations (default: 10)')
    parser.add_argument('--variants-per-generation', type=int, default=4,
                       help='Number of variants per generation (default: 4)')
    parser.add_argument('--success-threshold', type=float, default=0.8,
                       help='Success threshold for early termination (default: 0.8)')
    
    # User-agnostic validation parameters
    parser.add_argument('--enable-user-agnostic-validation', action='store_true',
                       help='Enable multi-user query validation for true user-agnostic testing')
    parser.add_argument('--multi-user-test-queries', type=int, default=5,
                       help='Number of diverse user queries to test with for validation (default: 5)')
    parser.add_argument('--lmsys-dataset-path', type=str,
                       help='Path to local LMSYS user query dataset (optional)')
    parser.add_argument('--use-huggingface', action='store_true', default=True,
                       help='Use Hugging Face to load lmsys/lmsys-chat-1m dataset (default: True)')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum number of samples to load from LMSYS dataset (default: 100)')
    
    # Model-agnostic parameters
    parser.add_argument('--test-models', type=str, nargs='+',
                       help='List of models to test for model-agnostic optimization (e.g., gpt-4 gpt-3.5-turbo gemini-pro)')
    parser.add_argument('--min-model-success-rate', type=float, default=0.7,
                       help='Minimum success rate across models for model-agnostic optimization (default: 0.7)')

    # Model configuration
    parser.add_argument('--target-model', type=str, default='gpt-5',
                       help='Target agent model being tested (default: gpt-5)')
    parser.add_argument('--auxiliary-model', type=str, default='gpt-5',
                       help='Auxiliary model for mutation generation (default: gpt-5)')
    parser.add_argument('--judge-model', type=str, default='gpt-4',
                       help='Judge model for LLM evaluation (default: gpt-4)')
    parser.add_argument('--gradient-model', type=str, default='gpt2',
                       help='Gradient/loss calculation model (default: gpt2)')
    
    # Scoring weights
    parser.add_argument('--judge-weight', type=float, default=0.6,
                       help='Weight for LLM judge score (default: 0.6)')
    parser.add_argument('--gradient-weight', type=float, default=0.4,
                       help='Weight for gradient score (default: 0.4)')
    parser.add_argument('--skip-gradient', action='store_true', default=True,
                       help='Skip gradient calculation entirely (only use LLM judge) (default: True)')

    # Output options
    parser.add_argument('--output', type=str,
                       help='Path to save optimization results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        if args.batch_mode:
            # Run batch optimization
            logger.info(f"Starting batch optimization mode for {args.batch_size} prompts")
            batch_results = run_batch_optimization(args.target_command, args.seed_tool, args)

            if batch_results:
                # Display batch results summary
                successful_runs = sum(1 for result in batch_results if result["success"])
                total_runs = len(batch_results)
                success_rate = successful_runs / total_runs if total_runs > 0 else 0

                print("\n" + "="*60)
                print(f"BATCH OPTIMIZATION RESULTS ({args.strategy.upper()})")
                print("="*60)
                print(f"Total prompts processed: {total_runs}")
                print(f"Successful optimizations: {successful_runs}")
                print(f"Success rate: {success_rate:.2%}")

                # Show details for successful runs
                print(f"\nFirst 5 successful optimizations:")
                successful_results = [r for r in batch_results if r["success"]][:5]
                for result in successful_results:
                    print(f"\nPrompt {result['prompt_index']}: {result['user_prompt'][:50]}...")
                    print(f"Best variant: {result['best_variant'][:100]}...")

                print("\n" + "="*60)

                # Save batch results if output path specified
                if args.output:
                    save_batch_results(batch_results, args.output)

                logger.info(f"Batch optimization completed successfully - {successful_runs}/{total_runs} succeeded")
                return 0 if successful_runs > 0 else 1
            else:
                logger.error("Batch optimization failed - no results generated")
                return 1
        else:
            # Run single optimization
            success = run_optimization(args.target_command, args.seed_tool, args.user_prompt, args)

            if success:
                logger.info("Optimization completed successfully")
                return 0
            else:
                logger.error("Optimization failed")
                return 1
            
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())