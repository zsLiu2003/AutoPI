# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoPI is a Python-based prompt optimization framework that uses gradient-based evaluation and multiple LLM providers. The system is designed to optimize prompts through iterative evaluation and mutation processes.

## Commands

This is a pure Python project without package managers. Run Python files directly:

```bash
# Main execution
python main.py

# Individual components
python pipeline/optimizer.py
python pipeline/evaluator.py  
python pipeline/executor.py
```

## Architecture

The codebase follows a modular pipeline architecture:

### Core Components

- **pipeline/**: Contains the main processing pipeline
  - `optimizer.py`: PromptOptimizer class for prompt optimization
  - `evaluator.py`: Multiple evaluator classes (PromptEvaluator, GradientBasedEvaluator, RuleBasedEvaluator)  
  - `executor.py`: CommandExecutor for processing InputDefinition objects

- **utils/**: Utility modules
  - `llm_provider.py`: Abstract LLMProvider interface with concrete implementations for OpenAI, Gemini, Anthropic, and Grok
  - `get_loss.py`: Gradient computation using transformers and PyTorch
  - `logger.py`: Logging utilities

- **data/**: Input/output data management
  - `inputs.py`: InputData dataclass and system prompt loading functions
  - Various `.txt` files containing prompts and expected outputs
  - `outputs.py`: Output data structures

- **config/**: Configuration management
  - `parser.py`: YAML and JSON configuration loading
  - `api_config.json`: API configuration storage
  - `config.yaml`: Main configuration file (contains API keys)

### LLM Integration

The system supports multiple LLM providers through a factory pattern in `utils/llm_provider.py`:
- OpenAI (GPT models)
- Google (Gemini models) 
- Anthropic (Claude models)
- xAI (Grok models)

Provider selection is automatic based on model name prefixes.

### Data Flow

1. Configuration loaded from `config.yaml`
2. Input data defined using `InputData` dataclass
3. Prompts processed through pipeline components (optimizer → evaluator → executor)
4. LLM providers called for generation
5. Results evaluated using gradient-based or rule-based methods

## Configuration

API keys are stored in `config.yaml`. The system expects keys for:
- openai
- google  
- anthropic
- xai

Data path is configurable via the `data_path` setting in `config.yaml`.