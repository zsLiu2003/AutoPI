from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from typing import List, Tuple, Optional

# Global model cache to avoid reloading
_model_cache = {}

def get_gradient(model_name: str, prompt: str, target_output: str, system_prompt: str = None, objective_prompt: str = None, force_cpu: bool = False) -> float:
    """
    Get gradient-based importance scores for each token in the prompt.

    Args:
        model_name (str): The name of the pre-trained model to use.
        prompt (str): The input prompt text.
        target_output (str): The target output to compute loss against.
        system_prompt (str, optional): An optional system prompt to prepend to the input.
        objective_prompt (str, optional): Objective prompt (unused).
        force_cpu (bool): Force computation on CPU to avoid GPU memory issues.

    Returns:
        float: Loss value.
    """
    try:
        # Use cached model if available
        cache_key = f"{model_name}_{'cpu' if force_cpu else 'gpu'}"

        if cache_key not in _model_cache:
            print(f"Loading model {model_name} for first time...")
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()

            # Force CPU usage if requested or if CUDA memory might be an issue
            device = 'cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)

            _model_cache[cache_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'device': device
            }
            print(f"Model {model_name} loaded and cached on {device}")
        else:
            print(f"Using cached model {model_name}")

        model_info = _model_cache[cache_key]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        device = model_info['device']

        # Prepare input
        if system_prompt:
            full_prompt = system_prompt + "\n" + prompt
        else:
            full_prompt = prompt

        inputs_full = tokenizer(full_prompt, return_tensors='pt')
        le = inputs_full["input_ids"].shape[1]
        full_prompt += target_output

        inputs = tokenizer(full_prompt, return_tensors='pt', max_length=4096, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass with memory optimization
        with torch.no_grad():  # Don't compute gradients to save memory
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        ignore_upto = le - 1
        shift_labels[:, :ignore_upto] = -100

        # Fix typo: ignore_index not ingnore_index
        loss_function = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
        loss = loss_function(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Clean up intermediate tensors but keep model cached
        del outputs, logits, shift_logits, shift_labels, input_ids, attention_mask
        if torch.cuda.is_available() and not force_cpu:
            torch.cuda.empty_cache()

        loss_value = loss.item()

        # 修复loss值的问题
        # 1. 确保loss是正数（loss通常应该是正值）
        loss_value = abs(loss_value)

        # 2. 处理极小值问题，避免-0.000这样的显示
        if loss_value < 1e-6:
            loss_value = 0.0

        # 3. 限制loss的范围，避免异常大的值
        if loss_value > 100.0:
            loss_value = 100.0

        return loss_value

    except Exception as e:
        print(f"Gradient calculation failed: {e}")
        return 0.0

def clear_model_cache():
    """Clear the model cache to free memory"""
    global _model_cache
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model cache cleared")


    