from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
from typing import List, Tuple

def get_gradient(model_name: str, prompt: str, target_output: str, system_prompt: str = None, objective_prompt: str = None) -> List[Tuple[str, float]]:
    """
    Get gradient-based importance scores for each token in the prompt.
    
    Args:
        model_name (str): The name of the pre-trained model to use.
        prompt (str): The input prompt text.
        target_token (str): The target token to compute gradients against.
        system_prompt (str, optional): An optional system prompt to prepend to the input.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing tokens and their importance scores.
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    if torch.cuda.is_available():
        model.to('cuda')

    # Prepare input
    if system_prompt:
        full_prompt = system_prompt + "\n" + prompt
    else:
        full_prompt = prompt

    full_prompt += target_output
    inputs = tokenizer(full_prompt, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
    

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask) # shape = (1, seq_len, vocab_size)
    logits = outputs.logits # shape = (1, seq_len, vocab_size)
    shift_logits = logits[:, :-1, :].contiguous() # shape = (1, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:].contiguous() # shape = (1, seq_len-1)
    # Compute loss for the target token

    # Backward pass
    loss.backward()

    # Get gradients
    gradients = input_ids.grad[0]

    # Compute importance scores
    token_importance = []
    for i, token_id in enumerate(input_ids[0]):
        token = tokenizer.decode([token_id])
        grad_norm = gradients[i].norm().item()
        token_importance.append((token, grad_norm))

    return token_importance