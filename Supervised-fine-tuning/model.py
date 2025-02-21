# model.py

import torch
from transformers import AutoModelForCausalLM
from config import MODEL_ID

def load_model():
    """
    Loads the base model with model parallelism.
    Note: Using torch.float32 for model weights (adjust if needed).
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    return model
