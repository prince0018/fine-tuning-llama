# model.py

import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from config import MODEL_ID, LORA_PARAMS

def load_model():
    """
    Loads the pre-trained model and attaches LoRA adapters.
    """
    # Load the model with fp16 precision and auto device mapping.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # Needed for models that require custom code
    )
    # Create LoRA configuration from parameters
    lora_config = LoraConfig(**LORA_PARAMS)
    # Wrap the model with LoRA adapters; only adapter parameters will be trainable.
    model = get_peft_model(model, lora_config)
    return model
