MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Training arguments configuration
TRAINING_ARGS = {
    "output_dir": "./result# config.py

# Model identifier
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Training arguments configuration
TRAINING_ARGS = {
    "output_dir": "./results_squad",
    "eval_strategy": "steps",      # Evaluate during training
    "eval_steps": 100,
    "logging_steps": 100,
    "save_steps": 200,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "fp16": True,                  # Enable mixed precision training
    "report_to": "none",           # Adjust reporting as needed (e.g., "tensorboard")
    "log_level": "info",
    "learning_rate": 1e-5,
    "max_grad_norm": 2,
}
s",
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,           # Adjust as needed
    "logging_steps": 10,
    "fp16": True,                    # Mixed precision training
    "save_strategy": "steps",        # Save checkpoint every few steps
    "save_steps": 500,               # Save every 500 steps
    "save_total_limit": 3,           # Keep only the 3 most recent checkpoints
}

# LoRA parameters for PEFT
LORA_PARAMS = {
    "r": 16,                   # Rank of the LoRA adapter matrices
    "lora_alpha": 16,          # Scaling factor
    "lora_dropout": 0.0,       # Dropout for LoRA layers (set >0 for regularization)
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
