🚀 Fine-Tuning Llama 3.2 (1B) with Hugging Face 🤗
This repository contains code and instructions for fine-tuning the Llama 3.2 (1B) model using Hugging Face's Transformers and PEFT (Parameter-Efficient Fine-Tuning) framework. The goal is to adapt the model for domain-specific tasks while optimizing resource efficiency.

📝 Features
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
Implements DeepSpeed & bitsandbytes for memory-efficient training
Supports custom datasets in JSON, CSV, and Parquet formats
Training on consumer GPUs with mixed-precision (FP16/BF16)
Inference with optimized quantization for deployment
