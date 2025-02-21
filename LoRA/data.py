# data.py

from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_ID

def load_and_tokenize_dataset():
    """
    Loads the SQuAD training dataset and tokenizes it using the model's tokenizer.
    """
    # Load the SQuAD dataset (training split)
    dataset = load_dataset("squad", split="train")

    # Load the tokenizer with trust_remote_code enabled
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Set pad token to eos_token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # Tokenize question and context together
        outputs = tokenizer(
            examples["question"],
            examples["context"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        # For causal language modeling, use input_ids as labels
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    # Apply tokenization in batched mode
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenizer, tokenized_dataset
