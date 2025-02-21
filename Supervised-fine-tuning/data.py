import os
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_ID

def get_cache_dir():
    """Return the path to the Hugging Face cache directory."""
    return os.environ.get(
        "TRANSFORMERS_CACHE", 
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
    )

def load_and_preprocess_data():
    """
    Loads the SQuAD dataset and applies a chat prompt template,
    tokenizes the examples, and removes unnecessary columns.
    """
    cache_dir = get_cache_dir()
    print("Hugging Face models are cached in:", cache_dir)

    # Load the SQuAD dataset
    dataset = load_dataset("squad")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    def apply_chat_template(example):
        # Use the first available answer (if any)
        answer = example['answers']['text'][0] if len(example['answers']['text']) > 0 else ""
        # Create a prompt combining question and context
        messages = [
            {"role": "user", "content": f"Question: {example['question']}\nContext: {example['context']}"},
            {"role": "assistant", "content": answer}
        ]
        # Use the tokenizer's chat template method if available, else fall back to a simple concatenation
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"Question: {example['question']}\nContext: {example['context']}\nAnswer: {answer}"
        return {"prompt": prompt}

    # Apply the chat template function to the train and validation splits
    train_dataset = dataset["train"].map(apply_chat_template)
    eval_dataset = dataset["validation"].map(apply_chat_template)

    # Define a tokenization function
    def tokenize_function(example):
        tokens = tokenizer(
            example['prompt'],
            padding="max_length",
            truncation=True,
            max_length=128
        )
        # Replace pad token IDs with -100 so they are ignored during loss calculation
        tokens['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']]
        return tokens

    # Tokenize the prompts
    train_dataset = train_dataset.map(tokenize_function, batched=False)
    eval_dataset = eval_dataset.map(tokenize_function, batched=False)

    # Remove unnecessary columns
    columns_to_remove = ['id', 'title', 'context', 'question', 'answers', 'prompt']
    train_dataset = train_dataset.remove_columns(columns_to_remove)
    eval_dataset = eval_dataset.remove_columns(columns_to_remove)

    return tokenizer, train_dataset, eval_dataset
