from transformers import TrainingArguments, Trainer
from model import load_model
from data import load_and_tokenize_dataset
from config import TRAINING_ARGS

def train():
    """
    Sets up the Trainer and begins the fine-tuning process.
    """
    # Load tokenizer and tokenized dataset
    tokenizer, tokenized_dataset = load_and_tokenize_dataset()
    # Load the pre-trained model with LoRA adapters attached
    model = load_model()

    # Create TrainingArguments using our configuration
    training_args = TrainingArguments(**TRAINING_ARGS)

    # Initialize the Trainer with model, data, and training arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    # Begin fine-tuning; resume from checkpoint if available.
    trainer.train(resume_from_checkpoint=True)