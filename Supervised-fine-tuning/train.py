from transformers import Trainer, TrainingArguments
from model import load_model
from data import load_and_preprocess_data
from config import TRAINING_ARGS

def train():
    """
    Sets up the Trainer and begins the fine-tuning process.
    """
    # Load data and tokenizer
    tokenizer, train_dataset, eval_dataset = load_and_preprocess_data()
    # Load the model
    model = load_model()

    # Create TrainingArguments from our configuration
    training_args = TrainingArguments(**TRAINING_ARGS)

    # Initialize the Trainer with model, data, and training arguments
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Begin fine-tuning; resume from a checkpoint if one exists
    trainer.train(resume_from_checkpoint=True)
