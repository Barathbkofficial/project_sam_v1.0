from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=4,   # Batch size for training
    save_steps=10_000,               # Save steps
    save_total_limit=2,              # Limit the total amount of checkpoints
    logging_dir='./logs',            # Directory for storing logs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],  # If you have a validation dataset
)

trainer.train()
