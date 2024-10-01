from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load your dataset
dataset = load_dataset('google/Synthetic-Persona-Chat')

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token to be the same as the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Prepare inputs and labels
def preprocess_function(examples):
    # Ensure the input is a string
    conversations = examples['Best Generated Conversation']
    
    # Check if the input is a string or list of strings
    if isinstance(conversations, list):
        conversations = [str(conv) for conv in conversations]  # Convert each conversation to a string
    elif not isinstance(conversations, str):
        raise ValueError("Each conversation must be a string or a list of strings.")
    
    # Tokenize inputs
    input_ids = tokenizer(conversations, truncation=True, padding='max_length', max_length=512)
    
    return {
        'input_ids': input_ids['input_ids'],
        'attention_mask': input_ids['attention_mask'],
        'labels': input_ids['input_ids'],  # Use the same input for labels for language modeling
    }

# Tokenizing the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Initialize the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# Train the model
trainer.train()
