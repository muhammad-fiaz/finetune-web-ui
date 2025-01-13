import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

def train_model(dataset_url, model_name="meta-llama/Llama-3.2-1B"):
    # Optional: Set the GPU device ID
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={"train": dataset_url},
        split="train"
    )

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True  # Ensure this is supported by your setup
    )

    # Add LoRA weights with gradient checkpointing enabled
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True  # Correct usage of gradient checkpointing
    )

    # Tokenizing the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    # Split into train and validation sets
    train_dataset = tokenized_datasets
    eval_dataset = train_dataset  # Optionally use a separate validation set

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        output_dir="../outputs",
        logging_steps=10,
        fp16=True,  # Enabling mixed precision
        save_steps=10,  # Optional: Save model checkpoints every 10 steps
        logging_dir='./logs',  # Optional: Log directory for tensorboard
        evaluation_strategy="steps",  # Optional: You can add evaluation
        save_total_limit=2,  # Optional: Keep only the latest two checkpoints
        eval_steps=10,  # Optional: Evaluation every 10 steps
    )

    # Set up the trainer with model, dataset, tokenizer, and training arguments
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    # Example usage (this can be triggered from webui.py as well)
    train_model("https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl")
