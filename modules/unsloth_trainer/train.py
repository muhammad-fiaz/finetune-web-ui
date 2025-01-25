from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
from trl import SFTTrainer
import torch

class UnslothTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None

    def load_model(self, model_name="unsloth/Llama-3.2-1B-Instruct", max_seq_length=2048, dtype=None, load_in_4bit=True, token=None):
        """
        Load the model and tokenizer from the specified model name.
        """
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token
        )

    def apply_peft(self, r=16, target_modules=None, lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407, use_rslora=False, loftq_config=None):
        """
        Apply Parameter-Efficient Fine-Tuning (PEFT) to the model.
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

    def set_chat_template(self, chat_template="llama-3.1"):
        """
        Set the chat template for the tokenizer.
        """
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=chat_template
        )

    def format_prompts(self, examples):
        """
        Format the prompts using the chat template.
        """
        convos = examples["conversations"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    def load_dataset(self, dataset_name="mlabonne/FineTome-100k", split="train", dataset_num_proc=2, packing=False):
        """
        Load and format the dataset using the specified chat template.
        """
        self.dataset = load_dataset(dataset_name, split=split)
        self.dataset = standardize_sharegpt(self.dataset)
        self.dataset = self.dataset.map(self.format_prompts, batched=True, num_proc=dataset_num_proc)
        self.packing = packing

    def setup_trainer(
        self,
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none"
    ):
        """
        Set up the SFTTrainer with the specified training arguments.
        """
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            dataset_num_proc=self.packing,
            packing=self.packing,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
                learning_rate=learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=logging_steps,
                optim=optim,
                weight_decay=weight_decay,
                lr_scheduler_type=lr_scheduler_type,
                seed=seed,
                output_dir=output_dir,
                report_to=report_to,  # Use this for WandB etc
            ),
        )

    def train_on_responses_only(self, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n", response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"):
        """
        Configure the trainer to train only on the assistant's responses.
        """
        self.trainer = train_on_responses_only(
            self.trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )

    def train(self):
        """
        Train the model using the configured trainer.
        """
        self.trainer.train()

    def inference(self, messages, max_new_tokens=64, temperature=1.5, min_p=0.1):
        """
        Generate responses based on the input messages.
        """
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def inference_stream(self, messages, max_new_tokens=128, temperature=1.5, min_p=0.1):
        """
        Stream responses token by token.
        """
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        _ = self.model.generate(
            input_ids=inputs["input_ids"],
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
        )

    def save_model(self, save_path="lora_model"):
        """
        Save the trained model and tokenizer.
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_saved_model(self, save_path="lora_model", max_seq_length=2048, dtype=None, load_in_4bit=True):
        """
        Load a saved model and tokenizer for inference.
        """
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=save_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference

# Example usage for testing:
trainer = UnslothTrainer()

# Load model with parameters
trainer.load_model(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    dtype=torch.float32,
    load_in_4bit=True,
    token=None
)

# Apply PEFT with parameters
trainer.apply_peft(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

# Set chat template
trainer.set_chat_template(chat_template="llama-3.1")

# Load dataset with parameters
trainer.load_dataset(
    dataset_name="mlabonne/FineTome-100k",
    split="train",
    dataset_num_proc=2,
    packing=False
)

# Setup trainer with parameters
trainer.setup_trainer(
    max_seq_length=2048,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none"
)

# Train on responses only
trainer.train_on_responses_only(
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Train the model
trainer.train()

# Inference example
messages = [{"role": "user", "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"}]
print(trainer.inference(messages))

# Save model with parameters
trainer.save_model(save_path="lora_model")

# Load saved model for inference
trainer.load_saved_model(
    save_path="lora_model",
    max_seq_length=2048,
    dtype=torch.float32,
    load_in_4bit=True
)

# Inference example with loaded model
print(trainer.inference(messages))