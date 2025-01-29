from torch.utils.hipify.hipify_python import mapping

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

    def load_model(self, model_name="unsloth/Llama-3.2-1B-Instruct", max_seq_length=2048, dtype=None, load_in_4bit=True,
                   token=None):
        """
        Load the model and tokenizer from the specified model name.
        """
        # Ensure dtype is set to bfloat16 if supported, otherwise fallback to float16
        dtype = dtype if dtype else (torch.bfloat16 if is_bfloat16_supported() else torch.float16)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token
        )

    def apply_peft(self, r=16, target_modules=None, lora_alpha=16, lora_dropout=0, bias="none",
                   use_gradient_checkpointing="unsloth", random_state=3407, use_rslora=False, loftq_config=None):
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


    def set_chat_template(self, chat_template="llama-3.1", mapping={"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"},map_eos_token = False,    system_message = None):
        """
        Set the chat template for the tokenizer.
        """
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=chat_template,
            system_message=system_message,
            mapping=mapping,
            map_eos_token=map_eos_token,
        )

    def format_prompts(self, examples):
        """
        Format the prompts using the chat template.
        """
        convos = examples["conversations"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in
                 convos]
        return {"text": texts}

    def load_dataset(self, dataset_name="mlabonne/FineTome-100k", split="train", dataset_num_proc=1, packing=False):
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
            report_to="none",
            dataset_num_proc=1,
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
            dataset_num_proc=dataset_num_proc,
            packing=False,  # Can make training 5x faster for short sequences.
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

    def train_on_responses_only(self, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"):
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
        return self.trainer.train()

    def inference(self, messages, max_new_tokens=64, temperature=1.5, min_p=0.1, use_cache=True):
        """
        Generate responses based on the input messages.
        """
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3.1",
        )
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        outputs = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
            use_cache=use_cache
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def inference_stream(self, messages, max_new_tokens=128, temperature=1.5, min_p=0.1, use_cache=True):
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
            input_ids=inputs,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
            use_cache=use_cache,
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
        dtype = dtype if dtype else (torch.bfloat16 if is_bfloat16_supported() else torch.float16)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=save_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference

    def show_memory_stats(self):
        """
        Show current memory stats.
        """
        gpu_stats = torch.cuda.get_device_properties(0)
        self.start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        self.max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {self.max_memory} GB.")
        print(f"{self.start_gpu_memory} GB of memory reserved.")

    def show_final_memory_and_time_stats(self, trainer_stats):
        """
        Show final memory and time stats after training.
        """
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - self.start_gpu_memory, 3)
        used_percentage = round(used_memory / self.max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / self.max_memory * 100, 3)
        print(f"{trainer_stats['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats['train_runtime'] / 60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


