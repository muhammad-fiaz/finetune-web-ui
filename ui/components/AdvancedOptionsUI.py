import gradio as gr


class AdvancedOptionsUI:
    def __init__(self):
        self.block = None
        self.options = None

    def create_ui(self):
        """Create the advanced options UI."""
        with gr.Group(visible=False) as advanced_block:
            with gr.Row():
                max_seq_length = gr.Slider(
                    label="Max Sequence Length",
                    minimum=1,
                    maximum=342733,
                    value=2048,
                    info="Set the maximum sequence length.",
                )
            with gr.Row():
                load_in_4bit = gr.Checkbox(
                    label="Load in 4-bit",
                    value=True,
                    info="Enable loading the model in 4-bit precision to save memory.",
                )
            with gr.Row():
                learning_rate = gr.Slider(
                    label="Learning Rate",
                    minimum=0.0001,
                    maximum=0.1,
                    step=0.0001,
                    value=0.001,
                    info="Adjust the learning rate for training.",
                )
            with gr.Row():
                trainer_max_seq_length = gr.Number(
                    label="Max Sequence Length",
                    value=2048,
                    info="Set the maximum sequence length for the trainer.",
                )
                batch_size = gr.Number(
                    label="Batch Size",
                    value=32,
                    info="Specify the number of samples per batch.",
                )
                epochs = gr.Number(
                    label="Epochs", value=10, info="Set the number of training epochs."
                )
                gradient_accumulation_steps = gr.Number(
                    label="Gradient Accumulation Steps",
                    value=4,
                    info="Number of gradient accumulation steps.",
                )
                warmup_steps = gr.Number(
                    label="Warmup Steps",
                    value=5,
                    info="Number of warmup steps during training.",
                )
                max_steps = gr.Number(
                    label="Max Steps",
                    value=60,
                    info="Set the maximum number of training steps.",
                )
                lora_r = gr.Number(
                    label="LoRA r", value=16, info="LoRA hyperparameter r value."
                )
                lora_alpha = gr.Number(
                    label="LoRA Alpha", value=16, info="LoRA alpha value."
                )
                lora_dropout = gr.Number(
                    label="LoRA Dropout", value=0.0, info="LoRA dropout rate."
                )
                random_state = gr.Number(
                    label="Random State",
                    value=3407,
                    info="Set the random state for reproducibility.",
                )
                optim = gr.Textbox(
                    label="Optimizer",
                    value="adamw_8bit",
                    info="Specify the optimizer to use (e.g., AdamW).",
                )
                weight_decay = gr.Number(
                    label="Weight Decay",
                    value=0.01,
                    info="Set the weight decay for regularization.",
                )
                lr_scheduler_type = gr.Textbox(
                    label="LR Scheduler Type",
                    value="linear",
                    info="Specify the type of learning rate scheduler (e.g., linear).",
                )
                loftq_config = gr.Textbox(
                    label="LoftQ Config",
                    placeholder="Enter LoftQ Config",
                    info="Provide the configuration for LoftQ.",
                )
                use_rslora = gr.Checkbox(
                    label="Use RSLora",
                    value=False,
                    info="Enable or disable the use of RSLora.",
                )
                logging_steps = gr.Number(
                    label="Logging Steps",
                    value=1,
                    info="Specify the number of steps between logging updates.",
                )
                dataset_split = gr.Textbox(
                    label="Dataset Split",
                    value="train",
                    info="Specify the dataset split to use.",
                )
                dataset_num_proc = gr.Number(
                    label="Dataset Num Proc",
                    value=1,
                    info="Specify the number of processes to use for the dataset.",
                )
                dataset_packing = gr.Checkbox(
                    label="Dataset Packing",
                    value=False,
                    info="Enable or disable dataset packing.",
                )
                use_gradient_checkpointing = gr.Dropdown(
                    label="Use Gradient Checkpointing",
                    choices=["unsloth", "True"],
                    value="unsloth",
                    info="Select the gradient checkpointing method.",
                )
                map_eos_token = gr.Checkbox(
                    label="Map EOS Token",
                    value=False,
                    info="Enable or disable mapping the EOS token.",
                )
            with gr.Row():
                target_modules = gr.CheckboxGroup(
                    choices=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    value=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    label="Select target modules",
                    info="Select the target modules to be fine-tuned.",
                )

            with gr.Row(equal_height=True):
                train_report_to = gr.Textbox(
                    label="Training Report To",
                    value="none",
                    info="Specify the report destination for training.",
                )
                output_dir = gr.Textbox(
                    label="Output Directories",
                    placeholder="Enter the output directories",
                    info="Specify the folder where models need to be placed!",
                    value="../outputs",
                )
            with gr.Row(equal_height=True):
                set_chat_template = gr.Textbox(
                    label="Set Chat Template",
                    value="llama-3.1",
                    info="Set the chat template for the tokenizer.",
                )
                mapping_template = gr.Textbox(
                    label="Mapping Template",
                    info="Set the mapping template for training.",
                )

            with gr.Row(equal_height=True):
                instruction_part = gr.Textbox(
                    label="Instruction Part",
                    value="<|start_header_id|>user<|end_header_id|>\n\n",
                    info="Set the instruction part for training.",
                )
                response_part = gr.Textbox(
                    label="Response Part",
                    value="<|start_header_id|>assistant<|end_header_id|>\n\n",
                    info="Set the response part for training.",
                )

            self.block = advanced_block
            self.options = {
                "max_seq_length": max_seq_length,
                "load_in_4bit": load_in_4bit,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "random_state": random_state,
                "loftq_config": loftq_config,
                "use_rslora": use_rslora,
                "target_modules": target_modules,
                "logging_steps": logging_steps,
                "trainer_max_seq_length": trainer_max_seq_length,
                "weight_decay": weight_decay,
                "optim": optim,
                "lr_scheduler_type": lr_scheduler_type,
                "output_dir": output_dir,
                "dataset_split": dataset_split,
                "dataset_num_proc": dataset_num_proc,
                "dataset_packing": dataset_packing,
                "train_report_to": train_report_to,
                "set_chat_template": set_chat_template,
                "instruction_part": instruction_part,
                "response_part": response_part,
                "use_gradient_checkpointing": use_gradient_checkpointing,
                "map_eos_token": map_eos_token,
                "mapping_template": mapping_template,
            }
            return advanced_block, self.options
