import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

class UnSlothTrainer:
    def __init__(self):
             self.model = None
             self.tokenizer = None


    def FastLanguageModel_from_pretrained(self,
        model_name= "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length             = None,
        dtype                      = None,
        load_in_4bit               = True,
        token                      = None,
        device_map                 = "sequential",
        rope_scaling               = None,
        fix_tokenizer              = True,
        trust_remote_code          = False,
        use_gradient_checkpointing = "unsloth",
        resize_model_vocab         = None,
        revision                   = None,
        use_exact_model_name       = False
                   ):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length             = max_seq_length,
            dtype                      = dtype,
            load_in_4bit               = load_in_4bit,
            token                      = token,
            device_map                 = device_map,
            rope_scaling               = rope_scaling,
            fix_tokenizer              = fix_tokenizer,
            trust_remote_code          = trust_remote_code,
            use_gradient_checkpointing = use_gradient_checkpointing,
            resize_model_vocab         = resize_model_vocab,
            revision                   = revision,
            use_exact_model_name       = use_exact_model_name
        )
        return self.model, self.tokenizer

    def FastLanguageModel_get_peft_model(self,
        r                   = 16,
        target_modules=None,
        lora_alpha          = 16,
        lora_dropout        = 0,
        bias                = "none",
        layers_to_transform = None,
        layers_pattern      = None,
        use_gradient_checkpointing = True,
        random_state        = 3407,
        max_seq_length      = 2048, # not used anymore
        use_rslora          = False,
        modules_to_save     = None,
        init_lora_weights   = True,
                                         loftq_config=None,
        temporary_location  = "_unsloth_temporary_saved_buffers"
                                         ):
        if loftq_config is None:
            loftq_config = {}
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r                   = r,
            target_modules=target_modules,
            lora_alpha          = lora_alpha,
            lora_dropout        = lora_dropout,
            bias                = bias,
            layers_to_transform = layers_to_transform,
            layers_pattern      = layers_pattern,
            use_gradient_checkpointing = use_gradient_checkpointing,
            random_state        = random_state,
            use_rslora          = use_rslora,
            modules_to_save     = modules_to_save,
            init_lora_weights   = init_lora_weights,
            loftq_config       = loftq_config,
            temporary_location  = temporary_location
        )
        return self.model


    def get_chat_template(self,
                          chat_template="chatml",
                          mapping=None,
                          map_eos_token=True,
                          system_message=None,
                          ):
        if mapping is None:
            mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"}

        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=chat_template,
            mapping=mapping,
            map_eos_token=map_eos_token,
            system_message=system_message,
        )
        return self.tokenizer

    def formatting_prompts_func(self,examples):
        convos = examples["conversations"]
        texts = [self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts, }





