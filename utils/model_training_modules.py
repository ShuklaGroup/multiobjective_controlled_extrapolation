## Acknowledgments

# Parts of this codebase were adapted from:

# - https://github.com/vishakhpk/iter-extrapolation — which implements the iterative controlled extrapolation method
# - https://github.com/huggingface/transformers — for model loading, fine-tuning, and tokenization

# We thank the original authors for making their work openly available.


import torch
import numpy as np
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field

import pandas as pd

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AlbertTokenizer,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers import TrainingArguments, Trainer

import os
import peft
from peft import LoraConfig, get_peft_model, PeftModel,TaskType

def train_model_with_lora(model_id, dataset_name, device):
    # Prepare Training Data
    train_file = "pair_data_"+dataset_name+".json"
    data_files = {}
    data_files["train"] = train_file
    extension = train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, 
                                cache_dir="./cache_dir"
                               )
    
    # Create Config for Foundation Model
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir="./cache_dir"
    )

    # Set max target and source length
    max_target_length= 150
    max_source_length= 150

    # Add Special Tokens
    config.max_length = max_target_length
    tokenizer = T5Tokenizer.from_pretrained(
        model_id,
        cache_dir="./cache_dir",
        use_fast=True,
    )
    tokenizer.add_tokens(["<inc>", "<dec>"], special_tokens=True)

    # Load Foundation Model (Prot T5)
    foundation_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        from_tf=bool(".ckpt" in model_id),
        config=config,
        cache_dir="./cache_dir"
    ).to(device)

    foundation_model.resize_token_embeddings(len(tokenizer))

    target_lang="tgt"
    source_lang="src"

    foundation_model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(target_lang)
    prefix = ""
    column_names = raw_datasets["train"].column_names
    MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast]

    # Get the language codes for input/target.
    source_lang = source_lang.split("_")[0]
    target_lang = target_lang.split("_")[0]
    
    # Temporarily set max_target_length for training.
    max_target_length = max_target_length
    padding = "max_length"

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
    
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
    
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Prepare Train Dataset
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        desc="Running tokenizer on train dataset",
    )

    # Data collator
    label_pad_token_id = -100
    data_collator = default_data_collator
    
    # Metric
    metric = load_metric("sacrebleu", trust_remote_code=True)

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
        )
    
    peft_model = get_peft_model(foundation_model, lora_config)
    print(peft_model.print_trainable_parameters())

    # Create a directory to contain the Model
    working_dir = './'
    output_directory = os.path.join(working_dir, "peft_"+dataset_name+"_outputs")

    # Input Training Arguments
    training_args = TrainingArguments(
    output_dir=output_directory,
    per_device_train_batch_size=1, 
    learning_rate= 1e-4, # Higher learning rate than full fine-tuning.
    num_train_epochs= 1,
    weight_decay = 0.0001,
    logging_steps=500,
    save_strategy="epoch"
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # Save model
    peft_model_path = os.path.join(output_directory, f"lora_model")
    trainer.model.save_pretrained(peft_model_path)
