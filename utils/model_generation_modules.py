## Acknowledgments

# Parts of this codebase were adapted from:

# - https://github.com/vishakhpk/iter-extrapolation — which implements the iterative controlled extrapolation method
# - https://github.com/huggingface/transformers — for model loading, fine-tuning, and tokenization

# We thank the original authors for making their work openly available.


import os
import torch
import numpy as np
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field

import glob

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

import peft
from peft import LoraConfig, get_peft_model, PeftModel,TaskType

def load_model(model_id, dataset_name, device):
    # Define config
    config = AutoConfig.from_pretrained(
        model_id,
        cache_dir="./cache_dir"
    )
    
    # Define max target and source length
    max_target_length=150
    max_source_length=150

    # Define tokenizer with added special tokens <inc> and <dec>
    config.max_length = max_target_length
    tokenizer = T5Tokenizer.from_pretrained(
        model_id,
        cache_dir="./cache_dir",
        use_fast=True,
    )
    tokenizer.add_tokens(["<inc>", "<dec>"], special_tokens=True)

    # Define foundation model
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

    working_dir = './'
    output_directory = os.path.join(working_dir, "peft_"+dataset_name+"_outputs")
    peft_model_path = os.path.join(output_directory, f"lora_model")

    loaded_model = PeftModel.from_pretrained(foundation_model,
                                        peft_model_path,
                                        is_trainable=False).to(device)

    print(loaded_model)
    return tokenizer, loaded_model

def get_tgt_seq(ex_2, max_tokens, temp, tokenizer, device, loaded_model):
    inputs = tokenizer(ex_2, return_tensors="pt").to(device)
    outputs = loaded_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        top_k = 10, do_sample = True, num_return_sequences = 20, temperature = temp,
        early_stopping=True, #The model can stop before reach the max_length
        eos_token_id=tokenizer.eos_token_id)
    tgt_seq = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return(tgt_seq)

def get_mut_fromseq(mut):
    WT_seq = np.load("WT_seq.npy").tolist()
    mutants = []
    for i in range(len(WT_seq)):
        if WT_seq[i] != mut[i]:
            mutants.append(WT_seq[i]+str(i+1)+mut[i])
    sorted(mutants)
    if len(sorted(mutants)) == 0:
        return 'WT'
    else:
        return '_'.join(sorted(mutants))

def get_seq_frommut(var):
    if var=='WT':
        s = WT_seq
    else: 
        split_var=var.split("_")
        s = WT_seq
        for i in range(len(split_var)):
            AA=re.findall(r'\d+|\D+', split_var[i])[0]
            index=int(re.findall(r'\d+|\D+', split_var[i])[1])-1
            assert(s[index]==AA) 
            sub=re.findall(r'\d+|\D+', split_var[i])[2]
            s = s[:index] + sub + s[index + 1:]
    return s

def generating_muts(round_number, temp, starting_points, tokenizer, device, loaded_model, sorted_mutant_list):
    all_new_mutations = []
    for i in range(len(starting_points)):
        mut = starting_points.seq.tolist()[i]
        
        tok = "<inc> "
        rev_tok = "<dec> "
        
        mut_input = tok+rev_tok+' '.join(' '.join(mut))
        output = get_tgt_seq(mut_input, 150, temp, tokenizer, device, loaded_model)
        output_seq = [i.replace(" ", "") for i in output]
        proposed_mut_list = [get_mut_fromseq(i) for i in output_seq]
        new_mutations = set(proposed_mut_list).difference(set(sorted_mutant_list))
    
        all_new_mutations.append(new_mutations)
        all_items = [item for s in all_new_mutations for item in s]
        unique_items = np.unique(np.array(all_items))
        
        
        print(f"Processed {i} sequences, number of new mutations = {len(unique_items)}")
    os.makedirs("generated_muts", exist_ok=True)
    np.save("generated_muts/"+str(round_number)+"_t"+str(temp)+".npy", list(set.union(*all_new_mutations)))
    return unique_items
