# Acknowledgments

# Parts of this codebase were adapted from:

#    https://github.com/vishakhpk/iter-extrapolation — which implements the iterative controlled extrapolation method
#    https://github.com/huggingface/transformers — for model loading, fine-tuning, and tokenization

# We thank the original authors for making their work openly available.


import glob
import json
import pickle
import itertools
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def get_label(mut1, mut2, pb_thresh, zn_thresh):
    # Compute absolute differences
    pb_diff = abs(mut1.Pb - mut2.Pb)
    zn_diff = abs(mut1.Zn - mut2.Zn)

    # If either difference is below threshold, it's insignificant
    if pb_diff < pb_thresh or zn_diff < zn_thresh:
        return "insig"

    # If significant, label as inc or dec depending on direction
    label_pb = "inc" if mut2.Pb > mut1.Pb else "dec"
    label_zn = "inc" if mut2.Zn > mut1.Zn else "dec"

    return f"{label_pb}-{label_zn}"

def get_pair_data(pairs, pb_thresh, zn_thresh):
    # For all pairs, generate label
    pair_data = [
        {
            "Mutant1": mut1.Variant,
            "Mutant2": mut2.Variant,
            "Label": get_label(mut1, mut2, pb_thresh, zn_thresh)
        }
        for mut1, mut2 in pairs
    ]
    
    return pd.DataFrame(pair_data)


def balance_pair_data(pair_data_df, data_df, seed=42):
    # Count number of insignificant, different direction, and same direction pairs
    n_insig = (pair_data_df["Label"] == "insig").sum()

    significant = pair_data_df[pair_data_df["Label"] != "insig"]

    def is_same(label):
        parts = label.split("-")
        return len(parts) == 2 and parts[0] == parts[1]

    def is_opposite(label):
        parts = label.split("-")
        return len(parts) == 2 and parts[0] != parts[1]

    same_df = significant[significant["Label"].apply(is_same)]
    opposite_df = significant[significant["Label"].apply(is_opposite)]

    print(f"Number of insignificant comparisons: {n_insig}")
    print(f"Number of same direction significant comparisons: {len(same_df)}")
    print(f"Number of opposite direction significant comparisons: {len(opposite_df)}")

    # Balance the Number of Same and Opposite Direction Pairs
    same_df_sampled = same_df.sample(n=len(opposite_df), random_state=seed)
    balanced_df = pd.concat([opposite_df, same_df_sampled], ignore_index=True)

    # Check for Coverage of Variant seen in Balanced Pairs
    mutant_list = data_df["Variant"].unique()
    mutant_counts = {mutant: 0 for mutant in mutant_list}
    for _, row in balanced_df.iterrows():
        mutant_counts[row["Mutant1"]] += 1
        mutant_counts[row["Mutant2"]] += 1

    min_coverage = min(mutant_counts.values())
    max_coverage = max(mutant_counts.values())

    print(f"Minimum number of appearances per mutant: {min_coverage}")
    print(f"Maximum number of appearances per mutant: {max_coverage}")
    
    return balanced_df 

# Function creates paired data for LLM
def create_one_input_pair(pb_tok, zn_tok, mut1, mut2):
    d = {}
    d['translation'] = {}
    d['translation']['src'] = pb_tok+zn_tok+' '.join(' '.join(mut1.seq.values[0]))
    d['translation']['tgt'] = ' '.join(' '.join(mut2.seq.values[0]))
    d['translation']['ip-score'] = (mut1["Pb"].values[0],mut1["Zn"].values[0])
    d['translation']['op-score'] = (mut2["Pb"].values[0], mut2["Zn"].values[0])
    return d


def generate_json_file(data, balanced_list_pairs):
    op = []
    for pair in balanced_list_pairs:
        mut1 = data.query('Variant == @pair[0]')
        mut2 = data.query('Variant == @pair[1]')
        if mut2["Pb"].values - mut1["Pb"].values  > 0:
            pb_tok = "<inc> "
            pb_rev_tok = "<dec> "
        else:
            pb_tok = "<dec> "
            pb_rev_tok = "<inc> "
    
        if mut2["Zn"].values - mut1["Zn"].values  > 0:
            zn_tok = "<inc> "
            zn_rev_tok = "<dec> "
        else:
            zn_tok = "<dec> "
            zn_rev_tok = "<inc> "
        
        d = create_one_input_pair(pb_tok, zn_tok, mut1, mut2)
        op.append(d)
        d = create_one_input_pair(pb_rev_tok, zn_rev_tok, mut2, mut1)
        op.append(d)
        
    return op

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_json(filename):
    with open(filename) as f:
        d = json.load(f)
    return d

def save_to_jsonl(file_path, data):
    with open(file_path, 'w') as outfile:
        for item in data:
            json.dump(item, outfile)
            outfile.write('\n')

