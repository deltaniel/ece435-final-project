from datasets import load_dataset
from transformers import AutoTokenizer
import os
import numpy as np

CACHE_DIR = os.getenv("HF_HOME")
tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/alpaca-7b-reproduced", cache_dir=CACHE_DIR)
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-prompt", split="train", cache_dir=CACHE_DIR)

def count_tokens(example):
    return {'token_length': len(tokenizer(example['prompt'])['input_ids'])}

lengths = dataset.map(count_tokens, remove_columns=dataset.column_names)
print("Token length percentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"{p}th percentile: {np.percentile(lengths['token_length'], p):.1f}")
