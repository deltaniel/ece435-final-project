import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          LogitsProcessor)

from safe_rlhf.models import AutoModelForScore
from dataloader import RLHFDatasetLoader


logger = logging.getLogger("ppo_rlhf")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(levelname)5s %(message)s")
ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO);  ch.setFormatter(formatter)
fh = logging.FileHandler("ppo_rlhf.log", mode="a"); fh.setLevel(logging.DEBUG); fh.setFormatter(formatter)
logger.addHandler(ch); logger.addHandler(fh)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = RLHFDatasetLoader(
        tokenizer_name="PKU-Alignment/alpaca-7b-reproduced",
        max_length=128,
        batch_size=1,
        shuffle=True
    )

sft_dataset = dataloader.get_dataloader()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

reward_model = AutoModelForScore.from_pretrained(
            "PKU-Alignment/beaver-7b-unified-reward",
            quantization_config=bnb_config,
            device_map="auto",
        ).eval().requires_grad_(False)

base_actor = AutoModelForCausalLM.from_pretrained(
    "PKU-Alignment/alpaca-7b-reproduced",
    quantization_config=bnb_config,
    device_map="auto",
).eval().requires_grad_(False)

actor = PeftModel.from_pretrained(
    base_actor,
    "ppo_actor_lora",
    device_map="auto",
).eval().requires_grad_(False)

with torch.no_grad():
    wins = 0
    for step, batch in enumerate(sft_dataset):
        if step >= 100:
            break

        # move once
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 1) our fine-tuned actor
        seq_a = actor.generate(input_ids, attention_mask=attention_mask)
        L = input_ids.size(1)
        full_mask_a = torch.cat([attention_mask, torch.ones_like(seq_a[:, L:])], dim=1)

        out_a = reward_model(input_ids=seq_a, attention_mask=full_mask_a)
        score_a = out_a.end_scores.squeeze(-1)

        # 2) the base actor
        seq_b = base_actor.generate(input_ids, attention_mask=attention_mask)
        full_mask_b = torch.cat([attention_mask, torch.ones_like(seq_b[:, L:])], dim=1)

        out_b = reward_model(input_ids=seq_b, attention_mask=full_mask_b)
        score_b = out_b.end_scores.squeeze(-1)

        # compare
        if score_a > score_b:
            wins += 1

        logger.info(f"Step {step:03d}  score-tuned={score_a.item():.3f}  score-base={score_b.item():.3f}  wins={wins}")

logger.info(f"Out of 100 samples, tuned-actor won {wins} times")