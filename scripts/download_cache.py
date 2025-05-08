import os

from datasets import load_dataset
from safe_rlhf.models import AutoModelForScore
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = os.getenv("HF_HOME")

AutoTokenizer.from_pretrained("PKU-Alignment/alpaca-7b-reproduced", cache_dir=CACHE_DIR)
AutoModelForCausalLM.from_pretrained("PKU-Alignment/alpaca-7b-reproduced", cache_dir=CACHE_DIR)
AutoModelForScore.from_pretrained("PKU-Alignment/beaver-7b-v1.0-reward", cache_dir=CACHE_DIR)
AutoModelForScore.from_pretrained("PKU-Alignment/beaver-7b-v1.0-cost", cache_dir=CACHE_DIR)
load_dataset("PKU-Alignment/PKU-SafeRLHF-prompt", split="train", cache_dir=CACHE_DIR)
