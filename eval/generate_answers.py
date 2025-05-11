
from __future__ import annotations

import argparse
import json
import os
import time

import openai
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT



PROBLEM_PATH = os.path.join(os.path.dirname(__file__), 'problem.json')

def generate_answer(problems: list[dict[str, str]], model_name_or_path: str) -> list[str]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_name_or_path).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('PKU-Alignment/alpaca-7b-reproduced')

    answers = []
    print(f'Generating answers with {model_name_or_path}')
    for problem in tqdm(problems):
        prompt = PROMPT_INPUT.format(input=problem['prompt'])
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=2048,
            )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt) :]
        answers.append(answer)
    return answers

if __name__ == '__main__':

    with open(PROBLEM_PATH, encoding='utf-8') as f:
        problems = json.load(f)

    sft_answers = generate_answer(problems, "PKU-Alignment/alpaca-7b-reproduced")
    ppo_answers = generate_answer(problems, "PPO-Vanilla")
    safe_rlhf_answers = generate_answer(problems, "Safe-RLHF")
    
    with open('sft_answers.json', 'w', encoding='utf-8') as f:
        json.dump(sft_answers, f, ensure_ascii=False, indent=4)

    with open('ppo_answers.json', 'w', encoding='utf-8') as f:
        json.dump(ppo_answers, f, ensure_ascii=False, indent=4)
    
    with open('safe_rlhf_answers.json', 'w', encoding='utf-8') as f:
        json.dump(safe_rlhf_answers, f, ensure_ascii=False, indent=4)
        
    print('Answers generated and saved to sft_answers.json, ppo_answers.json, and safe_rlhf_answers.json')