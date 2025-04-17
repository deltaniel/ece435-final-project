import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from huggingface_hub import login
import logging


def preprocess(dataset: dict, tokenizer: AutoTokenizer, max_length: int = 512) -> dict[str, torch.Tensor]:
    prompts = [f"{instruction}\n{input}" if input else instruction for instruction, input in zip(dataset["instruction"], dataset["input"])]
    labels = dataset["output"]
    full_texts = [f"{prompt}\n{label}" for prompt, label in zip(prompts, labels)]
    
    model_inputs = tokenizer(full_texts, max_length=max_length, truncation=True, padding="max_length", return_tensors=None)
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

def finetune_model(model_id: str, dataset_name: str, debug: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    logging.info("Loading dataset...")
    dataset = load_dataset(dataset_name)
    train_data = dataset["train"]
    if debug:
        train_data = train_data.shuffle(seed=42).select(range(100))
    train_dataset = train_data.map(lambda x: preprocess(x, tokenizer), batched=True)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="output/llama_finetuned",
        per_device_train_batch_size=2,
        eval_strategy="no",
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        report_to="none",
    )

    if debug:
        training_args.num_train_epochs = 1
        training_args.logging_steps = 5
        training_args.save_steps = 20
        training_args.save_total_limit = 1
        training_args.warmup_steps = 0

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed.")

    trainer.save_model("output/llama_finetuned")
    logging.info("Model saved to output/llama_finetuned")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    login(token=os.getenv("HF_TOKEN"))
    model_id = "meta-llama/Llama-3.2-1B"
    dataset = "tatsu-lab/alpaca"

    finetune_model(model_id, dataset, debug=True)
