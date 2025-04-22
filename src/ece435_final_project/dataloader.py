import logging

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


class RLHFDatasetLoader:
    def __init__(self,
                 dataset_name: str = "PKU-Alignment/PKU-SafeRLHF-prompt",
                 dataset_split: str = "train",
                 tokenizer_name: str = "PKU-Alignment/alpaca-7b-reproduced",
                 max_length: int = 512,
                 batch_size: int = 32,
                 shuffle = True

    ):
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load the tokenizer once during initialization
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.dataset = None
        self.tokenized_dataset = None
        self.dataloader = None

    def load_dataset(self):
        # Load the dataset
        self.dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        logging.info(f"Loaded dataset: {self.dataset_name} with split: {self.dataset_split}")
        # Keep only "prompt" column
        self.dataset = self.dataset.remove_columns([col for col in self.dataset.column_names if col != "prompt"])

    def tokenize_dataset(self):
        # Tokenize the dataset
        if self.tokenizer.pad_token is None:
            # Use the EOS token as the pad token.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenized_dataset = self.dataset.map(
            lambda x: self.tokenizer(x['prompt'], truncation=True, max_length=self.max_length),
            batched=True,
            remove_columns=['prompt']
        )
        logging.info(f"Tokenized dataset with max length: {self.max_length}")
       # Convert to PyTorch tensors
        self.tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        logging.info("Converted dataset to PyTorch tensors")

    def create_dataloader(self):
        # Create a DataLoader
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.dataloader = DataLoader(
            self.tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=data_collator
        )
        logging.info(f"Created DataLoader with batch size: {self.batch_size} and shuffle: {self.shuffle}")

    def get_dataloader(self):
        # Load the dataset, tokenize it, and create a DataLoader
        self.load_dataset()
        self.tokenize_dataset()
        self.create_dataloader()
        return self.dataloader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create an instance of the dataset loader
    dataset_loader = RLHFDatasetLoader()
    # Prepare the dataloader
    train_dataloader = dataset_loader.get_dataloader()

    # Iterate over a single batch to verify the loader works
    for batch in train_dataloader:
        logging.info({key: value.shape for key, value in batch.items()})
        break