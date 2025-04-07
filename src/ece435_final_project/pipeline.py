import torch
from transformers import pipeline

class LlamaPipeline:
    def __init__(self):
        self.model_id = "meta-llama/Llama-3.2-1B"
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        output = self.pipeline(
            prompt,
            max_length=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )
        if output is None:
            raise RuntimeError("Model output is None")
        generated_text = output[0]['generated_text']
        return generated_text
    
if __name__ == "__main__":
    import os
    from huggingface_hub import login
    login(token=os.getenv("HF_TOKEN"))

    llama = LlamaPipeline()
    prompt = "The key to life is"
    response = llama.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
