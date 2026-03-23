from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"


@lru_cache(maxsize=1)
def load_shared_generation_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
