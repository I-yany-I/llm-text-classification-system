from transformers import pipeline
import re


def load_llm():

    classifier = pipeline(
        "text-generation",
        model="Qwen/Qwen2-1.5B-Instruct",
        device_map="auto"
    )

    return classifier


def classify_news(text, llm):

    prompt = f"""
Classify the following news into one category:

0 = World
1 = Sports
2 = Business
3 = Sci/Tech

Text:
{text}

Return ONLY the category number.
"""

    result = llm(
        prompt,
        max_new_tokens=5,
        do_sample=False
    )

    output = result[0]["generated_text"]

    match = re.search(r"[0-3]", output)

    if match:
        return int(match.group())

    return -1

