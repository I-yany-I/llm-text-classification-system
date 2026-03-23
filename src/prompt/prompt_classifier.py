import re

from src.llm.shared_pipeline import load_shared_generation_pipeline


def load_llm():
    return load_shared_generation_pipeline()


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
        do_sample=False,
        return_full_text=False
    )

    output = result[0]["generated_text"].strip()

    match = re.search(r"\b[0-3]\b", output)

    if match:
        return int(match.group())

    return -1

