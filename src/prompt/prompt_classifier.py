import re

from src.llm.shared_pipeline import load_shared_generation_pipeline


def load_llm():
    return load_shared_generation_pipeline()


def classify_news(text, llm):
    prompt = f"""
You are a strict AG News classifier.
Choose exactly one label id:
0 = World (politics, government, international affairs, wars)
1 = Sports (games, teams, athletes, tournaments)
2 = Business (companies, finance, markets, economy, deals)
3 = Sci/Tech (technology, science, software, gadgets, research)

Examples:
Text: The UN Security Council approved a new resolution on Middle East peace talks.
Label: 0
Text: Manchester United won 3-1 in the Premier League opener.
Label: 1
Text: Apple reported quarterly revenue growth driven by iPhone sales.
Label: 2
Text: Researchers released a new open-source large language model for coding.
Label: 3

Now classify:
Text: {text}

Output format rules:
- Output one and only one character: 0 or 1 or 2 or 3
- Do not output words, punctuation, or explanation
Answer:
"""

    result = llm(
        prompt,
        max_new_tokens=3,
        do_sample=False,
        return_full_text=False
    )

    output = result[0]["generated_text"].strip()

    match = re.search(r"[0-3]", output)

    if match:
        return int(match.group())

    return -1

