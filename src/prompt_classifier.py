from transformers import pipeline


def load_llm():

    classifier = pipeline(
        "text-generation",
        model="gpt2"
    )

    return classifier


def classify_news(text):

    classifier = load_llm()

    prompt = f"""
Classify the following news into one category:

World
Sports
Business
Sci/Tech

Text:
{text}

Category:
"""

    result = classifier(
        prompt,
        max_new_tokens=10
    )

    print(result)


if __name__ == "__main__":

    text = "Tesla launches new AI robot"

    classify_news(text)
