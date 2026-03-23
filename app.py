import gradio as gr

from src.inference.predictors import LABEL_MAP, METHODS, get_model, parse_batch_texts


METHOD_CHOICES = [method.upper() for method in METHODS]


def format_single_result(result):
    lines = [
        f"Method: {result.method.upper()}",
        f"Predicted Label: {result.label_id}",
        f"Category: {result.label_name}",
    ]

    if result.examples:
        lines.append("")
        lines.append("Retrieved Similar News:")
        lines.extend(f"- {example}" for example in result.examples)

    return "\n".join(lines)


def run_single_prediction(method_name: str, text: str):
    text = text.strip()
    if not text:
        return "Please enter one piece of news text."

    predictor = get_model(method_name.lower())
    result = predictor.predict_with_details(text)
    return format_single_result(result)


def run_batch_prediction(method_name: str, raw_text: str):
    texts = parse_batch_texts(raw_text)
    if not texts:
        return [], "Please enter one text per line for batch prediction."

    predictor = get_model(method_name.lower())
    predictions = predictor.predict_batch(texts)

    rows = [
        [text, label_id, LABEL_MAP.get(label_id, "Unknown")]
        for text, label_id in zip(texts, predictions)
    ]

    return rows, f"Predicted {len(rows)} texts with {method_name.upper()}."


with gr.Blocks(title="News Classification Comparison") as demo:
    gr.Markdown(
        """
        # News Classification Comparison
        Compare BERT, LoRA, Prompt, and RAG on the same input.

        Labels:
        - `0` = World
        - `1` = Sports
        - `2` = Business
        - `3` = Sci/Tech
        """
    )

    method_selector = gr.Dropdown(
        choices=METHOD_CHOICES,
        value="BERT",
        label="Prediction Method",
    )

    with gr.Tab("Single Prediction"):
        single_text = gr.Textbox(
            lines=6,
            placeholder="Enter one news text here...",
            label="News Text",
        )
        single_button = gr.Button("Predict")
        single_output = gr.Textbox(label="Prediction Result", lines=12)

    with gr.Tab("Batch Prediction"):
        batch_text = gr.Textbox(
            lines=12,
            placeholder="Enter one news text per line...",
            label="Batch Input",
        )
        batch_button = gr.Button("Run Batch Prediction")
        batch_status = gr.Textbox(label="Status", lines=2)
        batch_output = gr.Dataframe(
            headers=["text", "label_id", "label_name"],
            datatype=["str", "number", "str"],
            label="Batch Prediction Results",
        )

    single_button.click(
        fn=run_single_prediction,
        inputs=[method_selector, single_text],
        outputs=single_output,
    )

    batch_button.click(
        fn=run_batch_prediction,
        inputs=[method_selector, batch_text],
        outputs=[batch_output, batch_status],
    )


if __name__ == "__main__":
    demo.launch()
