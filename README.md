# llm-text-classification-system

Text classification system for comparing BERT, LoRA, Prompt, and RAG methods on AG News.

Features:
- Train and save final BERT and LoRA models
- Compare BERT, LoRA, Prompt, and RAG in one evaluation script
- Visualize accuracy and weighted F1 results
- Switch prediction methods in a Gradio web app
- Run single-text or batch-text prediction

Directory structure:
- `train.py`: root training entrypoint
- `evaluate.py`: root evaluation entrypoint
- `app.py`: Gradio app entrypoint
- `src/inference/`: unified prediction interface
- `src/models/`: BERT and LoRA model definitions
- `src/training/`: training logic
- `src/prompt/`: prompt-based classification logic
- `src/rag/`: RAG pipeline and vector retrieval logic
- `artifacts/models/`: final trained models
- `artifacts/checkpoints/`: training checkpoints
- `artifacts/plots/`: evaluation plots
- `artifacts/predictions/`: metric summaries
- `data/`: optional local dataset storage
- `vector_store/`: cached FAISS index and embeddings

Run commands:
```bash
pip install -r requirements.txt
python train.py --model all --epochs 1
python evaluate.py
python app.py
```

Useful variants:
```bash
python train.py --model bert --epochs 1
python train.py --model lora --epochs 1
```
