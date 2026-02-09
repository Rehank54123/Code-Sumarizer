# ML-Based Python Code Summarization

This project implements a sequence-to-sequence (Seq2Seq) model with Bahdanau Attention to generate natural language summaries for Python code snippets.

## Model Architecture: Seq2Seq + Attention
The core of this project is an **LSTM-based Encoder-Decoder architecture** enhanced with an **Attention Mechanism**.

### Why This Model?
Code summarization is a challenging task because source code contains long-range dependencies and a high density of information. Our model is superior to several traditional approaches:

1.  **Vs. Vanilla Seq2Seq (RNN/LSTM)**: 
    *   Standard models compress the entire code into a single "context vector." This creates a bottleneck where information is lost for longer functions.
    *   **Our Advantage**: Our **Attention layer** allows the decoder to "look back" at specific keywords or variable names in the original code while generating each word of the summary, ensuring no critical context is missed.

2.  **Vs. Transformer Models (e.g., CodeBERT, GPT)**:
    *   While Transformers are the current state-of-the-art, they are extremely resource-intensive and require millions of parameters.
    *   **Our Advantage**: This LSTM+Attention model provides **high precision with significantly faster training** (optimized to run in minutes/hours rather than days) and is specifically tuned for the local hardware constraints while maintaining professional-grade BLEU scores.

3.  **Vs. Rule-Based Systems**:
    *   Rules can only summarize simple patterns.
    *   **Our Advantage**: This model is **probalistic and semantic**, meaning it understands the *intent* behind the code logic rather than just the syntax.

### Key Alternatives in the Field
*   **Pointer-Generator Networks**: Excellent at "copying" rare variable names directly from code.
*   **Tree-LSTM / Graph Neural Networks (GNNs)**: Good at capturing the structural (AST) hierarchy of code.
*   **Transformers (BERT/T5)**: Best for massive datasets but require heavy GPU support.

## Project Structure
- `src/`: Core source code.
  - `dataset.py`: Data loading and preprocessing logic.
  - `train.py`: Training and validation routines.
  - `evaluate.py`: Performance evaluation (BLEU, ROUGE).
  - `summarize.py`: Inference script for single snippets.
  - `model/`: Neural network architecture components.
  - `tokenizer/`: Custom tokenization for code and text.
- `data/`: Partitioned datasets (train, valid, test).
- `models/`: Saved model artifacts and tokenizers.
- `scripts/`: Utility scripts (e.g., data splitting).

## Setup
Install dependencies:
```powershell
pip install torch nltk rouge-score
```

## Execution Instructions

### 1. Data Splitting
Before training, split the raw data into partitions:
```powershell
python scripts/split_data.py
```

### 2. Training
Train the model on the `data/train` set with validation on `data/valid`:
```powershell
python src/train.py
```
*The best model will be saved to `models/model.pt`.*

### 3. Evaluation
Evaluate the trained model on the `data/test` set:
```powershell
python src/evaluate.py
```
*Outputs BLEU and ROUGE metrics.*

### 4. Inference (Summarization)
Generate a summary for custom code:
```powershell
python src/summarize.py --input "def add(a, b): return a + b"
```

## Performance Optimization
To address the long training times (previously 4+ hours) and memory constraints:
- **Lazy Loading**: Text-based samples are stored as strings and tokenized on-the-fly.
- **Efficient Vocab Building**: Built from a representative 50k subset.
- **Optimized Batching**: Hyperparameters adjusted for fast convergence and system stability.
