# ML-Based Python Code Summarization

This project implements a sequence-to-sequence model with attention to generate natural language summaries for Python code snippets.

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
