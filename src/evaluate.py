"""
Evaluation script for the Code Summarizer project.
Calculates BLEU and ROUGE scores on the test dataset split.
"""

import torch
import torch.nn as nn
import pickle
import math
import os
import sys
import json

# Add src to path for internal imports
sys.path.append(os.path.join(os.getcwd(), 'src'))

from nltk.translate.bleu_score import sentence_bleu
# Rouge scorer requirement
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("rouge_score not found. This script requires it for ROUGE metrics.")
    # In a real environment, you'd pip install here, but we assume it's handled.
    from rouge_score import rouge_scorer

from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq
from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.text_tokenizer import TextTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_all():
    """Loads the trained model and tokenizers."""
    with open("models/code_tokenizer.pkl", "rb") as f:
        code_tok = pickle.load(f)

    with open("models/text_tokenizer.pkl", "rb") as f:
        text_tok = pickle.load(f)

    # Reconstruct architecture
    encoder = Encoder(len(code_tok.word2idx), 64, 128) # Updated dimensions to match train.py
    decoder = Decoder(len(text_tok.word2idx), 64, 128)
    attention = Attention(128)

    model = Seq2Seq(encoder, decoder, attention).to(DEVICE)
    model.load_state_dict(torch.load("models/model.pt", map_location=DEVICE))
    model.eval()

    return model, code_tok, text_tok

def evaluate(samples=100):
    """
    Evaluates the model on the test split.
    Args:
        samples (int): Number of samples to evaluate for metrics.
    """
    if not os.path.exists("models/model.pt"):
        print("Error: models/model.pt not found. Please run training first.")
        return

    model, code_tok, text_tok = load_all()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    
    # Path to test files split earlier
    test_files = [os.path.join("data/test", f) for f in os.listdir("data/test") if f.endswith(".jsonl")]
    
    if not test_files:
        print("Error: No test files found in data/test. Please run scripts/split_data.py first.")
        return

    print(f"Evaluating on {samples} samples from data/test...")
    count = 0
    for path in test_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if count >= samples: break
                
                obj = json.loads(line)
                code_raw = " ".join(obj["code_tokens"])
                summary_raw = " ".join(obj["docstring_tokens"])
                
                reference = summary_raw.lower().split()
                
                # Tokenize input
                src_ids = code_tok.encode(code_raw)
                src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Greedy decoding for evaluation
                    enc_out, hidden = model.encoder(src)
                    token = torch.tensor([[text_tok.word2idx["<sos>"]]]).to(DEVICE)
                    preds = []
                    
                    for _ in range(60): # Max sequence length
                        attn = model.attention(hidden, enc_out)
                        context = torch.sum(enc_out * attn.unsqueeze(2), dim=1)
                        out, hidden = model.decoder(token, hidden, context)
                        token = out.argmax(2)
                        
                        idx = token.item()
                        if idx == text_tok.word2idx["<eos>"]: break
                        preds.append(idx)
                    
                    # Convert IDs back to words
                    prediction_text = text_tok.decode(preds)
                    prediction_tokens = prediction_text.split()
                    
                    # Compute BLEU (nltk)
                    bleu_scores.append(sentence_bleu([reference], prediction_tokens))
                    
                    # Compute ROUGE (rouge_scorer)
                    scores = scorer.score(summary_raw, prediction_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                    
                    count += 1
        if count >= samples: break

    # Summary of metrics
    if count > 0:
        print("-" * 30)
        print(f"Evaluated {count} samples.")
        print(f"BLEU Score (avg): {sum(bleu_scores)/len(bleu_scores):.4f}")
        print(f"ROUGE-1 (avg): {sum(rouge1_scores)/len(rouge1_scores):.4f}")
        print(f"ROUGE-L (avg): {sum(rougeL_scores)/len(rougeL_scores):.4f}")
    else:
        print("No samples were evaluated.")

if __name__ == "__main__":
    evaluate()
