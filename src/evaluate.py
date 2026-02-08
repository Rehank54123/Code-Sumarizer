import torch
import torch.nn as nn
import pickle
import math
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from nltk.translate.bleu_score import sentence_bleu
# You might need to install rouge-score: pip install rouge-score
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("rouge_score not found. Installing...")
    os.system("pip install rouge-score")
    from rouge_score import rouge_scorer

from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq
from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.text_tokenizer import TextTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_all():
    with open("models/code_tokenizer.pkl", "rb") as f:
        code_tok = pickle.load(f)

    with open("models/text_tokenizer.pkl", "rb") as f:
        text_tok = pickle.load(f)

    encoder = Encoder(len(code_tok.word2idx), 128, 256)
    decoder = Decoder(len(text_tok.word2idx), 128, 256)
    attention = Attention(256)

    model = Seq2Seq(encoder, decoder, attention).to(DEVICE)
    model.load_state_dict(torch.load("models/model.pt", map_location=DEVICE))
    model.eval()

    return model, code_tok, text_tok

def evaluate(samples=100):
    model, code_tok, text_tok = load_all()
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []
    losses = []
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    test_files = [os.path.join("data/test", f) for f in os.listdir("data/test") if f.endswith(".jsonl")]
    
    count = 0
    for path in test_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if count >= samples: break
                
                obj = json.loads(line)
                code_raw = " ".join(obj["code_tokens"])
                summary_raw = " ".join(obj["docstring_tokens"])
                
                reference = summary_raw.lower().split()
                
                # Pre-processing same as training
                # ... (should be consistent)
                
                src_ids = code_tok.encode(code_raw)
                trg_ids = text_tok.encode(summary_raw)
                
                src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
                trg = torch.tensor(trg_ids).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    # Simplified evaluation for metrics
                    enc_out, hidden = model.encoder(src)
                    token = torch.tensor([[text_tok.word2idx["<sos>"]]]).to(DEVICE)
                    preds = []
                    
                    # Manual greedy decode for BLEU/ROUGE
                    for _ in range(60):
                        attn = model.attention(hidden, enc_out)
                        context = torch.sum(enc_out * attn.unsqueeze(2), dim=1)
                        out, hidden = model.decoder(token, hidden, context)
                        token = out.argmax(2)
                        
                        idx = token.item()
                        if idx == text_tok.word2idx["<eos>"]: break
                        preds.append(idx)
                    
                    prediction_text = text_tok.decode(preds)
                    prediction_tokens = prediction_text.split()
                    
                    # BLEU
                    bleu_scores.append(sentence_bleu([reference], prediction_tokens))
                    
                    # ROUGE
                    scores = scorer.score(summary_raw, prediction_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                    
                    # Loss check (requires teacher forcing style like in evaluate.py original)
                    # We'll skip complex loss here or use model(src, trg)
                    count += 1
        if count >= samples: break

    print(f"Evaluated {count} samples.")
    print(f"BLEU Score (avg): {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"ROUGE-1 (avg): {sum(rouge1_scores)/len(rouge1_scores):.4f}")
    print(f"ROUGE-L (avg): {sum(rougeL_scores)/len(rougeL_scores):.4f}")

if __name__ == "__main__":
    evaluate()
