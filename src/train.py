"""
Training script for the Code Summarizer project.
Handles vocabulary building, dataset loading, and the training/validation loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
import sys
import json

# Add src to path if needed for local execution
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.text_tokenizer import TextTokenizer
from dataset import CodeDataset, collate_fn
from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq

# Setup device (GPU/CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Hyperparameters
EPOCHS = 10
BATCH_SIZE = 8 # Reduced further for maximum stability
EMBED = 64
HIDDEN = 128
LR = 0.001
MAX_CODE_LEN = 128 # Reduced sequence length to save memory
MAX_SUM_LEN = 40

def train():
    """Main training routine."""
    # ---------- TOKENIZERS ----------
    print("Initializing tokenizers...")
    code_tok = CodeTokenizer(max_vocab=8000)
    text_tok = TextTokenizer()

    # ---------- BUILD VOCAB ----------
    # Builds vocabulary from a representative subset of the training data
    print("Building vocab...")
    train_files = [os.path.join("data/train", f) for f in os.listdir("data/train") if f.endswith(".jsonl")]
    
    all_codes = []
    all_summaries = []
    
    count = 0
    for path in train_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_codes.append(" ".join(obj["code_tokens"]))
                all_summaries.append(" ".join(obj["docstring_tokens"]))
                count += 1
                if count > 50000: break # Use 50k samples for vocab
        if count > 50000: break

    code_tok.build_vocab(all_codes)
    text_tok.build_vocab(all_summaries)

    # ---------- DATASETS ----------
    print("Indexing datasets (takes a minute for large files)...")
    # Limiting to a subset for reasonable training time on CPU
    TRAIN_LIMIT = 50000 
    VALID_LIMIT = 5000 

    train_dataset = CodeDataset("data/train", code_tok, text_tok, max_code_len=MAX_CODE_LEN, max_sum_len=MAX_SUM_LEN, limit=TRAIN_LIMIT)
    valid_dataset = CodeDataset("data/valid", code_tok, text_tok, max_code_len=MAX_CODE_LEN, max_sum_len=MAX_SUM_LEN, limit=VALID_LIMIT)

    # Use num_workers=0 to avoid Windows multiprocessing issues with file offsets
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ---------- MODEL ----------
    encoder = Encoder(len(code_tok.word2idx), EMBED, HIDDEN)
    decoder = Decoder(len(text_tok.word2idx), EMBED, HIDDEN)
    attention = Attention(HIDDEN)

    model = Seq2Seq(encoder, decoder, attention).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <pad> token index 0

    best_valid_loss = float('inf')

    # ---------- TRAIN LOOP ----------
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for i, (code, summary) in enumerate(train_loader):
            code, summary = code.to(DEVICE), summary.to(DEVICE)

            optimizer.zero_grad()
            output = model(code, summary)

            output = output.view(-1, output.size(-1))
            target = summary[:, 1:].contiguous().view(-1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # ---------- VALIDATION LOOP ----------
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for code, summary in valid_loader:
                code, summary = code.to(DEVICE), summary.to(DEVICE)
                output = model(code, summary)
                
                output = output.view(-1, output.size(-1))
                target = summary[:, 1:].contiguous().view(-1)
                
                loss = criterion(output, target)
                total_valid_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_valid = total_valid_loss / len(valid_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Valid Loss: {avg_valid:.4f}")

        # Save model if validation loss improves
        if avg_valid < best_valid_loss:
            best_valid_loss = avg_valid
            if not os.path.exists("models"): os.makedirs("models")
            torch.save(model.state_dict(), "models/model.pt")
            print("--- Best Model Saved ---")

    # ---------- SAVE TOKENIZERS ----------
    print("Saving tokenizers...")
    with open("models/code_tokenizer.pkl", "wb") as f:
        pickle.dump(code_tok, f)
    with open("models/text_tokenizer.pkl", "wb") as f:
        pickle.dump(text_tok, f)

    print("Training complete.")

if __name__ == "__main__":
    train()
