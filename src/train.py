import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
import sys

# Add src to path if needed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.text_tokenizer import TextTokenizer
from dataset import CodeDataset, collate_fn
from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
BATCH_SIZE = 16 # Reduced to avoid OOM
EMBED = 64     # Reduced
HIDDEN = 128   # Reduced
LR = 0.001

def train():
    # ---------- TOKENIZERS ----------
    code_tok = CodeTokenizer(max_vocab=8000)
    text_tok = TextTokenizer()

    # ---------- BUILD VOCAB (using a subset of train for speed if necessary, or full) ----------
    print("Building vocab...")
    train_files = [os.path.join("data/train", f) for f in os.listdir("data/train") if f.endswith(".jsonl")]
    
    all_codes = []
    all_summaries = []
    
    # Take a limit for vocab building to speed up
    count = 0
    for path in train_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_codes.append(" ".join(obj["code_tokens"]))
                all_summaries.append(" ".join(obj["docstring_tokens"]))
                count += 1
                if count > 50000: break # Sufficient for vocab
        if count > 50000: break

    code_tok.build_vocab(all_codes)
    text_tok.build_vocab(all_summaries)

    # ---------- DATASETS ----------
    train_dataset = CodeDataset("data/train", code_tok, text_tok)
    valid_dataset = CodeDataset("data/valid", code_tok, text_tok)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ---------- MODEL ----------
    encoder = Encoder(len(code_tok.word2idx), EMBED, HIDDEN)
    decoder = Decoder(len(text_tok.word2idx), EMBED, HIDDEN)
    attention = Attention(HIDDEN)

    model = Seq2Seq(encoder, decoder, attention).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_valid_loss = float('inf')

    # ---------- TRAIN LOOP ----------
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

        # ---------- VALIDATION ----------
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for code, summary in valid_loader:
                code, summary = code.to(DEVICE), summary.to(DEVICE)
                output = model(code, summary, teacher_forcing_ratio=0) # No teacher forcing in eval
                
                output = output.view(-1, output.size(-1))
                target = summary[:, 1:].contiguous().view(-1)
                
                loss = criterion(output, target)
                total_valid_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_valid = total_valid_loss / len(valid_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Valid Loss: {avg_valid:.4f}")

        if avg_valid < best_valid_loss:
            best_valid_loss = avg_valid
            torch.save(model.state_dict(), "models/model.pt")
            print("--- Model Saved ---")

    # ---------- SAVE TOKENIZERS ----------
    with open("models/code_tokenizer.pkl", "wb") as f:
        pickle.dump(code_tok, f)
    with open("models/text_tokenizer.pkl", "wb") as f:
        pickle.dump(text_tok, f)

    print("Training complete.")

if __name__ == "__main__":
    import json # Import inside for vocab building
    train()
