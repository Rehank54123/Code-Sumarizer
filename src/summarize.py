"""
Inference script for the Code Summarizer project.
Generates a human-readable summary for a given Python code snippet.
"""

import argparse
import torch
import pickle
import os
import sys

# Add src to path for internal imports
sys.path.append(os.path.join(os.getcwd(), 'src'))

from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq

# Setup device (GPU/CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_all():
    """Loads tokenizers and the best model weights."""
    BASE_DIR = os.getcwd() 
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    if not os.path.exists(os.path.join(MODEL_DIR, "model.pt")):
        raise FileNotFoundError("Model file not found in models/. Please run training first.")

    with open(os.path.join(MODEL_DIR, "code_tokenizer.pkl"), "rb") as f:
        code_tok = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "text_tokenizer.pkl"), "rb") as f:
        text_tok = pickle.load(f)

    # Reconstruct architecture with current hyperparameters
    encoder = Encoder(len(code_tok.word2idx), 64, 128)
    decoder = Decoder(len(text_tok.word2idx), 64, 128)
    attention = Attention(128)

    model = Seq2Seq(encoder, decoder, attention)
    model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "model.pt"), map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    return model, code_tok, text_tok

def summarize(code, max_len=30):
    """
    Summarizes a code snippet.
    Args:
        code (str): Raw Python code text to summarize.
        max_len (int): Maximum length of the generated summary.
    Returns:
        str: Generated summary text.
    """
    try:
        model, code_tok, text_tok = load_all()
    except FileNotFoundError as e:
        return str(e)

    # Encode input code
    code_ids = code_tok.encode(code)
    src = torch.tensor(code_ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        token = torch.tensor([[text_tok.word2idx["<sos>"]]]).to(DEVICE)
        result = []

        # Autoregressive decoding (greedy)
        for _ in range(max_len):
            attn = model.attention(hidden, enc_out)
            context = torch.sum(enc_out * attn.unsqueeze(2), dim=1)
            out, hidden = model.decoder(token, hidden, context)
            next_token = out.argmax(2).item()

            if next_token == text_tok.word2idx["<eos>"]:
                break

            result.append(next_token)
            token = torch.tensor([[next_token]]).to(DEVICE)

    return text_tok.decode(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary for code input.")
    parser.add_argument("--input", type=str, required=True, help="Python code string to summarize")
    args = parser.parse_args()

    print("\n[Input Code]:")
    print("-" * 20)
    print(args.input)
    print("-" * 20)
    print("\n[Generated Summary]:")
    print(summarize(args.input))
