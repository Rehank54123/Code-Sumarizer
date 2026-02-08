import torch
import pickle
from tokenizer.code_tokenizer import CodeTokenizer
from tokenizer.text_tokenizer import TextTokenizer
from model.encoder import Encoder
from model.decoder import Decoder
from model.attention import Attention
from model.seq2seq import Seq2Seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Load tokenizers (must rebuild vocab SAME WAY as training)
# --------------------
def load_tokenizers():
    with open("code_tokenizer.pkl", "rb") as f:
        code_tok = pickle.load(f)

    with open("text_tokenizer.pkl", "rb") as f:
        text_tok = pickle.load(f)

    return code_tok, text_tok


# --------------------
# Load trained model
# --------------------
def load_model(code_vocab, text_vocab):
    encoder = Encoder(code_vocab, 128, 256)
    decoder = Decoder(text_vocab, 128, 256)
    attention = Attention(256)

    model = Seq2Seq(encoder, decoder, attention).to(DEVICE)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.eval()
    return model


# --------------------
# Greedy decoding
# --------------------
def summarize_code(code, max_len=30):
    code_tok, text_tok = load_tokenizers()
    model = load_model(len(code_tok.word2idx), len(text_tok.word2idx))

    # Encode code
    code_ids = code_tok.encode(code)
    src = torch.tensor(code_ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc_outputs, hidden = model.encoder(src)

        trg_token = torch.tensor([[text_tok.word2idx["<sos>"]]]).to(DEVICE)
        output_tokens = []

        for _ in range(max_len):
            attn = model.attention(hidden, enc_outputs)
            context = torch.sum(enc_outputs * attn.unsqueeze(2), dim=1)

            out, hidden = model.decoder(trg_token, hidden, context)
            next_token = out.argmax(2).item()

            if next_token == text_tok.word2idx["<eos>"]:
                break

            output_tokens.append(next_token)
            trg_token = torch.tensor([[next_token]]).to(DEVICE)

    return text_tok.decode(output_tokens)


# --------------------
# Manual test
# --------------------
if __name__ == "__main__":
    test_code = """
def add(a, b):
    return a + b
"""
    print("Generated summary:")
    print(summarize_code(test_code))
