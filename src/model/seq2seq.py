import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def forward(self, src, trg):
        enc_outputs, hidden = self.encoder(src)
        outputs = []

        for t in range(trg.size(1)-1):
            attn = self.attention(hidden, enc_outputs)
            context = torch.sum(enc_outputs * attn.unsqueeze(2), dim=1)
            out, hidden = self.decoder(trg[:,t].unsqueeze(1), hidden, context)
            outputs.append(out)

        return torch.cat(outputs, dim=1)
