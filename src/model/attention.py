import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[0][-1].unsqueeze(1)
        energy = torch.tanh(self.attn(
            torch.cat((hidden.repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2)
        ))
        return torch.softmax(self.v(energy).squeeze(2), dim=1)
