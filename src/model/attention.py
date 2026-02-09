import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, hidden)
        self.v = nn.Linear(hidden, 1)

    def forward(self, hidden, encoder_outputs):
        # hidden is (num_layers, batch, hidden_size)
        # Use the last layer's hidden state
        h = hidden[0][-1].unsqueeze(1) if isinstance(hidden, tuple) else hidden[-1].unsqueeze(1)
        
        # encoder_outputs: [batch, seq_len, hidden]
        # h: [batch, 1, hidden]
        
        # Construct energy
        # Repeat h to match encoder_outputs sequence length
        h_rep = h.repeat(1, encoder_outputs.size(1), 1)
        
        energy = torch.tanh(self.attn(torch.cat((h_rep, encoder_outputs), dim=2)))
        # energy: [batch, seq_len, hidden]
        
        attention = torch.softmax(self.v(energy).squeeze(2), dim=1)
        # attention: [batch, seq_len]
        
        return attention
