import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, hidden = self.lstm(x)
        return outputs, hidden
