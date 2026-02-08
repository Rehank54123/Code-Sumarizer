import torch               # 🔴 THIS LINE WAS MISSING
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab, embed, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed + hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x, hidden, context):
        x = self.embedding(x)
        x = torch.cat((x, context.unsqueeze(1)), dim=2)
        output, hidden = self.lstm(x, hidden)
        return self.fc(output), hidden
