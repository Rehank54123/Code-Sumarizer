class TextTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        self.idx2word = {i:w for w,i in self.word2idx.items()}

    def tokenize(self, text):
        return text.lower().split()

    def build_vocab(self, texts):
        for text in texts:
            for word in self.tokenize(text):
                if word not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word

    def encode(self, text):
        return [1] + [self.word2idx.get(w,3) for w in self.tokenize(text)] + [2]

    def decode(self, tokens):
        return " ".join([self.idx2word.get(t,"") for t in tokens])
