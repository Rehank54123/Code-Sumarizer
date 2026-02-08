import re
from collections import Counter

class CodeTokenizer:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.word2idx = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        self.idx2word = {i:w for w,i in self.word2idx.items()}

    def tokenize(self, code):
        return re.findall(r"[A-Za-z_]+|==|!=|<=|>=|[0-9]+|[^\s]", code)

    def build_vocab(self, codes):
        counter = Counter()
        for code in codes:
            counter.update(self.tokenize(code))

        for word,_ in counter.most_common(self.max_vocab):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, code):
        return [self.word2idx.get(t, 3) for t in self.tokenize(code)]
