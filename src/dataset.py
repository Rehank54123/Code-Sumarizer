import json
import torch
import re
import os
import glob
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(
        self,
        data_dir,
        code_tok,
        text_tok,
        max_code_len=300,
        max_sum_len=60,
        is_directory=True
    ):
        self.code_tok = code_tok
        self.text_tok = text_tok
        self.max_code_len = max_code_len
        self.max_sum_len = max_sum_len
        self.samples = []

        if is_directory:
            files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        else:
            files = [data_dir]

        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    # We store the lines to parse them on demand if the dataset is huge,
                    # but for now, let's pre-parse but keep it efficient.
                    # If memory is still an issue, we can store file offsets.
                    obj = json.loads(line)

                    code_tokens = obj.get("code_tokens", [])
                    sum_tokens = obj.get("docstring_tokens", [])

                    if len(code_tokens) < 10 or len(sum_tokens) < 5 or len(code_tokens) > max_code_len:
                        continue

                    code = " ".join(code_tokens)
                    summary = " ".join(sum_tokens)

                    # FUNCTION NAME INJECTION
                    match = re.search(r"def\s+([a-zA-Z_]\w*)", code)
                    if match:
                        fname = match.group(1)
                        code = f"FUNC_NAME {fname} {code}"

                    self.samples.append((code, summary))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, summary = self.samples[idx]
        
        code_ids = self.code_tok.encode(code)[:self.max_code_len]
        sum_ids = self.text_tok.encode(summary)[:self.max_sum_len]
        
        return code_ids, sum_ids

def collate_fn(batch):
    codes, sums = zip(*batch)

    max_c = max(len(c) for c in codes)
    max_s = max(len(s) for s in sums)

    padded_codes = [c + [0] * (max_c - len(c)) for c in codes]
    padded_sums = [s + [0] * (max_s - len(s)) for s in sums]

    return torch.tensor(padded_codes), torch.tensor(padded_sums)
