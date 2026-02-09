"""
Dataset class for loading Python code and their natural language summaries.
Implements optimized lazy loading with file offsets to support large datasets on limited RAM.
"""

import json
import torch
import re
import os
import glob
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    """
    Optimized Lazy-loading PyTorch Dataset.
    Stores file offsets for O(1) random access to lines in large JSONL files.
    """
    def __init__(
        self,
        data_dir,
        code_tok,
        text_tok,
        max_code_len=300,
        max_sum_len=60,
        is_directory=True,
        limit=None
    ):
        self.code_tok = code_tok
        self.text_tok = text_tok
        self.max_code_len = max_code_len
        self.max_sum_len = max_sum_len
        
        # Store global index -> (file_path, byte_offset)
        self.index_map = []

        if is_directory:
            files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        else:
            files = [data_dir]

        print(f"Indexing byte offsets for {data_dir}...")
        for path in files:
            with open(path, 'rb') as f:
                offset = 0
                for line in f:
                    self.index_map.append((path, offset))
                    offset += len(line)
                    if limit and len(self.index_map) >= limit:
                        break
            if limit and len(self.index_map) >= limit:
                break

        print(f"[Dataset] Indexed {len(self.index_map)} samples.")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        """Random access load using byte offsets."""
        file_path, offset = self.index_map[idx]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
        
        if not line:
            return [0], [0]

        obj = json.loads(line)
        code_tokens = obj.get("code_tokens", [])
        sum_tokens = obj.get("docstring_tokens", [])

        code = " ".join(code_tokens)
        summary = " ".join(sum_tokens)

        # Function Name Injection
        match = re.search(r"def\s+([a-zA-Z_]\w*)", code)
        if match:
            fname = match.group(1)
            code = f"FUNC_NAME {fname} {code}"

        code_ids = self.code_tok.encode(code)[:self.max_code_len]
        sum_ids = self.text_tok.encode(summary)[:self.max_sum_len]
        
        return code_ids, sum_ids

def collate_fn(batch):
    """Pads sequences in the batch."""
    codes, sums = zip(*batch)

    max_c = max(len(c) for c in codes)
    max_s = max(len(s) for s in sums)

    padded_codes = [c + [0] * (max_c - len(c)) for c in codes]
    padded_sums = [s + [0] * (max_s - len(s)) for s in sums]

    return torch.tensor(padded_codes), torch.tensor(padded_sums)
