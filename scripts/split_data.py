import os
import json
import random
import glob

def split_dataset(data_dir, train_dir, valid_dir, test_dir, split_ratio=(0.8, 0.1, 0.1)):
    files = glob.glob(os.path.join(data_dir, "python_train_*.jsonl"))
    # Also include the subset if it exists
    subset_file = os.path.join(data_dir, "python_subset.jsonl")
    if os.path.exists(subset_file):
        files.append(subset_file)
    
    print(f"Found {len(files)} files to split.")
    
    # To avoid loading everything into memory, we split file by file
    # or buffer samples. Given the size, let's do it per file but distribute lines.
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        print(f"Processing {file_name}...")
        
        train_path = os.path.join(train_dir, file_name)
        valid_path = os.path.join(valid_dir, file_name)
        test_path = os.path.join(test_dir, file_name)
        
        with open(file_path, 'r', encoding='utf-8') as f, \
             open(train_path, 'w', encoding='utf-8') as f_train, \
             open(valid_path, 'w', encoding='utf-8') as f_valid, \
             open(test_path, 'w', encoding='utf-8') as f_test:
            
            for line in f:
                r = random.random()
                if r < split_ratio[0]:
                    f_train.write(line)
                elif r < split_ratio[0] + split_ratio[1]:
                    f_valid.write(line)
                else:
                    f_test.write(line)

if __name__ == "__main__":
    split_dataset(
        "data",
        "data/train",
        "data/valid",
        "data/test"
    )
    print("Dataset splitting complete.")
