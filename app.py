import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
from tkinter import ttk
from src.summarize import summarize

root = tk.Tk()
root.title("Python Code Summarizer")
root.geometry("900x600")

style = ttk.Style()
style.theme_use("clam")

ttk.Label(
    root,
    text="ML-Based Python Code Summarization",
    font=("Segoe UI", 18, "bold")
).pack(pady=10)

ttk.Label(root, text="Enter Python Code:").pack(anchor="w", padx=20)

code_box = tk.Text(root, height=15, font=("Consolas", 11))
code_box.pack(fill="x", padx=20)

def run_summary():
    code = code_box.get("1.0", tk.END)
    output.delete("1.0", tk.END)
    output.insert(tk.END, summarize(code))

ttk.Button(root, text="Summarize Code", command=run_summary).pack(pady=10)

ttk.Label(root, text="Generated Summary:").pack(anchor="w", padx=20)

output = tk.Text(root, height=6, font=("Segoe UI", 11))
output.pack(fill="x", padx=20)

root.mainloop()
