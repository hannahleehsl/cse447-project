#!/usr/bin/env python3
"""
train_gpt2.py (NO datasets dependency)

Fine-tunes GPT-2 on one or more local UTF-8 text files (or directories of text files),
then saves a local model to:

  work/FINAL/

This is meant to be run OFFLINE (not from predict.sh). Your inference script
(src/myprogram.py) will then load from work/FINAL.

Usage examples (run from project root):
  python src/train_gpt2.py --out_dir work/FINAL --train_text dataset/train.txt
  python src/train_gpt2.py --out_dir work/FINAL --train_text dataset/linguatools_wiki/
"""

import os
import math
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


def iter_texts(paths: List[str]) -> Iterable[str]:
    """Yield full text strings from files or directories (best-effort UTF-8)."""
    for p in paths:
        if not os.path.exists(p):
            continue
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in files:
                    fp = os.path.join(root, fn)
                    yield from iter_texts([fp])
        else:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    yield f.read()
            except Exception:
                continue


class TokenBlockDataset(Dataset):
    """
    Turns a long token stream into fixed-length blocks for causal LM training.
    Each item returns dict(input_ids=..., attention_mask=..., labels=...).
    """

    def __init__(self, token_ids: List[int], block_size: int):
        self.block_size = block_size
        # drop remainder for simplicity
        n_blocks = len(token_ids) // block_size
        self.data = [token_ids[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x),
            "labels": x.clone(),
        }


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base_model", default="gpt2", help="HF base model id to start from")
    parser.add_argument("--train_text", nargs="+", required=True, help="text file(s) or directory(ies)")
    parser.add_argument("--out_dir", default="work/FINAL", help="where to save the fine-tuned model")
    parser.add_argument("--block_size", type=int, default=256, help="token block length")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=0, help="if >0, limit total optimizer steps")
    parser.add_argument("--seed", type=int, default=0, help="random seed (0 = random)")
    args = parser.parse_args()

    if args.seed != 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading tokenizer/model:", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    device = get_device()
    print("Using device:", device)
    model.to(device)
    model.train()

    # Read and concatenate all training text
    print("Reading training text...")
    full_text_parts = []
    total_chars = 0
    for t in iter_texts(args.train_text):
        if t and not t.isspace():
            full_text_parts.append(t)
            total_chars += len(t)
    if not full_text_parts:
        raise SystemExit("No readable training text found in --train_text paths.")

    full_text = "\n".join(full_text_parts)
    print("Total chars:", total_chars)

    # Tokenize once into a long stream
    print("Tokenizing...")
    enc = tok(full_text, add_special_tokens=True)
    token_ids = enc["input_ids"]
    print("Total tokens:", len(token_ids))

    if len(token_ids) < args.block_size * 10:
        print("WARNING: very small training corpus; may overfit or not improve.")

    dataset = TokenBlockDataset(token_ids, block_size=args.block_size)
    print("Num blocks:", len(dataset))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # simple training loop
    step = 0
    accum = 0
    running_loss = 0.0

    print("Training...")
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()

            running_loss += loss.item()
            accum += 1

            if accum >= args.grad_accum:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accum = 0

                if step % 50 == 0:
                    avg = running_loss / 50
                    print(f"epoch {epoch+1} step {step} avg_loss {avg:.4f}")
                    running_loss = 0.0

                if args.max_steps and step >= args.max_steps:
                    break

        if args.max_steps and step >= args.max_steps:
            break

    print("Saving model to:", args.out_dir)
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Done. Files in out_dir:", os.listdir(args.out_dir))


if __name__ == "__main__":
    main()