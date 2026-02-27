#!/usr/bin/env python3
import os
import sys
import math
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Config
# -------------------------
DEFAULT_BASE_MODEL = "gpt2"
TOPK_TOKENS = 800          # take top-k next tokens, then derive 3 distinct chars
BATCH_SIZE = 32            # adjust for speed/memory
MAX_INPUT_LEN = 256        # truncate long prefixes for speed
MODEL_SUBDIR_CANDIDATES = ["FINAL", "final", "model", ""]  # try these under work_dir


# -------------------------
# IO helpers
# -------------------------
def load_test_data(fname: str) -> List[str]:
    data = []
    with open(fname, encoding="utf-8", errors="ignore") as f:
        for line in f:
            data.append(line[:-1])  # strip newline
    return data


def write_pred(preds: List[str], fname: str) -> None:
    with open(fname, "wt", encoding="utf-8") as f:
        for p in preds:
            # EXACTLY 3 chars per line (grader truncates, but keep clean)
            if len(p) < 3:
                p = p + ("#" * (3 - len(p)))
            f.write(p[:3] + "\n")


# -------------------------
# Prediction logic
# -------------------------


def pick_top3_chars_from_logits(
    tokenizer: AutoTokenizer,
    logits: torch.Tensor,
    topk_tokens: int = TOPK_TOKENS
) -> str:
    # Get the top-k token ids
    top_ids = torch.topk(logits, k=topk_tokens).indices.tolist()

    chars: List[str] = []
    for tid in top_ids:
        # Decode the token (this automatically turns Ġ into a normal space!)
        tok_str = tokenizer.decode([tid])
        
        # Skip empty strings
        if len(tok_str) == 0:
            continue
            
        # Grab the very first character of the decoded token
        first_char = tok_str[0]
        
        # Add it to our list if it's unique
        if first_char not in chars:
            chars.append(first_char)
            
        # Stop once we have 3 predictions
        if len(chars) == 3:
            return "".join(chars)

    # Fallback just in case
    while len(chars) < 3:
        chars.append("#")
    return "".join(chars[:3])


# -------------------------
# Model loader
# -------------------------
def resolve_model_dir(work_dir: str) -> str:
    """
    work_dir should be a local directory containing a fine-tuned model.
    We'll try common subdirs like work/FINAL.
    """
    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f"--work_dir must be a local directory, got: {work_dir}")

    # If work_dir itself looks like a HF model dir, accept it
    if os.path.exists(os.path.join(work_dir, "config.json")):
        return work_dir

    # Try common subfolders
    for sub in MODEL_SUBDIR_CANDIDATES:
        cand = os.path.join(work_dir, sub) if sub else work_dir
        if os.path.exists(os.path.join(cand, "config.json")):
            return cand

    raise FileNotFoundError(
        f"Could not find a saved model in {work_dir}.\n"
        f"Expected to see config.json in {work_dir} or one of: "
        f"{', '.join([os.path.join(work_dir, s) for s in MODEL_SUBDIR_CANDIDATES if s])}.\n"
        f"Fine-tune GPT-2 offline and save it into {work_dir}/FINAL (recommended)."
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# Main class
# -------------------------
class MyModel:
    def __init__(self, work_dir: str):
        model_dir = resolve_model_dir(work_dir)

        self.device = get_device()

        # Always load local files only (keeps grading reproducible/offline)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, work_dir: str):
        return MyModel(work_dir)

    def run_pred(self, lines: List[str]) -> List[str]:
        preds: List[str] = []

        # batch for speed
        for i in range(0, len(lines), BATCH_SIZE):
            batch = lines[i:i + BATCH_SIZE]
            batch = [s[-MAX_INPUT_LEN:] for s in batch]  # truncate for speed

            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_LEN,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            with torch.no_grad():
                out = self.model(**enc)
                # logits: (B, T, V); take last non-pad position per row
                logits = out.logits

            # Find last token position for each sequence via attention_mask
            attn = enc.get("attention_mask", None)
            if attn is None:
                # fallback: assume full length
                last_pos = torch.full((logits.size(0),), logits.size(1) - 1, device=self.device, dtype=torch.long)
            else:
                last_pos = attn.sum(dim=1) - 1  # (B,)

            for b in range(logits.size(0)):
                lp = int(last_pos[b].item())
                next_logits = logits[b, lp, :]  # (V,)
                pred3 = pick_top3_chars_from_logits(self.tokenizer, next_logits, topk_tokens=TOPK_TOKENS)
                preds.append(pred3)

        return preds


# -------------------------
# CLI
# -------------------------
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="LOCAL directory containing the saved fine-tuned model", default="work")
    parser.add_argument("--test_data", help="path to test data", default="example/input.txt")
    parser.add_argument("--test_output", help="path to write test predictions", default="pred.txt")

    args = parser.parse_args()

    if args.mode == "train":
        print(
            "Train mode is intentionally not implemented here.\n"
            "Per project rules, predict.sh must be inference-only.\n"
            "Fine-tune GPT-2 offline and save the model to work/FINAL, then run test.\n",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.mode == "test":
        model = MyModel.load(args.work_dir)
        test_data = load_test_data(args.test_data)
        pred = model.run_pred(test_data)
        write_pred(pred, args.test_output)
        return

    raise NotImplementedError(args.mode)


if __name__ == "__main__":
    main()