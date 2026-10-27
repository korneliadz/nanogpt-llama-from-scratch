"""
Character-level training on Tiny Shakespeare, following the setup from
Andrej Karpathy's "Let's build GPT" (same data, same default hyperparameters).

    python train.py                     # full run
    python train.py --max-iters 500     # quick check
"""

import argparse
import os
import time
import urllib.request
from dataclasses import asdict

import torch

from model import ModelConfig, Transformer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_text(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading Tiny Shakespeare to {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--seed", type=int, default=1337)
    # training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    # model
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    # --- data: character-level tokenizer, 90/10 train/val split ---
    text = load_text(args.data)
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # --- model ---
    cfg = ModelConfig(
        vocab_size=len(chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = Transformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device: {device} | parameters: {model.num_params() / 1e6:.2f}M | vocab: {len(chars)}")

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # --- training loop ---
    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss()
            print(f"step {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| {time.time() - t0:.0f}s")

        if it == args.max_iters:
            break
        _, loss = model(*get_batch("train"))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- save + sample ---
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "chars": chars}, args.out)
    print(f"saved checkpoint to {args.out}")

    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)  # index 0 = "\n"
    out = model.generate(start, max_new_tokens=500)[0].tolist()
    print("\n--- sample ---\n" + "".join(chars[i] for i in out))


if __name__ == "__main__":
    main()
