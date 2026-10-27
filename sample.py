"""
Generate text from a trained checkpoint.

    python sample.py --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8 --top-k 40
"""

import argparse

import torch

from model import ModelConfig, Transformer
from train import pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/model.pt")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    chars = ckpt["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}

    model = Transformer(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        raise SystemExit(f"Characters not in the vocabulary: {unknown}")

    idx = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(chars[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
