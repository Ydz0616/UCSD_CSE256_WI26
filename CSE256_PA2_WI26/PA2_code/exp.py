"""
exp.py - Ablation experiment runner for Part 3 architecture experiments.

Usage:
  python exp.py -e baseline      # Standard decoder (same as Part 2)
  python exp.py -e rope          # + RoPE only
  python exp.py -e gqa           # + GQA only
  python exp.py -e swiglu        # + SwiGLU only
  python exp.py -e all_three     # RoPE + GQA + SwiGLU combined
  python exp.py -e ablation      # Run ALL 5 configs and compare
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os, sys, argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer_exp import ExpTransformerDecoder

# ---- Hyperparameters (same as main.py) ----
batch_size = 16
block_size = 32
learning_rate = 1e-3
n_embd = 64
n_head = 4          # Upgraded from 2 to 4 for GQA experiments
n_layer = 4
dropout = 0.1
max_iters = 500
eval_interval = 100
eval_iters = 200
n_kv_head_gqa = 2   # For GQA: 4 Q heads share 2 KV heads

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Experiment Configs ----
CONFIGS = {
    "baseline":   {"use_rope": False, "n_kv_head": None,         "use_swiglu": False},
    "rope":       {"use_rope": True,  "n_kv_head": None,         "use_swiglu": False},
    "gqa":        {"use_rope": False, "n_kv_head": n_kv_head_gqa,"use_swiglu": False},
    "swiglu":     {"use_rope": False, "n_kv_head": None,         "use_swiglu": True},
    "all_three":  {"use_rope": True,  "n_kv_head": n_kv_head_gqa,"use_swiglu": True},
}


def load_texts(directory):
    texts = []
    for filename in os.listdir(directory):
        if "test" in filename:
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
            texts.append(f.read())
    return texts


def compute_perplexity(model, data_loader, eval_iters=200):
    model.eval()
    losses = []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            loss = model(X, Y)
            losses.append(loss.item())
            if len(losses) >= eval_iters:
                break
    losses = torch.tensor(losses)
    perplexity = torch.exp(losses.mean()).item()
    model.train()
    return perplexity


def run_experiment(name, config, tokenizer, vocab_size, train_loader, test_loaders):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name.upper()}")
    print(f"  Config: RoPE={config['use_rope']}, GQA n_kv_head={config['n_kv_head']}, SwiGLU={config['use_swiglu']}")
    print(f"{'='*60}")

    model = ExpTransformerDecoder(
        vocab_size, block_size, n_embd, n_head, n_layer, dropout,
        use_rope=config["use_rope"],
        n_kv_head=config["n_kv_head"],
        use_swiglu=config["use_swiglu"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i, (xb, yb) in enumerate(train_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = model(xb, targets=yb)
        loss.backward()
        optimizer.step()

        if (i + 1) % eval_interval == 0:
            train_ppl = compute_perplexity(model, train_loader, eval_iters)
            log = f"  Step {i+1}/{max_iters} | Loss: {loss.item():.4f} | Train PPL: {train_ppl:.2f}"
            for pname, loader in test_loaders.items():
                ppl = compute_perplexity(model, loader, eval_iters)
                log += f" | {pname}: {ppl:.2f}"
            print(log)

    # Final results
    results = {"name": name, "params": num_params}
    results["train_ppl"] = compute_perplexity(model, train_loader, eval_iters)
    for pname, loader in test_loaders.items():
        results[pname] = compute_perplexity(model, loader, eval_iters)
    return results


def print_comparison_table(all_results):
    print(f"\n{'='*80}")
    print(f"  ABLATION COMPARISON TABLE")
    print(f"{'='*80}")

    # Header
    test_names = [k for k in all_results[0].keys() if k not in ("name", "params", "train_ppl")]
    header = f"{'Config':<15} {'Params':>10} {'Train PPL':>12}"
    for tn in test_names:
        header += f" {tn+' PPL':>12}"
    print(header)
    print("-" * len(header))

    # Rows
    for r in all_results:
        row = f"{r['name']:<15} {r['params']:>10,} {r['train_ppl']:>12.2f}"
        for tn in test_names:
            row += f" {r[tn]:>12.2f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Part 3: Architecture Experiments")
    parser.add_argument("-e", "--experiment", type=str,
                        choices=list(CONFIGS.keys()) + ["ablation"],
                        default="ablation", help="Which experiment to run")
    args = parser.parse_args()

    # Setup
    os.makedirs("logs", exist_ok=True)
    print("Loading data and creating tokenizer ...")
    texts = load_texts("speechesdataset")
    tokenizer = SimpleTokenizer(' '.join(texts))
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Train data
    with open("speechesdataset/train_LM.txt", 'r', encoding='utf-8') as f:
        train_text = f.read()
    train_dataset = LanguageModelingDataset(tokenizer, train_text, block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test data (all 3 politicians)
    politician_files = {
        "obama": "speechesdataset/test_LM_obama.txt",
        "wbush": "speechesdataset/test_LM_wbush.txt",
        "hbush": "speechesdataset/test_LM_hbush.txt",
    }
    test_loaders = {}
    for name, path in politician_files.items():
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        ds = LanguageModelingDataset(tokenizer, text, block_size)
        test_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Determine which experiments to run
    if args.experiment == "ablation":
        experiments = list(CONFIGS.keys())
    else:
        experiments = [args.experiment]

    all_results = []
    for exp_name in experiments:
        config = CONFIGS[exp_name]
        result = run_experiment(exp_name, config, tokenizer, vocab_size, train_loader, test_loaders)
        all_results.append(result)

    # Print comparison table if running multiple experiments
    if len(all_results) > 1:
        print_comparison_table(all_results)


if __name__ == "__main__":
    main()
