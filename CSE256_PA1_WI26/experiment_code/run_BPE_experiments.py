import sys
import os
# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sentiment_data import read_word_embeddings, read_sentiment_examples
from DANmodels import SentimentDatasetDAN, BPEDatasetDAN, DAN
from bpe_tokenizer import BPETokenizer

from main import experiment

def plot_results(results_dict, title, filename):
    """
    Helper function to plot comparison results.
    results_dict: { 'Label': [accuracy_list] }
    """
    plt.figure(figsize=(10, 6))
    for label, acc_list in results_dict.items():
        plt.plot(acc_list, label=label)
    
    plt.xlabel('Epochs')
    plt.ylabel('Dev Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Ensure experiment_fig directory exists
    fig_dir = '../experiment_fig'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    save_path = os.path.join(fig_dir, filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def train_bpe_and_create_loaders(vocab_size, sentence_len=512):
    """Train BPE tokenizer and create data loaders (with caching)"""
    bpe_file = f"../bpe_models/bpe_vocab{vocab_size}.pkl"
    
    bpe = BPETokenizer()
    
    if os.path.exists(bpe_file):
        print(f"  Loading cached BPE tokenizer (vocab_size={vocab_size})...")
        bpe.load(bpe_file)
    else:
        print(f"  Training BPE tokenizer (vocab_size={vocab_size})...")
        train_examples = read_sentiment_examples("../data/train.txt")
        train_texts = [" ".join(ex.words) for ex in train_examples]
        bpe.train(train_texts, vocab_size)
        
        # Save for reuse
        os.makedirs("../bpe_models", exist_ok=True)
        bpe.save(bpe_file)
        print(f"  BPE tokenizer saved to {bpe_file}")
    
    # Create datasets
    train_data = BPEDatasetDAN("../data/train.txt", bpe, sentence_len)
    dev_data = BPEDatasetDAN("../data/dev.txt", bpe, sentence_len)
    
    loaders = {
        'train': DataLoader(train_data, batch_size=16, shuffle=True),
        'test': DataLoader(dev_data, batch_size=16, shuffle=False)
    }
    
    actual_vocab_size = 256 + len(bpe.merges)
    return loaders, actual_vocab_size

def run_all_experiments():
    
    print("=" * 70)
    print("BPE TOKENIZER + DAN MODEL EXPERIMENTS")
    print("=" * 70)
    
    # ==========================
    # Experiment 1: Vocab Size Variations
    # ==========================
    print("\n>>> Running Experiment 1: Vocab Size Comparison")
    exp1_results = {}
    
    vocab_sizes = [500, 1000, 1500, 2000]
    
    for vocab_size in vocab_sizes:
        print(f"\n[Vocab Size = {vocab_size}]")
        loaders, actual_vocab_size = train_bpe_and_create_loaders(vocab_size)
        
        model = DAN(
            embeddings=None,
            n_class=2,
            n_hidden=100,
            n_layers=2,
            from_pretrained=False,
            embed_dim=100,
            dropout=0.25,
            vocab_size=actual_vocab_size
        )
        
        _, test_acc = experiment(model, loaders['train'], loaders['test'])
        exp1_results[f'Vocab {vocab_size}'] = test_acc
    
    plot_results(exp1_results, "Experiment 1: BPE Vocab Size Impact", "bpe_exp1_vocab_size.png")
    
    
    # ==========================
    # Experiment 2: BPE vs Pretrained Embeddings
    # Compare BPE (2000 vocab) with DAN 300d pretrained
    # ==========================
    print("\n>>> Running Experiment 2: BPE vs Pretrained GloVe")
    exp2_results = {}
    
    # 2.1 BPE with 2000 vocab
    print("\n[BPE with 2000 vocab, 300d embeddings]")
    loaders_bpe, actual_vocab_size = train_bpe_and_create_loaders(2000)
    model_bpe = DAN(
        embeddings=None,
        n_class=2,
        n_hidden=100,
        n_layers=2,
        from_pretrained=False,
        embed_dim=300,  # Use 300d for fair comparison
        dropout=0.25,
        vocab_size=actual_vocab_size
    )
    _, test_acc = experiment(model_bpe, loaders_bpe['train'], loaders_bpe['test'])
    exp2_results['BPE (2000 vocab, 300d)'] = test_acc
    
    # 2.2 Pretrained GloVe 300d
    print("\n[GloVe Pretrained 300d]")
    embeddings_300 = read_word_embeddings("../data/glove.6B.300d-relativized.txt")
    train_data_300 = SentimentDatasetDAN("../data/train.txt", embeddings_300, sentence_len=512)
    dev_data_300 = SentimentDatasetDAN("../data/dev.txt", embeddings_300, sentence_len=512)
    
    loaders_glove = {
        'train': DataLoader(train_data_300, batch_size=16, shuffle=True),
        'test': DataLoader(dev_data_300, batch_size=16, shuffle=False)
    }
    
    model_glove = DAN(
        embeddings=embeddings_300,
        n_class=2,
        n_hidden=100,
        n_layers=2,
        from_pretrained=True,
        dropout=0.25
    )
    _, test_acc = experiment(model_glove, loaders_glove['train'], loaders_glove['test'])
    exp2_results['GloVe Pretrained 300d'] = test_acc
    
    plot_results(exp2_results, "Experiment 2: BPE vs Pretrained Embeddings", "bpe_exp2_vs_pretrained.png")
    
    
    # ==========================
    # Experiment 3: Layer Depth
    # Fixed: BPE (vocab_size=1000), embed_dim=100, hidden=100, dropout=0.25
    # ==========================
    print("\n>>> Running Experiment 3: Layer Depth")
    exp3_results = {}
    
    # Prepare BPE loaders once for this experiment
    loaders, actual_vocab_size = train_bpe_and_create_loaders(1000)
    
    for n_layers in [2, 3, 4]:
        print(f"\n[Training {n_layers} Layers with BPE]")
        model = DAN(
            embeddings=None,
            n_class=2,
            n_hidden=100,
            n_layers=n_layers,
            from_pretrained=False,
            embed_dim=100,
            dropout=0.25,
            vocab_size=actual_vocab_size
        )
        _, test_acc = experiment(model, loaders['train'], loaders['test'])
        exp3_results[f'{n_layers} Layers'] = test_acc
        
    plot_results(exp3_results, "Experiment 3: BPE Layer Depth Impact", "bpe_exp3_layers.png")
    
    
    # ==========================
    # Experiment 4: Hidden Dimensions
    # Fixed: BPE (vocab_size=1000), embed_dim=100, layers=2, dropout=0.25
    # ==========================
    print("\n>>> Running Experiment 4: Hidden Dimensions")
    exp4_results = {}
    
    # Reuse loaders and vocab_size from Exp 3
    for hidden in [100, 200, 300]:
        print(f"\n[Training Hidden Dim {hidden} with BPE]")
        model = DAN(
            embeddings=None,
            n_class=2,
            n_hidden=hidden,
            n_layers=2,
            from_pretrained=False,
            embed_dim=100,
            dropout=0.25,
            vocab_size=actual_vocab_size
        )
        _, test_acc = experiment(model, loaders['train'], loaders['test'])
        exp4_results[f'Hidden {hidden}'] = test_acc
        
    plot_results(exp4_results, "Experiment 4: BPE Hidden Layer Dimension", "bpe_exp4_hidden_dim.png")
    
    
    # ==========================
    # Experiment 5: Dropout Rates
    # Fixed: BPE (vocab_size=1000), embed_dim=100, layers=2, hidden=100
    # ==========================
    print("\n>>> Running Experiment 5: Dropout Rates")
    exp5_results = {}
    
    # Reuse loaders and vocab_size from Exp 3
    for p in [0.0, 0.25, 0.5, 0.75]:
        print(f"\n[Training Dropout {p} with BPE]")
        model = DAN(
            embeddings=None,
            n_class=2,
            n_hidden=100,
            n_layers=2,
            from_pretrained=False,
            embed_dim=100,
            dropout=p,
            vocab_size=actual_vocab_size
        )
        _, test_acc = experiment(model, loaders['train'], loaders['test'])
        exp5_results[f'Dropout {p}'] = test_acc
        
    plot_results(exp5_results, "Experiment 5: BPE Dropout Regularization", "bpe_exp5_dropout.png")
    
    
    print("\n" + "=" * 70)
    print("ALL BPE EXPERIMENTS COMPLETED!")
    print("Figures saved in '../experiment_fig/' directory.")
    print("=" * 70)

if __name__ == "__main__":
    run_all_experiments()
