import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from sentiment_data import read_word_embeddings
from DANmodels import SentimentDatasetDAN, DAN

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
    
    # Ensure figs directory exists
    if not os.path.exists('figs'):
        os.makedirs('figs')
        
    save_path = os.path.join('figs', filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def run_all_experiments():
    
    # 1. Load Embeddings (Standard 50d and 300d)
    embeddings_50 = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    embeddings_300 = read_word_embeddings("data/glove.6B.300d-relativized.txt")
    
    # 2. Prepare Datasets (We need different datasets for different embeddings)
    # For 50d
    train_data_50 = SentimentDatasetDAN("data/train.txt", embeddings_50, sentence_len=512)
    dev_data_50 = SentimentDatasetDAN("data/dev.txt", embeddings_50, sentence_len=512)
    # For 300d
    train_data_300 = SentimentDatasetDAN("data/train.txt", embeddings_300, sentence_len=512)
    dev_data_300 = SentimentDatasetDAN("data/dev.txt", embeddings_300, sentence_len=512)
    
    # Create Loaders
    loaders_50 = {
        'train': DataLoader(train_data_50, batch_size=16, shuffle=True),
        'test': DataLoader(dev_data_50, batch_size=16, shuffle=False)
    }
    loaders_300 = {
        'train': DataLoader(train_data_300, batch_size=16, shuffle=True),
        'test': DataLoader(dev_data_300, batch_size=16, shuffle=False)
    }

    # Baseline Config
    # Model: DAN, n_hidden=100, n_class=2, n_layers=2, dropout=0.25
    
    # ==========================
    # Experiment 1: Embeddings
    # ==========================
    print("\n>>> Running Experiment 1: Embedding Variations")
    exp1_results = {}
    
    # 1.1 Pretrained 50d (Baseline)
    print("Training Pretrained 50d...")
    model = DAN(embeddings_50, n_class=2, n_hidden=100, n_layers=2, from_pretrained=True, dropout=0.25)
    _, test_acc = experiment(model, loaders_50['train'], loaders_50['test'])
    exp1_results['Pretrained 50d'] = test_acc
    
    # 1.2 Pretrained 300d
    print("Training Pretrained 300d...")
    model = DAN(embeddings_300, n_class=2, n_hidden=100, n_layers=2, from_pretrained=True, dropout=0.25)
    _, test_acc = experiment(model, loaders_300['train'], loaders_300['test'])
    exp1_results['Pretrained 300d'] = test_acc

    # 1.3 Scratch 50d
    print("Training Scratch 50d...")
    # Note: passing embeddings_50 just for vocab size
    model = DAN(embeddings_50, n_class=2, n_hidden=100, n_layers=2, from_pretrained=False, embed_dim=50, dropout=0.25)
    _, test_acc = experiment(model, loaders_50['train'], loaders_50['test'])
    exp1_results['Scratch 50d'] = test_acc

    # 1.4 Scratch 300d
    print("Training Scratch 300d...")
    model = DAN(embeddings_50, n_class=2, n_hidden=100, n_layers=2, from_pretrained=False, embed_dim=300, dropout=0.25)
    _, test_acc = experiment(model, loaders_50['train'], loaders_50['test'])
    exp1_results['Scratch 300d'] = test_acc
    
    plot_results(exp1_results, "Experiment 1: Embedding Method & Dimension", "exp1_embeddings.png")


    # ==========================
    # Experiment 2: Layers
    # Fixed: Pretrained 50d, Hidden 100, Dropout 0.25
    # ==========================
    print("\n>>> Running Experiment 2: Layer Depth")
    exp2_results = {}
    
    for n_layers in [2, 3, 4]:
        print(f"Training {n_layers} Layers...")
        model = DAN(embeddings_50, n_class=2, n_hidden=100, n_layers=n_layers, from_pretrained=True, dropout=0.25)
        _, test_acc = experiment(model, loaders_50['train'], loaders_50['test'])
        exp2_results[f'{n_layers} Layers'] = test_acc
        
    plot_results(exp2_results, "Experiment 2: Layer Depth Impact", "exp2_layers.png")


    # ==========================
    # Experiment 3: Hidden Dimensions
    # Fixed: Pretrained 50d, Layers 2, Dropout 0.25
    # ==========================
    print("\n>>> Running Experiment 3: Hidden Dimensions")
    exp3_results = {}
    
    for hidden in [100, 200, 300]:
        print(f"Training Hidden Dim {hidden}...")
        model = DAN(embeddings_50, n_class=2, n_hidden=hidden, n_layers=2, from_pretrained=True, dropout=0.25)
        _, test_acc = experiment(model, loaders_50['train'], loaders_50['test'])
        exp3_results[f'Hidden {hidden}'] = test_acc
        
    plot_results(exp3_results, "Experiment 3: Hidden Layer Dimension", "exp3_hidden_dim.png")


    # ==========================
    # Experiment 4: Dropout
    # Fixed: Pretrained 50d, Layers 2, Hidden 100
    # ==========================
    print("\n>>> Running Experiment 4: Dropout Rates")
    exp4_results = {}
    
    for p in [0.0, 0.25, 0.5]:
        print(f"Training Dropout {p}...")
        model = DAN(embeddings_50, n_class=2, n_hidden=100, n_layers=2, from_pretrained=True, dropout=p)
        _, test_acc = experiment(model, loaders_50['train'], loaders_50['test'])
        exp4_results[f'Dropout {p}'] = test_acc
        
    # Also add 0.75 as per request
    print("Training Dropout 0.75...")
    model = DAN(embeddings_50, n_class=2, n_hidden=100, n_layers=2, from_pretrained=True, dropout=0.75)
    _, test_acc = experiment(model, loaders_50['train'], loaders_50['test'])
    exp4_results['Dropout 0.75'] = test_acc

    plot_results(exp4_results, "Experiment 4: Dropout Regularization", "exp4_dropout.png")

    print("\nAll experiments completed! Figures saved in 'figs/' directory.")

if __name__ == "__main__":
    run_all_experiments()
