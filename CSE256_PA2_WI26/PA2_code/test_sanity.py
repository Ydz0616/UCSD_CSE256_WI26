import torch
import sys
import argparse
sys.path.append("/Users/yuandong/Downloads/CSE256_PA2_WI26/PA2_code")

from transformer import TransformerEncoder
from tokenizer import SimpleTokenizer
from utilities import Utilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sanity check on Transformer Encoder.")
    parser.add_argument("-m", "--model_name", type=str, default="attention_map", help="Prefix for the saved attention map images")
    args = parser.parse_args()

    # Create dummy tokenizer
    text = "The quick brown fox jumps over the lazy dog"
    tokenizer = SimpleTokenizer(text)
    vocab_size = tokenizer.vocab_size

    # Hyperparameters from main.py
    
    block_size = 32
    n_embd = 64
    n_head = 2
    n_layer = 4

    print("=== Encoder Sanity Check ===")
    print("Initializing TransformerEncoder...")
    encoder = TransformerEncoder(vocab_size, block_size, n_embd, n_head, n_layer)
    utils = Utilities(tokenizer, encoder)
    try:
        utils.sanity_check("The quick brown fox", block_size, filename_prefix=args.model_name + "_encoder")
        print("SUCCESS: Encoder sanity_check passed!\n")
    except Exception as e:
        import traceback
        traceback.print_exc()

    print("=== Decoder Sanity Check ===")
    print("Initializing TransformerDecoder...")
    from transformer import TransformerDecoder
    decoder = TransformerDecoder(vocab_size, block_size, n_embd, n_head, n_layer)
    utils_dec = Utilities(tokenizer, decoder)
    try:
        utils_dec.sanity_check("The quick brown fox", block_size, filename_prefix=args.model_name + "_decoder")
        print("SUCCESS: Decoder sanity_check passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
