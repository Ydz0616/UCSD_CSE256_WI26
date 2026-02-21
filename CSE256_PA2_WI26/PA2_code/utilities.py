
import matplotlib.pyplot as plt
import torch
import sys
class Logger(object):
    def __init__(self,filepath):
        self.terminal = sys.stdout
        self.log = open (filepath,'a')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

def plot_accuracy_curves(train_accs, test_accs, save_path="figures/classifier_accuracy.png"):
    """
    Plots training and testing accuracy curves and saves to a file.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(train_accs) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    
    plt.title('Classifier Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Accuracy plot saved to {save_path}")
    plt.close()


def plot_decoder_perplexity(steps, train_ppl, obama_ppl, wbush_ppl, hbush_ppl, save_path="figures/decoder_perplexity.png"):
    """
    Plots decoder train and per-president test perplexity over training steps.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_ppl, "b-", label="Train", linewidth=2)
    plt.plot(steps, obama_ppl, "C0--", label="Test (obama)", linewidth=1.5)
    plt.plot(steps, wbush_ppl, "C1--", label="Test (wbush)", linewidth=1.5)
    plt.plot(steps, hbush_ppl, "C2--", label="Test (hbush)", linewidth=1.5)

    plt.title("Decoder perplexity over training")
    plt.xlabel("Training step")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Decoder perplexity plot saved to {save_path}")
    plt.close()


class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model


    def sanity_check(self, sentence, block_size, **kwargs):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)

        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0)

        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)

        # Process the input tensor through the encoder model
        _,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps

        # Display the number of attention maps
        print("Number of attention maps:", len(attn_maps))

        # Create figures directory if it doesn't exist
        import os
        os.makedirs("figures", exist_ok=True)
        filename_prefix = kwargs.get('filename_prefix', 'attention_map')

        # Visualize and save the attention maps
        for j, attn_map in enumerate(attn_maps):
            # attn_map shape: (1, n_head, T, T) or (1, T, T)
            att_map = attn_map.squeeze(0).detach().cpu()  # Remove batch dimension
            
            # If it was averaged, make it artificially 1 head so loop works
            if att_map.dim() == 2:
                att_map = att_map.unsqueeze(0)
                
            n_heads = att_map.shape[0]
            
            for h in range(n_heads):
                head_map = att_map[h] # Shape: (T, T)
                # Check if the attention probabilities sum to 1 over columns (dim=1)
                total_prob_over_rows = torch.sum(head_map, dim=1)
                if torch.any(total_prob_over_rows < 0.99) or torch.any(total_prob_over_rows > 1.01):
                    print(f"Failed normalization test in layer {j+1} head {h+1}: probabilities do not sum to 1.0 over rows")
                    print("Total probability over rows:", total_prob_over_rows.numpy())

                # Create a heatmap of the attention map
                fig, ax = plt.subplots()
                cax = ax.imshow(head_map.numpy(), cmap='hot', interpolation='nearest')
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.set_xlabel('Key Token Position')
                ax.set_ylabel('Query Token Position')
                fig.colorbar(cax, ax=ax)  
                plt.title(f"Attention Map Layer {j + 1} Head {h + 1}")
                
                # Save the plot
                filepath = os.path.join("figures", f"{filename_prefix}_layer{j + 1}_head{h + 1}.png")
                plt.savefig(filepath)
                print(f"Saved attention map to {filepath}")
                
                # Close the plot
                plt.close(fig)
            


