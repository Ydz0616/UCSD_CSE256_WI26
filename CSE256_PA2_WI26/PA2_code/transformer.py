import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    It projects the input into Query, Key, and Value, splits into multiple heads,
    computes attention scores, and aggregates the values.
    """
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        
        self.n_head = n_head
        self.n_embd = n_embd
        # The dimension of each individual head
        self.head_size = n_embd // n_head
        
        # Separate layers for Q, K, V 
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        
        # Final output projection
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Channel(n_embd)
        
        # --- 1. Compute Q, K, V ---
        # q, k, v shape: (B, T, C)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # --- 2. Split into Heads ---
        # TODO: Use .view() to split the C dimension into (self.n_head, self.head_size)
        # Target shape for q, k, v: (B, T, n_head, head_size)
        
        q = q.view(B,T,self.n_head, self.head_size)
        k = k.view(B,T,self.n_head,self.head_size)
        v = v.view(B,T,self.n_head,self.head_size)



        # --- 3. Transpose for Matrix Multiplication ---
        # TODO: Use .transpose() to swap the Time (dim 1) and Head (dim 2) dimensions.
        # Target shape for q, k, v: (B, n_head, T, head_size)
        q = q.transpose(-3,-2)
        k = k.transpose(-3,-2)
        v = v.transpose(-3,-2)


        # --- 4. Compute Attention Scores ---
        # TODO: Compute Q * K^T. 
        # Remember: K needs its last two dimensions transposed (from (T, head_size) to (head_size, T))
        # Use matrix multiplication operator '@'
        # Crucial: Divide the result by math.sqrt(self.head_size) to stabilize gradients
        # Target shape for att: (B, n_head, T, T)
        wei  = q @ k.transpose(-2,-1) /math.sqrt(self.head_size)

        # --- 5. Softmax and Dropout ---
        if wei is not None:
             # Apply softmax along the last dimension (the key sequence length dimension)
            wei = F.softmax(wei, dim=-1)
            # Return the full (B, n_head, T, T) map so we can visualize each head
            att_weights = wei
            wei = self.attn_dropout(wei)

            # --- 6. Apply Attention to Values ---
            # Multiply the attention weights (wei) with the values (v)
            # Use matrix multiplication operator '@'
            # Target shape for y: (B, n_head, T, head_size)
            # (B, n_head, T, T) @ (B, n_head, T, head_size)
            y  = wei @ v
        else:
            y = None
            att_weights = None


        if y is not None:
            # --- 7. Re-assemble the Heads ---
            # Transpose y back to (B, T, n_head, head_size). 
            # Then use .contiguous().view(...) to flatten back to (B, T, C)
            y = y.transpose(-3,-2)
            y = y.contiguous().view(B,T,C)

            # --- 8. Final Output Projection ---
            # Apply the final nn.Linear projection (self.proj) and dropout (self.resid_dropout)
            y = self.resid_dropout(self.proj(y))
            
        return y, att_weights


class FeedForward(nn.Module):
    """ 
    A simple position-wise feed-forward network consisting of two linear layers 
    with a GELU activation in between.
    """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # TODO: Build an nn.Sequential network.
        # It should contain:
        # 1. nn.Linear from n_embd to 4 * n_embd
        # 2. nn.GELU() activation
        # 3. nn.Linear from 4 * n_embd back to n_embd
        # 4. nn.Dropout(dropout)
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block: Groups communication (Self-Attention) and computation (FeedForward).
    Implements the "Pre-Norm" architecture typical in modern transformers (like GPT-2).
    """
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        # TODO: Initialize components:
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd,n_head,dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd,dropout)



    def forward(self, x):
        att_map = None
        # TODO: Implement the Pre-Norm forward pass with Residual connections:
        # 1. Apply ln_1 to x, then pass it to attn. (Remember attn returns output AND att_map)
        # 2. Add the attention output to the original x (residual connection 1)
        # 3. Apply ln_2 to the new x, pass it to mlp.
        # 4. Add the mlp output to the current x (residual connection 2)

        
        att_out, att_map = self.attn(self.ln_1(x))
        # residual 1
        x = x + att_out
        # residual 2 
        x = x + self.mlp(self.ln_2(x))

        return x, att_map

    
    

class TransformerEncoder(nn.Module):
    """
    The full Transformer Encoder model.
    Takes token indices, converts them to embeddings (token + position), 
    and passes them through a stack of TransformerBlocks.
    """
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.1):
        super().__init__()
        
        # --- 1. Embeddings ---
        # TODO: Initialize the Token Embedding table (vocab_size -> n_embd)
        # TODO: Initialize the Position Embedding table (block_size -> n_embd)
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.token_position_table = nn.Embedding(block_size,n_embd)
        # --- 2. Transformer Blocks ---
        # TODO: Initialize an nn.ModuleList containing n_layer instances of TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(n_embd,n_head,dropout) for _ in range(n_layer)
        ])
        # --- 3. Final LayerNorm ---
        # TODO: Initialize a final nn.LayerNorm(n_embd)
        self.ln = nn.LayerNorm(n_embd)


    def forward(self, idx):
        B, T = idx.shape # Batch size, Sequence length (Time)

        # --- 1. Compute Embeddings ---
        # TODO: Get token embeddings using idx
        # TODO: Generate a position tensor (from 0 to T-1) on the same device as idx
        # TODO: Get position embeddings using the position tensor
        # TODO: Add token and position embeddings together to form x

        #(B, T, C)
        token_embeddings = self.token_embedding_table(idx)
        # position
        pos_tensor = torch.arange(0,T, dtype = torch.long, device = idx.device)
        # position embeddings, (T,C)
        position_embeddings = self.token_position_table(pos_tensor)

        x = token_embeddings + position_embeddings 
        
        attn_maps = []
        
        if x is not None:
            # --- 2. Pass through Blocks ---
            # TODO: Iterate through self.blocks
            # For each block: pass x through, update x, and append the returned att_map to attn_maps
            for block in self.layers:
                x,att_map = block(x)
                attn_maps.append(att_map)
            
            # --- 3. Final LayerNorm ---
            # TODO: Apply the final LayerNorm to x
        x = self.ln(x)
        
        return x, attn_maps


class Classifier(nn.Module):

    def __init__(self,n_input,n_hidden,n_output):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_input,n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden,n_output)
        )

    def forward(self,x):
        # x (B,T,n_input) --> (B,n_input)
        x = x.mean(dim=1)
        logits = self.layers(x)
        return logits


# ============================================================
# PART 2: TRANSFORMER DECODER
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-Head CAUSAL Self-Attention.
    Identical to MultiHeadAttention but applies a causal mask
    so that each position can only attend to earlier positions
    (preventing the model from seeing future tokens).
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key   = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj  = nn.Linear(n_embd, n_embd)

        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # TODO: --- IMPLEMENT THE CAUSAL MASK ---
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        q = self.query(x)  # (B, T, C)
        k = self.key(x)
        v = self.value(x)

        # Split into heads -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Compute raw attention scores: (B, n_head, T, T)
        wei = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)

        # --- TODO: APPLY THE CAUSAL MASK HERE ---
        wei = wei.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        

        wei = F.softmax(wei, dim=-1)
        att_weights = wei  # Save pure probabilities before dropout
        wei = self.attn_dropout(wei)

        y = wei @ v  # (B, n_head, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))

        return y, att_weights


class DecoderBlock(nn.Module):
    """
    Transformer Decoder Block.
    Same Pre-Norm residual structure as EncoderBlock but:
      - Uses CausalSelfAttention (with causal mask)
      - FeedForward uses ReLU (not GELU) per the assignment spec
      - Hidden dim of FFN is 100 (not 4*n_embd)
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)

        # hard-coded to be 100, as per assignment
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 100),
            nn.ReLU(),
            nn.Linear(100, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # --- TODO: Implement the pre-norm residual forward pass ---
        att_out, att_map = self.attn(self.ln_1(x))
        x = x + att_out
        x = x + self.mlp(self.ln_2(x))
        return x, att_map


class TransformerDecoder(nn.Module):
    """
    GPT-style Transformer Decoder for autoregressive language modeling.
    Given a sequence of token indices, it predicts the NEXT token at every position.
    """
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        # Token and position embeddings (same as Encoder)
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of Decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        # Final LayerNorm
        self.ln = nn.LayerNorm(n_embd)

        # Language Model head: projects from n_embd -> vocab_size to get next-token logits
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # --- TODO: Compute Embeddings ---
        # Same as TransformerEncoder: token_emb + position_emb -> x
        token_emb = self.token_embedding_table(idx)
        pos_tensor = torch.arange(0,T, dtype = torch.long, device = idx.device)
        position_emb = self.position_embedding_table(pos_tensor)

        x = token_emb + position_emb


        # --- TODO: Pass through Decoder blocks ---
        attn_maps = []
        
        if x is not None:
            for block in self.layers:
                x,att_map = block(x)
                attn_maps.append(att_map)
            

        # --- TODO: Apply final LayerNorm ---
        x = self.ln(x)
        # Project to vocabulary logits
        logits = self.lm_head(x)


        # Compute cross-entropy loss if targets are provided
        # (This is what main.py's compute_perplexity expects)
        if targets is not None:
            # logits must be reshaped to (B*T, vocab_size); targets to (B*T,)
            # Flatten (B, T, vocab_size) -> (B*T, vocab_size) and (B, T) -> (B*T,)
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
            return loss
        
        return logits, attn_maps
