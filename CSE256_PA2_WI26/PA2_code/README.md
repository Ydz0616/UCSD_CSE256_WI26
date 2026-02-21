# CSE256 PA2: Transformer Blocks

From-scratch PyTorch implementation (no `nn.TransformerEncoder` / decoder libs). Encoder and decoder follow a Pre-Norm design; implementation references Karpathy’s transformer tutorials and `main.py` hyperparameters. Part 3 (RoPE, GQA, SwiGLU) lives in `transformer_exp.py` and is run via `exp.py`.

## How to run

**From the `PA2_code` directory:**

- **Part 1 + Part 2 (default):**  
  `python main.py`  
  Runs encoder+classifier then decoder; stdout is logged to `logs/training_log.txt`.

- **Part 1 only (encoder + classifier):**  
  `python main.py -x encoder`

- **Part 2 only (decoder LM):**  
  `python main.py -x decoder`  
  Use `-p obama|wbush|hbush|all` to choose which test sets to evaluate (default: all).

- **Part 3 (architectural ablation):**  
  `python exp.py`  
  Runs all five configs (baseline, rope, gqa, swiglu, all_three) and prints the comparison table. To run a single config:  
  `python exp.py -e rope`  
  (same for `baseline`, `gqa`, `swiglu`, `all_three`).

- **Sanity checks (attention plots):**  
  `python test_sanity.py`  
  Optional: `-m <prefix>` to set the filename prefix for saved attention maps.

- **Decoder perplexity curve (from log data):**  
  `python plot_decoder_ppl.py`  
  Produces the train/test perplexity figure used in the report.

## Output locations

- **Logs:** `logs/`  
  - `training_log.txt` — main.py (encoder + decoder)  
  - `exp_log.txt` — save Part 3 by redirecting: `python exp.py > logs/exp_log.txt`

- **Figures:** `figures/`  
  - `classifier_accuracy.png` — encoder evaluation (from main.py)  
  - `decoder_perplexity.png` — decoder PPL curves (from plot_decoder_ppl.py)  
  - `attention_map_encoder_layer*_head*.png`, `attention_map_decoder_layer*_head*.png` — sanity check heatmaps (from test_sanity.py)



