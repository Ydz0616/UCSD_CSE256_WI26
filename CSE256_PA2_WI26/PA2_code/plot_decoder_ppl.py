"""
Plot decoder perplexity curves from decoder_log.txt (Train + obama/wbush/hbush test PPL).
Run from PA2_code: python plot_decoder_ppl.py
Saves figures/decoder_perplexity.png (copy to figs/ for the report if needed).
"""
from utilities import plot_decoder_perplexity

# From logs/decoder_log.txt: eval every 100 steps up to 500
steps = [100, 200, 300, 400, 500]
train_ppl = [581.04, 450.83, 330.43, 240.20, 180.48]
obama_ppl = [704.14, 608.50, 496.66, 427.86, 392.34]
wbush_ppl = [798.97, 688.39, 585.11, 513.16, 496.65]
hbush_ppl = [726.64, 620.62, 526.21, 466.29, 436.06]

if __name__ == "__main__":
    plot_decoder_perplexity(steps, train_ppl, obama_ppl, wbush_ppl, hbush_ppl)
