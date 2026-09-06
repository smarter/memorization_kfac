"""Build a large dolmino reference pool (dolmino_50B mix, seed 7) of 512-token sequences for the imprint experiments."""
import sys, torch
sys.path.insert(0, "/home/guillaume/memorization_kfac")
from data.calibration import collect_sequences
from transformers import AutoTokenizer
S = "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad"
tok = AutoTokenizer.from_pretrained("/home/guillaume/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B/snapshots/7df9a82518afdecae4e8c026b27adccc8c1f0032")
seqs = collect_sequences("dolmo", tok, 512, 80_000_000, seed=7, mix_strategy="dolmino_50B")
t = torch.tensor(seqs, dtype=torch.long); print("pool", t.shape, flush=True); torch.save(t, f"{S}/dolmino_pool_big.pt"); print("saved", flush=True)
