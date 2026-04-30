# PagedEviction for vLLM

Block-wise KV cache pruning for vLLM — implementation of the EACL 2026 paper by Chitty-Venkata et al.

> **Paper:** [PagedEviction: Structured Block-wise KV Cache Pruning](https://aclanthology.org/2026.findings-eacl.168)

---

## Requirements

| Library | Version |
|---|---|
| Python | 3.10 – 3.12 |
| torch | 2.1+ |
| vllm | 0.9.0 |
| transformers | **4.51.3** |
| accelerate | 0.26+ |

> **GPU:** NVIDIA GPU required for full inference. CPU dry-run works without a GPU.  
> **Tested on:** Tesla T4 (15GB), vLLM V0 engine, XFormers backend.

---

## Install

```bash
# 1. Clone
git clone https://github.com/PulkitChatwal/Paged_eviction_exec.git
cd Paged_eviction_exec

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 3. Install exact dependencies
pip install torch==2.1.0
pip install transformers==4.51.3
pip install vllm==0.9.0
pip install -e .
```

---

## Verify install (no GPU needed)

```bash
python examples/offline_inference.py --dry-run
```

Expected output:
```
=== PagedEviction dry-run (CPU, no GPU required) ===

Prefill:  10 -> 8 blocks  (freed 2)
Decode:   8 blocks after 64 steps  (seq_len=192)
Stats:    {'active_sequences': 1, 'total_decode_evictions': 4, ...}

✅  Dry-run passed — package is working correctly.
```

---

## Run on any prompt

```python
import sys
sys.path.insert(0, "/path/to/Paged_eviction_exec")

from vllm import LLM, SamplingParams
from paged_eviction import PagedEvictionConfig, apply_paged_eviction

# 1. Load model
llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

# 2. Attach PagedEviction
mgr = apply_paged_eviction(llm, PagedEvictionConfig(cache_budget=1024))

# 3. Your prompt
prompt = "Explain the difference between KV cache eviction and preemption in vLLM."

# 4. Generate
outputs = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=512))
print(outputs[0].outputs[0].text)
print("\nStats:", mgr.get_stats())
```

---

## Config options

```python
PagedEvictionConfig(
    cache_budget=1024,       # max KV tokens to keep per sequence
    protect_recent_blocks=1, # never evict the N most-recent blocks
)
```

| `cache_budget` | Effect |
|---|---|
| 2048 | Near full-cache quality, some memory saving |
| 1024 | Best accuracy/memory tradeoff (paper recommendation) |
| 512  | Aggressive eviction, visible quality drop on hard tasks |

---

## Google Colab (T4)

```python
# Cell 1 — Install
!pip install -q transformers==4.51.3 vllm==0.9.0
!git clone https://github.com/PulkitChatwal/Paged_eviction_exec.git
!pip install -q -e /content/Paged_eviction_exec

# Cell 2 — Run
import sys
sys.path.insert(0, "/content/Paged_eviction_exec")

from vllm import LLM, SamplingParams
from paged_eviction import PagedEvictionConfig, apply_paged_eviction

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
mgr = apply_paged_eviction(llm, PagedEvictionConfig(cache_budget=1024))

outputs = llm.generate(
    ["Explain PagedEviction in simple terms."],
    SamplingParams(temperature=0.0, max_tokens=300)
)
print(outputs[0].outputs[0].text)
print("\nStats:", mgr.get_stats())
```

---

## What the stats mean

```python
{
  'active_sequences': 1,        # sequences currently tracked
  'total_decode_evictions': 4,  # blocks evicted during generation
  'cache_budget': 1024,         # your configured token budget
  'block_size': 16              # tokens per vLLM page
}
```

`total_decode_evictions > 0` means the prompt exceeded your budget and eviction fired. Each eviction frees 16 tokens worth of KV cache memory.

---

## Supported models

Any model vLLM supports — architecture does not matter. Tested with:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen3-1.7B`
- `meta-llama/Llama-3.2-1B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

---

## Citation

```bibtex
@inproceedings{chittyvenkata2026pagedeviction,
  title     = {PagedEviction: Structured Block-wise {KV} Cache Pruning for
               Efficient Large Language Model Inference},
  author    = {Chitty-Venkata, Krishna Teja and others},
  booktitle = {Findings of EACL 2026},
  year      = {2026},
  url       = {https://aclanthology.org/2026.findings-eacl.168},
}
```
