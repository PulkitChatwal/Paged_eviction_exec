"""
paged_eviction
==============

Production-ready PagedEviction KV cache pruning for vLLM >= 0.9.0.

Reference
---------
Chitty-Venkata et al., EACL 2026 — https://aclanthology.org/2026.findings-eacl.168

Quick start
-----------
::

    from vllm import LLM, SamplingParams
    from paged_eviction import PagedEvictionConfig, apply_paged_eviction

    llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
    mgr = apply_paged_eviction(llm, PagedEvictionConfig(cache_budget=1024))
    outputs = llm.generate(prompts, SamplingParams(max_tokens=512))
    print(mgr.get_stats())
"""

from .config import PagedEvictionConfig
from .hooks import apply_paged_eviction
from .manager import PagedEvictionManager, compute_block_scores

__all__ = [
    "PagedEvictionConfig",
    "apply_paged_eviction",
    "PagedEvictionManager",
    "compute_block_scores",
]

__version__ = "0.1.0"
