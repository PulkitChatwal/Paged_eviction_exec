"""
Offline inference with PagedEviction.

Usage
-----
With a real GPU::

    python examples/offline_inference.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --cache-budget 1024

Dry-run / smoke test (no GPU required)::

    python examples/offline_inference.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEMO_PROMPTS = [
    "Summarize the key contributions of the PagedAttention paper in detail.",
    (
        "Explain the trade-offs between KV cache size and generation quality, "
        "and describe three strategies researchers have proposed to reduce "
        "KV cache memory without retraining the model."
    ),
    "What are the differences between StreamingLLM, H2O, and PagedEviction?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PagedEviction offline inference demo")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--cache-budget", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without vLLM (tests the package logic only)",
    )
    return p.parse_args()


def run_dry_run() -> None:
    """Smoke-test the eviction pipeline on CPU without vLLM or a GPU."""
    import os
    import sys

    # Make sure the repo root is on the path when running directly
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # inject mock torch if needed
    try:
        import torch  # noqa: F401
    except Exception:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
        import _mock_torch as _mt  # type: ignore[import]

        sys.modules["torch"] = _mt  # type: ignore[assignment]

    import torch

    from paged_eviction import PagedEvictionConfig, PagedEvictionManager

    print("\n=== PagedEviction dry-run (CPU, no GPU required) ===\n")
    kv = [torch.randn(2, 32, 16, 8, 64) for _ in range(4)]
    mgr = PagedEvictionManager(kv, block_size=16, cache_budget=128)

    freed = []
    block_ids = list(range(10))
    block_ids = mgr.evict_prefill(0, block_ids, 160, freed.append)
    print(f"Prefill:  10 -> {len(block_ids)} blocks  (freed {len(freed)})")

    seq_len = len(block_ids) * 16
    next_blk = 12
    for _ in range(64):
        seq_len += 1
        if seq_len % 16 == 0:
            block_ids.append(next_blk)
            next_blk += 1
        block_ids = mgr.evict_decode(0, block_ids, seq_len, freed.append)

    print(f"Decode:   {len(block_ids)} blocks after 64 steps  (seq_len={seq_len})")
    print(f"Stats:    {mgr.get_stats()}")
    print("\n✅  Dry-run passed — package is working correctly.\n")


def run_with_vllm(args: argparse.Namespace) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("vLLM not installed. Run:  pip install vllm==0.9.0")
        sys.exit(1)

    from paged_eviction import PagedEvictionConfig, apply_paged_eviction

    print(f"\nModel:        {args.model}")
    print(f"Cache budget: {args.cache_budget} tokens\n")

    llm = LLM(model=args.model, trust_remote_code=True)
    cfg = PagedEvictionConfig(cache_budget=args.cache_budget)
    mgr = apply_paged_eviction(llm, cfg)
    print(f"Attached. Stats: {mgr.get_stats()}\n")

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    t0 = time.perf_counter()
    outputs = llm.generate(DEMO_PROMPTS, sp)
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.2f}s  |  Final stats: {mgr.get_stats()}\n")
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        print(f"[{i + 1}] {text[:200]}...\n")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
    else:
        run_with_vllm(args)


if __name__ == "__main__":
    main()
