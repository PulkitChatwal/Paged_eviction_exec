"""
Throughput benchmark replicating Figure 3 from the PagedEviction paper.

Usage::

    python benchmarks/benchmark_throughput.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --cache-budgets 256 512 1024 2048 4096 \\
        --input-len 1024 \\
        --output-len 8192 \\
        --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Optional

BUDGETS_DEFAULT = [256, 512, 1024, 2048, 4096]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cache-budgets", nargs="+", type=int, default=BUDGETS_DEFAULT)
    p.add_argument("--input-len", type=int, default=1024)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--protect-recent-blocks", type=int, default=1)
    return p.parse_args()


def make_prompts(n: int, approx_tokens: int) -> List[str]:
    return [("hello " * (approx_tokens // 2)) for _ in range(n)]


def benchmark_one(llm: object, prompts: List[str], sp: object, label: str) -> Dict[str, object]:
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)  # type: ignore[attr-defined]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_in = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(sum(len(x.token_ids) for x in o.outputs) for o in outputs)
    total = total_in + total_out
    throughput = total / elapsed
    tpot = (elapsed / total_out * 1000) if total_out > 0 else 0.0

    print(f"  [{label}]  {throughput:.1f} tok/s  |  TPOT {tpot:.2f} ms")
    return {
        "label": label,
        "throughput_tok_s": round(throughput, 2),
        "tpot_ms": round(tpot, 3),
        "total_tokens": total,
        "elapsed_s": round(elapsed, 3),
    }


def main() -> None:
    args = parse_args()

    try:
        from vllm import LLM, SamplingParams

        from paged_eviction import PagedEvictionConfig, apply_paged_eviction
    except ImportError as exc:
        print(f"Import error: {exc}\nRun: pip install vllm==0.9.0 paged-eviction-vllm")
        return

    prompts = make_prompts(args.batch_size, args.input_len)
    sp = SamplingParams(temperature=0.0, max_tokens=args.output_len)
    results: List[Dict[str, object]] = []

    # Baseline
    print("\n[Baseline] Full cache (no eviction)")
    llm_base = LLM(model=args.model, trust_remote_code=True)
    r = benchmark_one(llm_base, prompts, sp, "FullCache")
    r["cache_budget"] = -1
    results.append(r)
    del llm_base

    # PagedEviction at each budget
    for budget in args.cache_budgets:
        print(f"\n[PagedEviction] cache_budget={budget}")
        llm = LLM(model=args.model, trust_remote_code=True)
        cfg = PagedEvictionConfig(
            cache_budget=budget,
            protect_recent_blocks=args.protect_recent_blocks,
        )
        try:
            mgr = apply_paged_eviction(llm, cfg)
            r = benchmark_one(llm, prompts, sp, f"PagedEviction_{budget}")
            r["cache_budget"] = budget
            r["eviction_stats"] = str(mgr.get_stats())
            results.append(r)
        except Exception as exc:
            print(f"  ERROR: {exc}")
        finally:
            del llm

    # Save CSV
    model_short = args.model.split("/")[-1].replace("-", "_")
    out_path = args.output_csv or f"benchmarks/results_{model_short}.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["label", "cache_budget", "throughput_tok_s", "tpot_ms", "total_tokens", "elapsed_s"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
