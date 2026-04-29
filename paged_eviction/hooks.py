"""
vLLM integration for PagedEviction.

Patches the *live engine instance* returned by ``LLM()`` — never patches
classes, which avoids Python import-time class-caching issues.

Supports vLLM V1 engine (default in vLLM >= 0.8) and legacy V0 engine.
"""

from __future__ import annotations

import logging
import types
from typing import Any, List, Optional

import torch

from .config import PagedEvictionConfig
from .manager import PagedEvictionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_paged_eviction(
    llm: Any,
    config: PagedEvictionConfig,
) -> PagedEvictionManager:
    """
    Attach PagedEviction to a fully-initialised ``vllm.LLM`` instance.

    Args:
        llm: A ``vllm.LLM`` object (must already be initialised).
        config: Eviction configuration.

    Returns:
        The attached :class:`~paged_eviction.PagedEvictionManager`.
        Call ``.get_stats()`` on it at any time.

    Example::

        llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
        mgr = apply_paged_eviction(llm, PagedEvictionConfig(cache_budget=1024))
        outputs = llm.generate(prompts, SamplingParams(max_tokens=512))
        print(mgr.get_stats())
    """
    engine = _unwrap_engine(llm)
    kv_caches = _find_kv_caches(engine)
    block_size = _find_block_size(engine, kv_caches)

    if not kv_caches:
        raise RuntimeError(
            "PagedEviction: could not locate KV caches inside the engine. "
            "Make sure the model is fully loaded before calling apply_paged_eviction()."
        )

    mgr = PagedEvictionManager(
        kv_caches=kv_caches,
        block_size=block_size,
        cache_budget=config.cache_budget,
        protect_recent_blocks=config.protect_recent_blocks,
    )

    if _is_v1_engine(engine):
        logger.info("PagedEviction: V1 engine detected")
        _hook_v1(engine, mgr)
    else:
        logger.info("PagedEviction: V0 engine detected")
        _hook_v0(engine, mgr)

    logger.info(
        "PagedEviction ready — cache_budget=%d tokens, block_size=%d",
        config.cache_budget,
        block_size,
    )
    return mgr


# ---------------------------------------------------------------------------
# Engine / KV-cache discovery helpers
# ---------------------------------------------------------------------------


def _unwrap_engine(llm: Any) -> Any:
    """LLM → underlying engine."""
    if hasattr(llm, "llm_engine"):
        return llm.llm_engine
    if hasattr(llm, "_engine"):
        return llm._engine
    return llm


def _is_v1_engine(engine: Any) -> bool:
    # V1 engine lives in vllm.v1.*; V0 lives in vllm.engine.*
    # Both have model_executor, so we cannot use hasattr alone.
    mod = type(engine).__module__
    return "v1" in mod


def _find_kv_caches(engine: Any) -> List[torch.Tensor]:
    """Walk the engine object graph to locate per-layer KV cache tensors."""

    # ---- V1 paths ----
    v1_paths = [
        "model_executor.driver_worker.worker.model_runner.kv_cache",
        "model_executor.driver_worker.model_runner.kv_cache",
        "driver_worker.worker.model_runner.kv_cache",
        "driver_worker.model_runner.kv_cache",
    ]
    for path in v1_paths:
        val = _getattr_path(engine, path)
        if _valid_kv(val):
            return list(val)

    # ---- V0 primary: model_executor.driver_worker (GPUExecutor layout) ----
    v0_paths = [
        "model_executor.driver_worker.model_runner.kv_cache",
        "model_executor.driver_worker.cache_engine[0].gpu_cache",
    ]
    for path in v0_paths:
        val = _getattr_path(engine, path)
        if _valid_kv(val):
            return list(val)

    # ---- V0: engine.workers list ----
    workers = getattr(engine, "workers", None)
    if workers:
        worker = workers[0]
        for attr in [
            "model_runner.kv_cache",
            "cache_engine[0].gpu_cache",
        ]:
            val = _getattr_path(worker, attr)
            if _valid_kv(val):
                return list(val)  # type: ignore[arg-type]

    # ---- V0: cache_engine directly on engine ----
    ce = getattr(engine, "cache_engine", None)
    if ce is not None:
        if isinstance(ce, list):
            ce = ce[0]
        gpu = getattr(ce, "gpu_cache", None)
        if _valid_kv(gpu):
            return list(gpu)  # type: ignore[arg-type]

    # ---- Last resort: depth-first search for any list of KV tensors ----
    return _search_kv_caches(engine, depth=0)


def _valid_kv(val: Any) -> bool:
    """Return True if val looks like a list of per-layer KV tensors."""
    if not isinstance(val, (list, tuple)) or len(val) == 0:
        return False
    first = val[0]
    # Each element should be a tensor with at least 3 dims (2, blocks, block_size, ...)
    return hasattr(first, "shape") and len(first.shape) >= 3


def _search_kv_caches(obj: Any, depth: int) -> List[torch.Tensor]:
    """Recursively search object attributes for KV cache tensors (max depth 5)."""
    if depth > 5:
        return []
    for attr in vars(obj) if hasattr(obj, "__dict__") else []:
        try:
            val = getattr(obj, attr)
            if _valid_kv(val):
                return list(val)
            if hasattr(val, "__dict__") and depth < 5:
                result = _search_kv_caches(val, depth + 1)
                if result:
                    return result
        except Exception:
            pass
    return []


def _find_block_size(engine: Any, kv_caches: List[torch.Tensor]) -> int:
    for path in [
        "cache_config.block_size",
        "vllm_config.cache_config.block_size",
        "model_executor.driver_worker.cache_config.block_size",
    ]:
        val = _getattr_path(engine, path)
        if isinstance(val, int) and val > 0:
            return val
    # Infer from KV cache shape: [2, total_blocks, block_size, heads, dim]
    if kv_caches and len(kv_caches[0].shape) >= 3:
        return kv_caches[0].shape[2]
    return 16  # vLLM default


def _get_scheduler(engine: Any) -> Optional[Any]:
    sched = getattr(engine, "scheduler", None)
    if isinstance(sched, list):
        return sched[0] if sched else None
    return sched


def _getattr_path(obj: Any, path: str) -> Any:
    """Safely traverse a dotted path; returns None on any failure."""
    for part in path.split("."):
        if obj is None:
            return None
        if "[" in part:
            name, rest = part.split("[", 1)
            idx = int(rest.rstrip("]"))
            obj = getattr(obj, name, None)
            if obj is None:
                return None
            try:
                obj = obj[idx]
            except (IndexError, TypeError):
                return None
        else:
            obj = getattr(obj, part, None)
    return obj


# ---------------------------------------------------------------------------
# V1 engine hooks
# ---------------------------------------------------------------------------


def _hook_v1(engine: Any, mgr: PagedEvictionManager) -> None:
    original_step = engine.step

    def _step(self_engine: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        _v1_post_step(self_engine, mgr)
        return result

    engine.step = types.MethodType(_step, engine)
    _wrap_v1_finish(engine, mgr)


def _v1_post_step(engine: Any, mgr: PagedEvictionManager) -> None:
    scheduler = _get_scheduler(engine)
    if scheduler is None:
        return

    requests = getattr(scheduler, "requests", {})
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)

    for req_id, req in list(requests.items()):
        seq_len = _v1_seq_len(req)
        if seq_len <= 0:
            continue

        block_ids = _v1_block_ids(req, kv_mgr)
        if not block_ids:
            continue

        seq_id = hash(req_id)

        def _free(blk_id: int, _m: Any = kv_mgr, _r: Any = req) -> None:
            _v1_free_block(blk_id, _m, _r)

        state = mgr._state.get(seq_id)
        prefill_done = _v1_prefill_done(req)

        if prefill_done and (state is None or not state.prefill_evicted):
            new_ids = mgr.evict_prefill(seq_id, block_ids, seq_len, _free)
            _v1_set_block_ids(req, kv_mgr, new_ids)
        elif prefill_done:
            new_ids = mgr.evict_decode(seq_id, block_ids, seq_len, _free)
            if new_ids != block_ids:
                _v1_set_block_ids(req, kv_mgr, new_ids)


def _v1_seq_len(req: Any) -> int:
    for attr in ["num_computed_tokens", "num_prompt_tokens", "seq_len"]:
        val = getattr(req, attr, None)
        if isinstance(val, int):
            return val
    return 0


def _v1_prefill_done(req: Any) -> bool:
    status = getattr(req, "status", None)
    if status is not None:
        return str(status) in {"RUNNING", "DECODE", "RequestStatus.RUNNING"}
    return getattr(req, "num_generated_tokens", 0) > 0


def _v1_block_ids(req: Any, kv_mgr: Any) -> List[int]:
    bt = getattr(req, "block_table", None)
    if bt is not None:
        if hasattr(bt, "get_physical_block_ids"):
            return list(bt.get_physical_block_ids())
        if hasattr(bt, "physical_block_ids"):
            return list(bt.physical_block_ids)
        if isinstance(bt, (list, tuple)):
            return [int(x) for x in bt if x >= 0]
    if kv_mgr is not None:
        for attr in ["block_table", "get_block_table"]:
            fn = getattr(kv_mgr, attr, None)
            if callable(fn):
                try:
                    res = fn(req)
                    if isinstance(res, (list, tuple)):
                        return [int(x) for x in res if x >= 0]
                except Exception:
                    pass
    return []


def _v1_set_block_ids(req: Any, kv_mgr: Any, new_ids: List[int]) -> None:
    bt = getattr(req, "block_table", None)
    if bt is not None and hasattr(bt, "physical_block_ids"):
        try:
            bt.physical_block_ids = new_ids
        except Exception:
            pass


def _v1_free_block(blk_id: int, kv_mgr: Any, req: Any) -> None:
    if kv_mgr is None:
        return
    for method in ["free_block", "free", "release_block"]:
        fn = getattr(kv_mgr, method, None)
        if callable(fn):
            try:
                fn(blk_id)
                return
            except TypeError:
                try:
                    fn(req, blk_id)
                    return
                except Exception:
                    pass
            except Exception:
                pass


def _wrap_v1_finish(engine: Any, mgr: PagedEvictionManager) -> None:
    scheduler = _get_scheduler(engine)
    if scheduler is None:
        return
    orig = getattr(scheduler, "finish_requests", None)
    if orig is None:
        return

    def _finish(req_ids: Any, *args: Any, **kwargs: Any) -> Any:
        ids = req_ids if isinstance(req_ids, list) else [req_ids]
        for rid in ids:
            mgr.remove_sequence(hash(rid))
        return orig(req_ids, *args, **kwargs)

    scheduler.finish_requests = _finish


# ---------------------------------------------------------------------------
# V0 engine hooks
# ---------------------------------------------------------------------------


def _hook_v0(engine: Any, mgr: PagedEvictionManager) -> None:
    original_step = engine.step

    def _step(self_engine: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_step(*args, **kwargs)
        _v0_post_step(self_engine, mgr)
        return result

    engine.step = types.MethodType(_step, engine)


def _v0_post_step(engine: Any, mgr: PagedEvictionManager) -> None:
    scheduler = _get_scheduler(engine)
    if scheduler is None:
        return
    bm = getattr(scheduler, "block_manager", None)
    if bm is None:
        return

    for seq_group in getattr(scheduler, "running", []):
        for seq in seq_group.get_seqs():
            seq_id = seq.seq_id
            seq_len = seq.get_len()
            block_ids = _v0_block_ids(bm, seq_id)
            if not block_ids:
                continue

            def _free(blk_id: int, _bm: Any = bm) -> None:
                _v0_free_block(blk_id, _bm)

            state = mgr._state.get(seq_id)
            is_prefill = getattr(seq, "is_prefill", lambda: False)()

            if not is_prefill and (state is None or not state.prefill_evicted):
                new_ids = mgr.evict_prefill(seq_id, block_ids, seq_len, _free)
                _v0_update_block_table(bm, seq_id, block_ids, new_ids)
            elif not is_prefill:
                new_ids = mgr.evict_decode(seq_id, block_ids, seq_len, _free)
                if new_ids != block_ids:
                    _v0_update_block_table(bm, seq_id, block_ids, new_ids)
            else:
                mgr.register_sequence(seq_id)


def _v0_block_ids(bm: Any, seq_id: int) -> List[int]:
    tables = getattr(bm, "block_tables", {})
    blocks = tables.get(seq_id, [])
    return [b.block_number if hasattr(b, "block_number") else int(b) for b in blocks]


def _v0_free_block(blk_id: int, bm: Any) -> None:
    alloc = getattr(bm, "gpu_allocator", None)
    if alloc is None or not hasattr(alloc, "free"):
        return
    for blocks in getattr(bm, "block_tables", {}).values():
        for b in blocks:
            num = b.block_number if hasattr(b, "block_number") else b
            if num == blk_id:
                try:
                    alloc.free(b)
                except Exception:
                    pass
                return


def _v0_update_block_table(bm: Any, seq_id: int, old_ids: List[int], new_ids: List[int]) -> None:
    tables = getattr(bm, "block_tables", {})
    if seq_id not in tables:
        return
    evicted = set(old_ids) - set(new_ids)
    tables[seq_id] = [
        b
        for b in tables[seq_id]
        if (b.block_number if hasattr(b, "block_number") else b) not in evicted
    ]
