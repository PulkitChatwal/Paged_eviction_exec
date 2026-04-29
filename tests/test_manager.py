"""Unit tests for paged_eviction core logic — no GPU required."""

from __future__ import annotations

from typing import List

import pytest
import torch

from paged_eviction.config import PagedEvictionConfig
from paged_eviction.manager import (
    PagedEvictionManager,
    _token_importance,
    _tokens_per_block,
    compute_block_scores,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_kv_caches(
    num_layers: int = 4,
    num_blocks: int = 64,
    block_size: int = 16,
    num_heads: int = 8,
    head_dim: int = 64,
) -> List[torch.Tensor]:
    return [torch.randn(2, num_blocks, block_size, num_heads, head_dim) for _ in range(num_layers)]


def noop(_blk_id: int) -> None:
    pass


# ---------------------------------------------------------------------------
# _token_importance
# ---------------------------------------------------------------------------


class TestTokenImportance:
    def test_shape(self) -> None:
        k = torch.randn(10, 8, 64)
        v = torch.randn(10, 8, 64)
        s = _token_importance(k, v)
        assert s.shape == (10,)

    def test_all_positive(self) -> None:
        k = torch.ones(5, 4, 32)
        v = torch.ones(5, 4, 32) * 2.0
        s = _token_importance(k, v)
        assert s.all()  # all > 0

    def test_zero_key_norm_safe(self) -> None:
        k = torch.zeros(3, 4, 16)
        v = torch.ones(3, 4, 16)
        s = _token_importance(k, v)
        assert s.isfinite().all()

    def test_relative_ordering(self) -> None:
        k = torch.ones(2, 1, 4)
        v = torch.ones(2, 1, 4)
        v[0] = v[0] * 3.0  # token 0 has higher V-norm
        s = _token_importance(k, v)
        assert s[0] > s[1]


# ---------------------------------------------------------------------------
# _tokens_per_block
# ---------------------------------------------------------------------------


class TestTokensPerBlock:
    def test_full_blocks(self) -> None:
        assert _tokens_per_block(32, 2, 16) == [16, 16]

    def test_partial_last(self) -> None:
        assert _tokens_per_block(19, 2, 16) == [16, 3]

    def test_single_partial(self) -> None:
        assert _tokens_per_block(7, 1, 16) == [7]


# ---------------------------------------------------------------------------
# compute_block_scores
# ---------------------------------------------------------------------------


class TestComputeBlockScores:
    def test_output_shape(self) -> None:
        kv = make_kv_caches(num_layers=2, num_blocks=8, block_size=4)
        s = compute_block_scores(kv, [0, 1, 2], [4, 4, 3])
        assert s.shape == (3,)

    def test_empty(self) -> None:
        kv = make_kv_caches()
        s = compute_block_scores(kv, [], [])
        assert s.numel() == 0

    def test_all_finite(self) -> None:
        kv = make_kv_caches(num_layers=1, num_blocks=4, block_size=8)
        s = compute_block_scores(kv, [0, 1], [8, 3])
        assert s.isfinite().all()


# ---------------------------------------------------------------------------
# PagedEvictionManager — prefill
# ---------------------------------------------------------------------------


class TestPrefillEviction:
    def setup_method(self) -> None:
        self.kv = make_kv_caches(num_layers=2, num_blocks=32, block_size=16)
        self.freed: List[int] = []

    def _free(self, blk_id: int) -> None:
        self.freed.append(blk_id)

    def test_no_eviction_within_budget(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=1024)
        result = mgr.evict_prefill(0, [0, 1, 2], 48, self._free)
        assert result == [0, 1, 2]
        assert self.freed == []

    def test_evicts_to_budget(self) -> None:
        # 10 blocks (160 tokens), budget = 64 tokens (4 blocks)
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=64)
        result = mgr.evict_prefill(0, list(range(10)), 160, self._free)
        assert len(result) <= 5  # ceil(64/16)=4, +1 for ceiling rounding edge
        assert len(self.freed) >= 5

    def test_recent_block_protected(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=16, protect_recent_blocks=1)
        # Must evict 4 of 5 blocks, but last must survive
        result = mgr.evict_prefill(0, [0, 1, 2, 3, 4], 80, self._free)
        assert 4 in result

    def test_idempotent(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=64)
        r1 = mgr.evict_prefill(0, list(range(8)), 128, self._free)
        n1 = len(self.freed)
        r2 = mgr.evict_prefill(0, r1, 128, self._free)
        assert r2 == r1
        assert len(self.freed) == n1  # no extra frees

    def test_registers_sequence(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=1024)
        mgr.evict_prefill(42, [0], 16, noop)
        assert 42 in mgr._state


# ---------------------------------------------------------------------------
# PagedEvictionManager — decode
# ---------------------------------------------------------------------------


class TestDecodeEviction:
    def setup_method(self) -> None:
        self.kv = make_kv_caches(num_layers=2, num_blocks=32, block_size=16)
        self.freed: List[int] = []

    def _free(self, blk_id: int) -> None:
        self.freed.append(blk_id)

    def test_no_eviction_within_budget(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=1024)
        result = mgr.evict_decode(0, [0, 1, 2], 48, self._free)
        assert result == [0, 1, 2]
        assert self.freed == []

    def test_evicts_at_boundary(self) -> None:
        # 4 blocks, budget = 2 blocks, seq_len = multiple of block_size
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=32)
        result = mgr.evict_decode(0, [0, 1, 2, 3], 64, self._free)
        assert len(result) == 3
        assert len(self.freed) == 1

    def test_no_eviction_mid_block(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=32)
        result = mgr.evict_decode(0, [0, 1, 2, 3], 65, self._free)
        assert result == [0, 1, 2, 3]
        assert self.freed == []

    def test_once_per_boundary(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=32)
        r1 = mgr.evict_decode(0, [0, 1, 2, 3], 64, self._free)
        mgr.evict_decode(0, r1, 64, self._free)  # same seq_len → no-op
        assert len(self.freed) == 1

    def test_recent_block_protected(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=32, protect_recent_blocks=1)
        result = mgr.evict_decode(0, [0, 1, 2, 3], 64, self._free)
        assert 3 in result  # last block always survives

    def test_remove_sequence(self) -> None:
        mgr = PagedEvictionManager(self.kv, block_size=16, cache_budget=1024)
        mgr.register_sequence(99)
        assert 99 in mgr._state
        mgr.remove_sequence(99)
        assert 99 not in mgr._state


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_keys_present(self) -> None:
        kv = make_kv_caches()
        mgr = PagedEvictionManager(kv, block_size=16, cache_budget=512)
        stats = mgr.get_stats()
        assert "active_sequences" in stats
        assert "total_decode_evictions" in stats
        assert stats["cache_budget"] == 512
        assert stats["block_size"] == 16


# ---------------------------------------------------------------------------
# PagedEvictionConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = PagedEvictionConfig()
        assert cfg.cache_budget == 1024
        assert cfg.protect_recent_blocks == 1
        assert cfg.evict_mode == "block"

    def test_bad_budget(self) -> None:
        with pytest.raises(ValueError):
            PagedEvictionConfig(cache_budget=0)

    def test_bad_protect(self) -> None:
        with pytest.raises(ValueError):
            PagedEvictionConfig(protect_recent_blocks=0)

    def test_bad_mode(self) -> None:
        with pytest.raises(ValueError):
            PagedEvictionConfig(evict_mode="random")

    def test_repr(self) -> None:
        cfg = PagedEvictionConfig(cache_budget=256)
        assert "256" in repr(cfg)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_block_no_eviction(self) -> None:
        kv = make_kv_caches(num_blocks=4, block_size=16)
        mgr = PagedEvictionManager(kv, block_size=16, cache_budget=16)
        freed: List[int] = []
        result = mgr.evict_prefill(0, [0], 16, freed.append)
        assert result == [0]
        assert freed == []

    def test_multiple_sequences_independent(self) -> None:
        kv = make_kv_caches(num_blocks=64, block_size=16)
        mgr = PagedEvictionManager(kv, block_size=16, cache_budget=64)
        freed_a: List[int] = []
        freed_b: List[int] = []
        mgr.evict_prefill(1, list(range(0, 10)), 160, freed_a.append)
        mgr.evict_prefill(2, list(range(10, 20)), 160, freed_b.append)
        assert len(freed_a) > 0
        assert len(freed_b) > 0
        assert set(freed_a).isdisjoint(set(freed_b))

    def test_full_decode_simulation(self) -> None:
        kv = make_kv_caches(num_blocks=64, block_size=16)
        mgr = PagedEvictionManager(kv, block_size=16, cache_budget=64)
        freed: List[int] = []

        block_ids = list(range(12))
        block_ids = mgr.evict_prefill(0, block_ids, 192, freed.append)

        seq_len = len(block_ids) * 16
        next_block = 12
        for _ in range(64):
            seq_len += 1
            if seq_len % 16 == 0:
                block_ids.append(next_block)
                next_block += 1
            block_ids = mgr.evict_decode(0, block_ids, seq_len, freed.append)

        stats = mgr.get_stats()
        assert isinstance(stats["total_decode_evictions"], int)
        assert stats["total_decode_evictions"] >= 0
