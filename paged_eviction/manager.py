"""
PagedEviction: block-wise KV cache eviction for vLLM.

Algorithm (EACL 2026, Chitty-Venkata et al.):
  Token importance:  S_i = ||V_i||_2 / ||K_i||_2   (averaged over heads & layers)
  Prefill eviction:  after prompt, evict lowest-scoring blocks to reach cache_budget
  Decode eviction:   each time seq_len % block_size == 0, evict 1 lowest-scoring block
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level importance computation
# ---------------------------------------------------------------------------


def _token_importance(
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    S_i = mean_over_heads( ||V_i_h||_2 / (||K_i_h||_2 + eps) )

    Args:
        k: shape [n_tokens, num_heads, head_dim]
        v: shape [n_tokens, num_heads, head_dim]

    Returns:
        shape [n_tokens]
    """
    k_norm = k.norm(dim=-1)  # [n_tokens, num_heads]
    v_norm = v.norm(dim=-1)  # [n_tokens, num_heads]
    return (v_norm / (k_norm + 1e-8)).mean(dim=-1)  # [n_tokens]


def compute_block_scores(
    kv_caches: List[torch.Tensor],
    block_ids: List[int],
    tokens_per_block: List[int],
) -> torch.Tensor:
    """
    Compute per-block importance score, averaged across all transformer layers.

    Args:
        kv_caches: List of per-layer tensors, each shaped
                   [2, total_blocks, block_size, num_heads, head_dim]
                   where index 0 = K cache, index 1 = V cache.
        block_ids: Physical block IDs for this sequence (logical order).
        tokens_per_block: Valid token count per block (last block may be partial).

    Returns:
        Tensor [num_blocks] — higher score means more important (keep).
    """
    num_blocks = len(block_ids)
    if num_blocks == 0:
        return torch.tensor([], dtype=torch.float32)

    device = kv_caches[0].device
    scores = torch.zeros(num_blocks, dtype=torch.float32, device=device)

    for layer_kv in kv_caches:
        k_cache = layer_kv[0]  # [total_blocks, block_size, num_heads, head_dim]
        v_cache = layer_kv[1]

        for i, (blk_id, n_tok) in enumerate(zip(block_ids, tokens_per_block)):
            if n_tok <= 0:
                continue
            k = k_cache[blk_id, :n_tok].float()  # [n_tok, heads, dim]
            v = v_cache[blk_id, :n_tok].float()
            scores[i] += _token_importance(k, v).mean()

    scores /= len(kv_caches)
    return scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokens_per_block(seq_len: int, num_blocks: int, block_size: int) -> List[int]:
    result = []
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, seq_len)
        result.append(max(0, end - start))
    return result


# ---------------------------------------------------------------------------
# Per-sequence state
# ---------------------------------------------------------------------------


class _SeqState:
    def __init__(self) -> None:
        self.prefill_evicted: bool = False
        self.last_eviction_len: int = 0
        self.decode_evictions: int = 0


# ---------------------------------------------------------------------------
# Main manager
# ---------------------------------------------------------------------------


class PagedEvictionManager:
    """
    Manages KV cache eviction for all active sequences.

    Attach to a live vLLM engine via :func:`paged_eviction.apply_paged_eviction`.
    You can also use this class directly for testing or custom integrations.

    Args:
        kv_caches: Per-layer KV tensors from the vLLM model runner.
        block_size: Tokens per paged block (vLLM default: 16).
        cache_budget: Max KV tokens to keep per sequence.
        protect_recent_blocks: Never evict the N most-recent blocks.
    """

    def __init__(
        self,
        kv_caches: List[torch.Tensor],
        block_size: int,
        cache_budget: int,
        protect_recent_blocks: int = 1,
    ) -> None:
        self.kv_caches = kv_caches
        self.block_size = block_size
        self.cache_budget = cache_budget
        self.protect_recent_blocks = max(1, protect_recent_blocks)
        self._state: Dict[int, _SeqState] = {}

    # ------------------------------------------------------------------
    # Sequence lifecycle
    # ------------------------------------------------------------------

    def register_sequence(self, seq_id: int) -> None:
        if seq_id not in self._state:
            self._state[seq_id] = _SeqState()

    def remove_sequence(self, seq_id: int) -> None:
        self._state.pop(seq_id, None)

    # ------------------------------------------------------------------
    # Prefill eviction
    # ------------------------------------------------------------------

    def evict_prefill(
        self,
        seq_id: int,
        block_ids: List[int],
        seq_len: int,
        free_block_fn: Callable[[int], None],
    ) -> List[int]:
        """
        After prefill, evict lowest-importance blocks to reach ``cache_budget``.

        Args:
            seq_id: Sequence identifier.
            block_ids: Physical block IDs in logical order.
            seq_len: Total prompt length.
            free_block_fn: Called with each physical block ID to release it.

        Returns:
            Surviving block IDs in logical order (update your block table with this).
        """
        self.register_sequence(seq_id)
        state = self._state[seq_id]

        if state.prefill_evicted:
            return block_ids

        budget_blocks = (self.cache_budget + self.block_size - 1) // self.block_size
        n_evict = len(block_ids) - budget_blocks

        if n_evict <= 0:
            state.prefill_evicted = True
            state.last_eviction_len = seq_len
            return block_ids

        tpb = _tokens_per_block(seq_len, len(block_ids), self.block_size)
        scores = compute_block_scores(self.kv_caches, block_ids, tpb)

        # Protect most-recent blocks
        protected = min(self.protect_recent_blocks, len(block_ids) - 1)
        if protected > 0:
            scores[-protected:] = float("inf")

        _, evict_idx = torch.topk(scores, k=n_evict, largest=False)
        evict_set = set(evict_idx.tolist())

        surviving: List[int] = []
        for i, blk_id in enumerate(block_ids):
            if i in evict_set:
                try:
                    free_block_fn(blk_id)
                except Exception as exc:
                    logger.warning("Could not free block %d: %s", blk_id, exc)
                    surviving.append(blk_id)
            else:
                surviving.append(blk_id)

        state.prefill_evicted = True
        state.last_eviction_len = seq_len
        logger.info(
            "prefill evict seq=%d  %d→%d blocks (budget=%d)",
            seq_id,
            len(block_ids),
            len(surviving),
            budget_blocks,
        )
        return surviving

    # ------------------------------------------------------------------
    # Decode eviction
    # ------------------------------------------------------------------

    def evict_decode(
        self,
        seq_id: int,
        block_ids: List[int],
        seq_len: int,
        free_block_fn: Callable[[int], None],
    ) -> List[int]:
        """
        After each decode step, evict one block when a block boundary is crossed.

        Eviction only fires when:
        1. ``seq_len % block_size == 0``  (a block just filled up)
        2. The number of blocks exceeds the budget
        3. This boundary hasn't already been evicted

        Args:
            seq_id: Sequence identifier.
            block_ids: Current physical block IDs in logical order.
            seq_len: Current total sequence length.
            free_block_fn: Called with the physical block ID to release.

        Returns:
            Updated block IDs (one shorter when eviction fires, else unchanged).
        """
        self.register_sequence(seq_id)
        state = self._state[seq_id]

        budget_blocks = (self.cache_budget + self.block_size - 1) // self.block_size

        # Only evict when over budget, at a block boundary, and not already done
        if len(block_ids) <= budget_blocks:
            return block_ids
        if seq_len % self.block_size != 0:
            return block_ids
        if seq_len == state.last_eviction_len:
            return block_ids

        tpb = _tokens_per_block(seq_len, len(block_ids), self.block_size)
        scores = compute_block_scores(self.kv_caches, block_ids, tpb)

        protected = min(self.protect_recent_blocks, len(block_ids) - 1)
        if protected > 0:
            scores[-protected:] = float("inf")

        evict_idx = int(scores.argmin().item())
        if scores[evict_idx].item() == float("inf"):
            return block_ids  # all candidates are protected

        evict_blk = block_ids[evict_idx]
        try:
            free_block_fn(evict_blk)
        except Exception as exc:
            logger.warning("Could not free block %d: %s", evict_blk, exc)
            return block_ids

        state.last_eviction_len = seq_len
        state.decode_evictions += 1
        return [b for j, b in enumerate(block_ids) if j != evict_idx]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, object]:
        """Return a dict with eviction statistics."""
        return {
            "active_sequences": len(self._state),
            "total_decode_evictions": sum(s.decode_evictions for s in self._state.values()),
            "cache_budget": self.cache_budget,
            "block_size": self.block_size,
        }
