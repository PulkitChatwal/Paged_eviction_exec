"""Configuration for PagedEviction."""

from __future__ import annotations


class PagedEvictionConfig:
    """
    Configuration for the PagedEviction KV cache pruning strategy.

    Args:
        cache_budget: Max KV tokens to retain per sequence. Default 1024.
        protect_recent_blocks: Number of most-recent blocks never evicted. Default 1.
        evict_mode: ``"block"`` (recommended) or ``"token"`` (experimental).
        verbose: Enable per-eviction debug logging.

    Example::

        cfg = PagedEvictionConfig(cache_budget=512, protect_recent_blocks=2)
    """

    def __init__(
        self,
        cache_budget: int = 1024,
        protect_recent_blocks: int = 1,
        evict_mode: str = "block",
        verbose: bool = False,
    ) -> None:
        if cache_budget <= 0:
            raise ValueError("cache_budget must be a positive integer.")
        if protect_recent_blocks < 1:
            raise ValueError("protect_recent_blocks must be >= 1.")
        if evict_mode not in ("block", "token"):
            raise ValueError("evict_mode must be 'block' or 'token'.")

        self.cache_budget = cache_budget
        self.protect_recent_blocks = protect_recent_blocks
        self.evict_mode = evict_mode
        self.verbose = verbose

    def __repr__(self) -> str:
        return (
            f"PagedEvictionConfig("
            f"cache_budget={self.cache_budget}, "
            f"protect_recent_blocks={self.protect_recent_blocks}, "
            f"evict_mode={self.evict_mode!r})"
        )
