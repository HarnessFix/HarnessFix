from __future__ import annotations


def chunk_and_return(items: list, num_chunks: int, chunk_index: int, balanced: bool = True) -> list:
    if num_chunks <= 1:
        return items
    return items[chunk_index::num_chunks]
