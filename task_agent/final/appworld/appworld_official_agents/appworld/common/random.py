from __future__ import annotations

import random
import string


def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)


def get_unique_id(length: int = 8) -> str:
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))
