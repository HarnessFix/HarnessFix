from __future__ import annotations

import time
from datetime import datetime


class Timer:
    def __init__(self, start: bool = True) -> None:
        self.start_time: float | None = None
        if start:
            self.start()

    def start(self) -> None:
        self.start_time = time.time()

    def get_time(self) -> float:
        return time.time()


def freezegun_bypassed_datetime() -> datetime:
    return datetime.now()
