from __future__ import annotations


class AppWorld:
    init_defaults = type("InitDefaults", (), {"experiment_name": "harnessfix"})()

    def __init__(self, *args, **kwargs):
        raise RuntimeError("HarnessFix runs AppWorld inside Docker; use the adapter world.")

    @classmethod
    def initializer(cls, *args, **kwargs):
        class _Initializer:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Initializer()


def load_task_ids(dataset_name: str) -> list[str]:
    return []
