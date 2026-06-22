from __future__ import annotations

from typing import Any, TypeAlias

Number: TypeAlias = int | float


class FromDict:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(subclass):
            cls._registry[name] = subclass
            subclass.type = name
            return subclass

        return decorator

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        config = dict(config)
        type_name = config.pop("type")
        if type_name not in cls._registry:
            raise ValueError(f"Unknown {cls.__name__} type: {type_name}")
        return cls._registry[type_name](**config)
