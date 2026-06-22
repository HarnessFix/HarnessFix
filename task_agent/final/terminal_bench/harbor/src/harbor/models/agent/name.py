from enum import Enum


class AgentName(str, Enum):
    ORACLE = "oracle"
    NOP = "nop"
    TERMINUS_2 = "terminus-2"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}
