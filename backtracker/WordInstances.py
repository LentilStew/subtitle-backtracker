import struct
import numpy as np
from typing import Any
import configparser

config = configparser.ConfigParser()
config.read("./config.ini")

PACKED_SHAPE = "11s{}H"


class WordInstances:
    def __init__(self, buffer):
        # the offset is in bits, not bytes
        (self.idx,) = struct.unpack("=11s", buffer[: struct.calcsize("11s")])
        self.instances = np.frombuffer(
            buffer[struct.calcsize("=11s") :],
            dtype=np.dtype(config["data"]["MARISA_TRIE_INSTANCE_TYPE"]),
        )  # Error maybe padding between 11s and array
        self.buffer = buffer

    @property
    def count(self):
        return len(self.instances)

    @classmethod
    def pack(cls, idx: str, instances: np.ndarray[Any]):
        packed_data = struct.pack(
            f"=11s{len(instances)}H",
            idx,
            *instances,
        )
        return cls(packed_data)

    def __iter__(self):
        return iter(self.instances)

    def __str__(self):
        return f"""first 10 values of {self.idx}: {self.instances[0:10]} ..."""

    def __repr__(self):
        return f"""{self.idx}"""
