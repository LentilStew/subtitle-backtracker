import struct
import numpy as np
from typing import Any
import ctypes

PACKED_SHAPE = "11sLL\{\}H"


def bit_count(
    arr: np.array,
):  # https://stackoverflow.com/questions/63954102/numpy-vectorized-way-to-count-non-zero-bits-in-array-of-integers
    arr_view = arr.view(dtype=np.uint64)
    t = arr_view.dtype.type
    mask = t(0xFFFFFFFFFFFFFFFF)
    s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr_view = arr_view - ((arr_view >> 1) & s55)
    arr_view = (arr_view & s33) + ((arr_view >> 2) & s33)
    arr_view = (arr_view + (arr_view >> 4)) & s0F
    return (arr_view * s01) >> (
        8 * (arr_view.itemsize - 1)
    )  # its aligned probably _mm512_popcnt_epi64 is better


class WordInstances:
    def __init__(self, buffer):
        # the offset is in bits, not bytes
        self.idx, self.offset, self.instances_size, self.count = struct.unpack(
            "11sLLL", buffer[: struct.calcsize("11sLLL")]
        )
        self.instances = np.frombuffer(
            buffer[struct.calcsize("11sLLL") :], dtype=np.uint8
        )  # Error maybe padding between L and array
        self.buffer = buffer

        # Create the aligned instances array
        self.instances_aligned = self.allocate_aligned_array(self.instances.size)
        # Copy data to the aligned array
        np.copyto(self.instances_aligned, self.instances)

    def allocate_aligned_array(self, num_elements, alignment=32):
        size = num_elements + (alignment - 1)

        raw_memory = bytearray(size)
        ctypes_raw_type = ctypes.c_char * size
        ctypes_raw_memory = ctypes_raw_type.from_buffer(raw_memory)
        raw_address = ctypes.addressof(ctypes_raw_memory)
        offset = raw_address % alignment
        offset_to_aligned = (alignment - offset) % alignment
        ctypes_aligned_type = ctypes.c_char * (size - offset_to_aligned)
        ctypes_aligned_memory = ctypes_aligned_type.from_buffer(
            raw_memory, offset_to_aligned
        )
        return np.frombuffer( ctypes_aligned_memory,dtype=np.uint8,count=num_elements)

    @classmethod
    def pack(cls, idx: str, offset: int, bits_instances: np.ndarray[Any]):
        """Packs the struct data into bytes using struct.pack."""
        packed_data = struct.pack(
            f"11sLLL{bits_instances.size}B",
            idx,
            offset,
            bits_instances.size,
            bit_count(bits_instances).sum(),
            *bits_instances,
        )
        return cls(packed_data)

    def instance_iterator(self):
        for byte_index in np.nonzero(self.instances)[0]:
            byte = self.instances[byte_index]
            for bit_position in range(8):
                if byte & (1 << bit_position):
                    yield self.offset * 8 + (byte_index * 8) + bit_position

    def __iter__(self):
        return self.instance_iterator()

    def __str__(self):
        return f"""{[np.binary_repr(v, width=8)[::-1] for v in self.instances[0:10]]} ..."""

    def __repr__(self):

        return f"""{self.idx} {self.offset} {self.instances_size} {[np.binary_repr(v, width=8) for v in self.buffer[0:10]]}"""
