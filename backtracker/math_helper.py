from numba import njit, types, cfunc, jit
import numpy as np

import ctypes
import numpy as np

# Load the shared library
drop_missing_lib = ctypes.CDLL("./backtracker/c/drop_missing_three.so")

# Define the function argument types
drop_missing_lib.drop_missing_three.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # arr1
    ctypes.POINTER(ctypes.c_uint8),  # arr2
    ctypes.POINTER(ctypes.c_uint8),  # arr3
    ctypes.POINTER(ctypes.c_uint8),  # buffer
    ctypes.c_int,  # size
]


def c_drop_missing_three(arr1, arr2, arr3, offset1, offset2, offset3, buffer):
    overlap_start = max(offset1, offset2, offset3)
    overlap_end = min(offset1 + arr1.size, offset2 + arr2.size, offset3 + arr3.size)
    if overlap_start > overlap_end:
        return

    drop_missing_lib.drop_missing_three(
        arr1[overlap_start - offset1 : overlap_end - offset1].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        arr2[overlap_start - offset2 : overlap_end - offset2].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        arr3[overlap_start - offset3 : overlap_end - offset3].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        buffer[overlap_start * 8: overlap_end *8].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        overlap_end-overlap_start,
    )


# @njit(cache=True)
def drop_missing_three(arr1, arr2, arr3, offset1, offset2, offset3, buffer):
    overlap_start = max(offset1, offset2, offset3)
    overlap_end = min(offset1 + arr1.size, offset2 + arr2.size, offset3 + arr3.size)
    if overlap_start > overlap_end:
        return

    tmp_buffer = (
        arr1[overlap_start - offset1 : overlap_end - offset1]
        & arr2[overlap_start - offset2 : overlap_end - offset2]
        & arr3[overlap_start - offset3 : overlap_end - offset3]
    )
    buffer[overlap_start * 8 : overlap_end * 8] += np.unpackbits(tmp_buffer)


@njit
def find_5tuple(l1, l2, l3, l4, l5, buffer):
    s1, s2, s3, s4, s5 = 0, 0, 0, 0, 0

    for i1 in range(s1, l1.size):
        v1 = l1[i1]

        for i2 in range(s2, l2.size):
            v2 = l2[i2]
            if not v2 == v1 + 1:
                continue
            s2 = i2

            for i3 in range(s3, l3.size):
                v3 = l3[i3]
                if not v3 == v2 + 1:
                    continue
                s3 = i3

                for i4 in range(s4, l4.size):
                    v4 = l4[i4]
                    if not v4 == v3 + 1:
                        continue
                    s4 = i4

                    for i5 in range(s5, l5.size):
                        v5 = l5[i5]
                        if not v5 == v4 + 1:
                            continue

                        s5 = i5
                        buffer[v1:v5] += 1


def find_consecutive_tuples(lists, buffer):
    n = len(lists)
    print(lists)
    if any(v.size == 0 for v in lists):
        print("ENDING")
        return
    indices = [0] * n  # Initialize indices for each list

    while True:
        # Get current values from each list
        current_values = [lists[i][indices[i]] for i in range(n)]

        # Check if the current values are in strictly increasing order
        if all(current_values[i] < current_values[i + 1] for i in range(n - 1)):
            # Update the buffer using the minimum and maximum values
            buffer[current_values[0] : current_values[-1] + 1] += 1

        # Move to the next combination
        for i in range(n - 1, -1, -1):
            indices[i] += 1
            if indices[i] < len(lists[i]):
                break
            indices[i] = 0
        else:
            break  # Exit if all indices are out of bounds
