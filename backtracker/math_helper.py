from numba import njit, types, cfunc, jit
import numpy as np

def drop_missing_three(arr1, arr2, arr3, buffer):
    end = min(arr1[-1], arr2[-1], arr3[-1])

    if arr2[0] > arr1[0]:
        tmp = arr2
        arr2 = arr1
        arr1 = tmp

    if arr3[0] > arr1[0]:
        tmp = arr3
        arr3 = arr1
        arr1 = tmp

    start = arr1[0]

    if end < start:
        return

    j = 0
    while arr2[j] < start:
        j += 1
    l = 0
    while arr3[l] < start:
        l += 1


    i = 0
    while 1:
        if arr1[i] == arr2[j] == arr3[l]:
            buffer[arr1[i]] += 1
            if arr1[i] == end:
                break

            i += 1
            j += 1
            l += 1

        elif arr1[i] < arr2[j]:
            if arr1[i] >= end:
                break
            i += 1
        elif arr2[j] < arr3[l]:
            if arr2[j] >= end:
                break
            j += 1
        else:
            if arr3[l] >= end:
                break
            l += 1


@njit()
def drop_missing_three_old(arr1, arr2, arr3, buffer):
    i = j = l = 0
    while i < arr1.size and j < arr2.size and l < arr3.size:
        if arr1[i] == arr2[j] == arr3[l]:
            buffer[arr1[i]] += 1
            i += 1
            j += 1
            l += 1
        elif arr1[i] < arr2[j]:
            i += 1
        elif arr2[j] < arr3[l]:
            j += 1
        else:
            l += 1


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
