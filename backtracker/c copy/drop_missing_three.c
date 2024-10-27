#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>

static inline void add_to_buffer(uint8_t *buffer, uint64_t value)
{
    int count = 0;
    while (value)
    {
        // Check if the bit is set
        if (value & 1)
        {
            // Calculate and print the global bit index
            buffer[count]++;
        }
        count++;
        value >>= 1;
    }
}

void drop_missing_three(
    const uint8_t *arr1, const uint8_t *arr2, const uint8_t *arr3,
    uint8_t *buffer, const int size)
{

    for (size_t i = 0; i < size; i += 32)
    {
        __m256i v1 = _mm256_load_si256((const __m256i *)(arr1 + i));
        __m256i v2 = _mm256_load_si256((const __m256i *)(arr2 + i));
         __m256i v3 = _mm256_load_si256((const __m256i *)(arr3 + i));
        __m256i tmp = _mm256_and_si256(_mm256_and_si256(v1, v2), v3);
        add_to_buffer(buffer + (i * 8) + 0, _mm256_extract_epi64(tmp, 0));
        add_to_buffer(buffer + (i * 8) + 64, _mm256_extract_epi64(tmp, 1));
        add_to_buffer(buffer + (i * 8) + 128, _mm256_extract_epi64(tmp, 2));
        add_to_buffer(buffer + (i * 8) + 192, _mm256_extract_epi64(tmp, 3));
    }
}

void drop_missing_three3(
    const uint8_t *arr1, const uint8_t *arr2, const uint8_t *arr3,
    uint8_t *buffer, const int size)
{
    for (size_t i = 0; i < size; i += 8) // Process 8 elements at a time
    {
        // Initialize temporary storage for results
        uint64_t result = 0;

        // Loop through the 8 elements to check conditions
        for (size_t j = 0; j < 8; j++)
        {
            if ((arr1[i + j] & arr2[i + j]) & arr3[i + j]) // Perform the AND operations
            {
                result |= (1ULL << j); // Set the corresponding bit in result
            }
        }

        // Add the result to the buffer
        add_to_buffer(buffer + (i / 8) * 8, result);
    }
}