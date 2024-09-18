from TrieTranscript import (
    TrieTranscript,
    binary_search_index,
    raw_id_get_id,
    raw_id_get_index_list,
)
from helper import get_transcript
from TrieTranscriptIndex import TrieTranscriptIndex
import configparser
import os
import srt
from typing import Union
from datetime import datetime
from helper import yt_link_format
import timeit
import marisa_trie
import struct
import numpy as np

import numpy as np
import timeit





import json
import time

config = configparser.ConfigParser()
config.read("config.ini")
all_transcripts = [
    os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], idx)
    for idx in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
]


# with open("./test","w") as f:
#    json.dump(get_transcript("MEZ3sKdaRXI"),f)

with open("./test", "r") as f:
    test_transcript = json.load(f)  # "MEZ3sKdaRXI"

# its from _2UfRAmNZU8
trie_list = TrieTranscript(all_transcripts, "./transcripts-list.marisa")
import matplotlib.pyplot as plt


def test():

    all_words = [w for subtitle in test_transcript for w in subtitle["text"].split(" ")]

    print("start")
    words_classified = trie_list.get_rarity(all_words)
    streams_set = trie_list.get_indexes(
        [
            *words_classified["extremely_rare"],
            *words_classified["very_rare"],
            *words_classified["rare"],
        ]
    )
    common_words_array = [*words_classified["very_common"], *words_classified["common"]]
    common_words_in_order = list(filter(lambda x: x in common_words_array, all_words))

    common_words = trie_list.get_indexes(
        [
            *words_classified["very_common"],
            *words_classified["common"],
            *words_classified["extremely_common"],
            
        ]
    )

    def find_common_elements(arr1, arr2, arr3):

        # Initialize pointers for each array
        i, j, k = 0, 0, 0
        result = []

        # Traverse all three arrays
        while i < len(arr1) and j < len(arr2) and k < len(arr3):
            # If elements are equal, it's a common element
            if arr1[i] == arr2[j] == arr3[k]:
                # Add to result and move all pointers
                result.append(arr1[i])
                i += 1
                j += 1
                k += 1
            # Move the pointer for the smallest element
            elif arr1[i] < arr2[j]:
                i += 1
            elif arr2[j] < arr3[k]:
                j += 1
            else:
                k += 1

        return result

    def find_sections_stream(trie, words_indexes: dict, common_words):
        stream_array = np.zeros(np.iinfo(np.dtype("uint16")).max)

        for word, indexes in words_indexes.items():
            indexes = np.frombuffer(indexes, np.dtype("uint16"))
            for index in indexes:
                stream_array[index] += trie.word_rarity.get(word, 0)

        for start in range(len(common_words_in_order) - 3):
            for common_index in find_common_elements(
                np.frombuffer(
                    common_words.get(common_words_in_order[start], b""),
                    dtype=np.dtype("uint16"),
                ),
                np.frombuffer(
                    common_words.get(common_words_in_order[start + 1], b""),
                    dtype=np.dtype("uint16"),
                ),
                np.frombuffer(
                    common_words.get(common_words_in_order[start + 2], b""),
                    dtype=np.dtype("uint16"),
                ),
            ):
                stream_array[common_index] += 0.1

        clumps = []
        in_clump = False
        empty_count = 0
        empty_count_limit = 2

        start_index = 0

        for i in range(len(stream_array)):
            if stream_array[i] != 0 and not in_clump:
                start_index = i
                in_clump = True
            elif stream_array[i] == 0 and in_clump:
                empty_count += 1
                if empty_count > empty_count_limit:
                    clumps.append((start_index, i - empty_count_limit - 1))
                    in_clump = False
                    empty_count = 0

        if in_clump:
            clumps.append((start_index, len(stream_array) - 1))

        return stream_array, clumps

    biggest = 0
    biggest_link = ""
    print("end")
    # Loop through each item in streams_set and plot
    for i, (idx, words_indexes) in enumerate(streams_set.items()):
        stream_array, clumps = find_sections_stream(
            trie_list, words_indexes, common_words[idx]
        )
        for clump in clumps:
            curr = np.sum(stream_array[clump[0] : clump[1] + 1])

            if curr > biggest:
                biggest += curr
                subtitle = trie_list.bsearch_transcript_index(idx, clump[0])
                subtitle2 = trie_list.bsearch_transcript_index(idx, clump[1])

                biggest_link = (
                    yt_link_format(idx, subtitle.start),
                    subtitle2.end.total_seconds(),
                )
                # Sort the dictionary by values in descending order

                plt.figure(figsize=(14, 7))
                plt.plot(
                    stream_array[0 : np.max(np.nonzero(stream_array)) + 50],
                    marker="o",
                    linestyle="-",
                    color="b",
                    label="Reverse Rarity",
                )

                plt.xlabel("Time (Sequential Order)")
                plt.ylabel("Reverse Rarity")
                plt.title("Time Series Plot of Reverse Word Rarity")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{i}foo{idx}.png")
    print(biggest_link)


time = timeit.timeit(test, number=1)
print(time)
"""
import matplotlib.pyplot as plt

reverse = trie_list.word_rarity_update()
# Sort the dictionary by values in descending order
sorted_reverse = dict(sorted(reverse.items(), key=lambda item: item[1], reverse=True))

# Create time series data: use index as time
times = list(range(len(sorted_reverse)))  # Creating a time axis (0, 1, 2, ..., n-1)
values = list(sorted_reverse.values())    # Reverse rarity values
words = list(sorted_reverse.keys())       # Words associated with the values

# Plotting the time series
plt.figure(figsize=(14, 7))
plt.plot(times, values, marker='o', linestyle='-', color='b', label='Reverse Rarity')

plt.xlabel('Time (Sequential Order)')
plt.ylabel('Reverse Rarity')
plt.title('Time Series Plot of Reverse Word Rarity')
plt.grid(True)
plt.tight_layout()
plt.savefig('foo3.png')

"""
