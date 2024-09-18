from TrieTranscript import TrieTranscript
import marisa_trie
import os
import srt
from typing import (
    Callable,
    Iterator,
    Union,
    Literal,
)  # https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
import struct
import configparser
import numpy as np
import json
from functools import reduce
import multiprocessing
from numba import njit
from numba import njit, types
from numba.typed import Dict
from typing import overload

config = configparser.ConfigParser()
config.read("config.ini")
from helper import yt_link_format


@njit(fastmath = True)
def drop_missing_three(arr1, arr2, arr3, buffer):
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
    return buffer



import matplotlib.pyplot as plt


class TranscriptBuffer:
    def __init__(self, idx="", dtype="uint16") -> None:
        self.idx = idx
        self.dtype = np.dtype(dtype)
        self.buffer = np.zeros(np.iinfo(self.dtype).max)
        self.NTUPLE_VALUE = 0.1

    def graph_buffer(self):
        plt.figure(figsize=(10, 6))

        # Plot the buffer with an old-school aesthetic
        plt.plot(
            self.buffer[0 : np.max(np.nonzero(self.buffer)) + 50],
            color="dodgerblue",
            linestyle="-",
            marker="o",
            markersize=2,
        )

        # Add gridlines for that classic look
        plt.grid(True, linestyle="--", color="gray", alpha=0.7)

        # Set retro fonts
        plt.title("Transcript Buffer Visualization", fontsize=16, style="italic")
        plt.xlabel("Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)

        # Show a classic legend
        plt.legend(["Buffer values"], loc="upper right", fontsize=10)

        # Set old-school x and y ticks
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # Show the plot
        plt.savefig(f"{self.idx}.png")
        return True

    def add_words(self, word_index_map: dict, word_rarity_map: dict):
        for word, indexes in word_index_map.items():
            for index in indexes:
                self.buffer[index] += word_rarity_map.get(word, 0)

    def fill_gaps(self, fill_value=0.01):

        non_zero_indices = np.where(self.buffer != 0)[0]

        if len(non_zero_indices) < 2:
            return

        for i in range(len(non_zero_indices) - 1):
            start_index = non_zero_indices[i]
            end_index = non_zero_indices[i + 1]

            # Fill gaps between the current and next non-zero value
            if end_index > start_index + 1:
                self.buffer[start_index + 1 : end_index] = fill_value

        return self.buffer

    def add_ntuples(self, word_index_map, transcript):
        buffer = np.zeros(np.iinfo(self.dtype).max, dtype=np.dtype("float"))
        empty = np.empty(0, dtype=self.dtype)

        for i in range(len(transcript) - 2):
            drop_missing_three(
                word_index_map.get(transcript[i + 0], empty),
                word_index_map.get(transcript[i + 1], empty),
                word_index_map.get(transcript[i + 2], empty),
                buffer,
            )

        self.buffer += buffer * self.NTUPLE_VALUE

    def get_clumps(self):
        masked_buffer = np.ma.masked_equal(self.buffer, 0)
        return np.ma.clump_unmasked(masked_buffer)


def transcript_to_triplets(transcript):
    ntuples = [tuple(sorted(transcript[i : i + 3])) for i in range(len(transcript) - 2)]

    n2_to_n1 = {}
    for n3 in set(ntuples):
        fdouble = n3[0:2]
        sdouble = n3[1:3]

        n2_to_n1[fdouble] = n2_to_n1.get(fdouble, [])  # why not set?
        n2_to_n1[fdouble].append(n3[2])
        n2_to_n1[sdouble] = n2_to_n1.get(sdouble, [])  # why not set?
        n2_to_n1[sdouble].append(n3[0])

    sorted_items = sorted(n2_to_n1.items(), key=lambda x: len(x[1]), reverse=True)

    seen = set()
    final = {}
    for n2, n1s in sorted_items:
        for n1 in n1s:
            n3 = tuple(sorted((n2[0], n2[1], n1)))

            if n3 in seen:
                continue

            seen.add(n3)
            final.setdefault(n2, []).append(n1)

    return final


class VideoMatcher:
    def __init__(
        self,
        trie: TrieTranscript,
        transcript: list[str],  # Transcript to search, in trieTranscript
    ) -> None:
        self.trie: TrieTranscript = trie
        self.transcript: list[str] = transcript
        self.transcript_rarity_map = self.trie.get_word_rarity(transcript)
        self.transcript_rarity_list = sorted(
            [
                (word, self.trie.word_rarity[word])
                for word in set(transcript)
                if word in self.trie.word_rarity
            ]
        )

    @staticmethod  # multiprocessing needs the function outside
    def process_index(input_data):
        idx = input_data["idx"]
        rare_words = input_data["rare_words"]
        word_rarity = input_data["word_rarity"]
        common_words = input_data.get("common_words", None)
        transcript = input_data.get("transcript", None)

        print(input_data["i"])

        tb = TranscriptBuffer(idx=idx)
        tb.add_words(rare_words, word_rarity)
        if common_words and transcript:
            tb.add_ntuples(input_data["common_words"], input_data["transcript"])

        slices = [curr_slice for curr_slice in tb.get_clumps()]
        return slices, tb

    def run(self, use_multiprocessing=True):
        # Collect rare and common words
        rare_words = self.trie.get_words_indexes(
            [
                *self.transcript_rarity_map["extremely_rare"],
                *self.transcript_rarity_map["very_rare"],
                *self.transcript_rarity_map["rare"],
                # *self.transcript_rarity_map["very_common"],
                # *self.transcript_rarity_map["common"],
            ]
        )

        common_words = self.trie.get_words_indexes(
            [
                *self.transcript_rarity_map["extremely_rare"],
                *self.transcript_rarity_map["very_rare"],
                *self.transcript_rarity_map["rare"],
                *self.transcript_rarity_map["common"],
                *self.transcript_rarity_map["very_common"],
                *self.transcript_rarity_map["extremely_common"],
                # *self.transcript_rarity_map["super_extremely_common"],
            ]
        )

        # Function to process each index in parallel
        inputs = [
            {
                "idx": idx,
                "rare_words": rare_words[idx],
                "word_rarity": self.trie.word_rarity,
                "common_words": common_words.get(idx, None),
                "transcript": self.transcript,
                #            "triplets": triplets,
                # "ntuples": set(
                #    tuple(sorted(self.transcript[i : i + 3]))
                #    for i in range(len(self.transcript) - 2)
                # ),
                "i": i,
            }
            for i, idx in enumerate(rare_words.keys())
        ]

        if use_multiprocessing:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(
                    self.process_index,
                    inputs,
                )
        else:
            results = []
            for i in inputs:
                results.append(self.process_index(i))

        results = [r for r in results if r is not None]
        slices = [(curr_slice, tb) for slices, tb in results for curr_slice in slices]
        slices_sorted = sorted(
            slices, key=lambda i: i[1].buffer[i[0]].sum(), reverse=False
        )

        [
            print(
                yt_link_format(
                    tb.idx,
                    self.trie.bsearch_transcript_by_index(
                        tb.idx, curr_slice.start
                    ).start,
                ),
                int(tb.buffer[curr_slice].sum()),
                tb.idx,
            )
            for curr_slice, tb in slices_sorted
            if curr_slice and int(tb.buffer[curr_slice].sum()) > 1
        ]


from helper import get_transcript

if __name__ == "__main__":
    all_transcripts = [
        os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], idx)
        for idx in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
    ]
    trie = TrieTranscript(all_transcripts, "./transcripts-list.marisa")
    with open("./test", "r") as f:
        test_transcript = json.load(f)
    test_transcript = [
        w for subtitle in test_transcript for w in subtitle["text"].split(" ")
    ]

    vm = VideoMatcher(trie, test_transcript)
    print("start")
    import timeit

    time = timeit.timeit(vm.run, number=1)
    print(f"finished multithreading in {time}")
