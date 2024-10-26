from TrieTranscript import TrieTranscript, binary_search_index
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
from numba import njit, types, cfunc, jit
from numba.typed import Dict
from typing import overload
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read("config.ini")
from helper import yt_link_format, yt_link_format_seconds
from math_helper import *
import multiprocessing as mp
import time


class TranscriptBuffer:
    def __init__(self, buffer, dtype="uint16", ntuple_value=0.1) -> None:
        self.buffer = buffer
        self.dtype = np.dtype(dtype)
        self.ntuple_value = ntuple_value
        self.slices = []

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

    def clear_buffer(self):
        self.buffer.fill(0)

    def slice(self):
        masked_buffer = np.ma.masked_equal(self.buffer, 0)
        # if the buffer is empty it doesn't return a list

        if (
            masked_buffer.size == 1
        ):  ## if buffer is empty returns of size 1 don't know whyyyyy
            return []

        self.slices = np.ma.clump_unmasked(masked_buffer)
        return self.slices


test_count = 0


def save_test(arr1, arr2, arr3, arr4):
    global test_count
    
    if test_count == 100:
        exit(0)

    if arr1.size == 0 or arr2.size == 0 or arr3.size == 0:
        return

    with open(f"./c/tests/test{test_count}", "wb") as f:
        test_count += 1

        f.write(struct.pack("L", arr1.size))
        f.write(struct.pack("L", arr2.size))
        f.write(struct.pack("L", arr3.size))
        f.write(struct.pack("L", arr4.size))
        f.write(arr1)
        f.write(arr2)
        f.write(arr3)
        f.write(arr4)


class TranscriptBufferStream(TranscriptBuffer):
    def __init__(self, idx="", dtype="uint16"):
        _dtype = np.dtype(dtype)
        self.buffer = np.zeros(np.iinfo(_dtype).max)
        super().__init__(self.buffer, dtype)
        self.idx = idx

    def graph_buffer(self):
        plt.figure(figsize=(10, 6))
        print(f"last {np.max(np.nonzero(self.buffer)) + 50}")
        plt.plot(
            self.buffer[0 : np.max(np.nonzero(self.buffer)) + 50],
            color="dodgerblue",
            linestyle="-",
            marker="o",
            markersize=2,
        )

        plt.grid(True, linestyle="--", color="gray", alpha=0.7)

        plt.title("Transcript Buffer Visualization", fontsize=16, style="italic")
        plt.xlabel("Index", fontsize=12)
        plt.ylabel("Value", fontsize=12)

        plt.legend(["Buffer values"], loc="upper right", fontsize=10)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.savefig(f"{self.idx}.png")
        return True

    def add_words(self, word_index_map: dict, word_rarity_map: dict):
        for word, indexes in word_index_map.items():
            for index in indexes:
                self.buffer[index] += word_rarity_map.get(word, 0)

    def add_ntuples(self, word_index_map: dict, transcript: list):
        buffer = np.zeros(np.iinfo(self.dtype).max)
        empty = np.zeros(0, dtype="uint16")
        for i in range(len(transcript) - 2):
            drop_missing_three(
                word_index_map.get(transcript[i + 0], empty),
                word_index_map.get(transcript[i + 1], empty),
                word_index_map.get(transcript[i + 2], empty),
                buffer,
            )
            
        self.buffer += buffer * self.ntuple_value


class TranscriptBufferQuery(TranscriptBuffer):
    def __init__(self, transcript_query, dtype="uint16"):
        self.transcript_query = transcript_query

        self.word_index_map = {}
        self.buffer_size = len(transcript_query)

        for i, word in enumerate(transcript_query):

            if word not in self.word_index_map:
                self.word_index_map[word] = []

            self.word_index_map[word].append(i)

        for word in self.word_index_map:
            self.word_index_map[word] = np.array(
                self.word_index_map[word], dtype=np.uint16
            )
        self.buffer = np.zeros(self.buffer_size)
        super().__init__(self.buffer, dtype)

    def add_ntuples(self, sub_transcript: list):
        ntuple_size = 5
        buffer = np.zeros(self.buffer_size)
        empty = np.zeros(0, dtype="uint16")

        if len(sub_transcript) == 1:
            return

        if len(sub_transcript) < ntuple_size:
            find_consecutive_tuples(
                [
                    self.word_index_map.get(sub_transcript[i], empty)
                    for i in range(len(sub_transcript))
                ],
                buffer,
            )  # SUPER EDGE CASE NEVER HAPPENS, subtitles of less than 5 words
            return
        for i in range(len(sub_transcript) - ntuple_size):
            find_5tuple(
                self.word_index_map.get(sub_transcript[i + 0], empty),
                self.word_index_map.get(sub_transcript[i + 1], empty),
                self.word_index_map.get(sub_transcript[i + 2], empty),
                self.word_index_map.get(sub_transcript[i + 3], empty),
                self.word_index_map.get(sub_transcript[i + 4], empty),
                buffer,
            )

        self.buffer += buffer * self.ntuple_value


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
        transcript_timestamp: list[dict],  # Transcript to search, in trieTranscript
    ) -> None:
        self.trie: TrieTranscript = trie
        self.transcript_timestamp = transcript_timestamp
        self.transcript: list[str] = [
            w
            for subtitle in transcript_timestamp
            for w in subtitle["text"].lower().split(" ")
        ]
        self.transcript_rarity_map = self.trie.get_word_rarity(self.transcript)
        self.transcript_rarity_list = sorted(
            [
                (word, self.trie.word_rarity[word])
                for word in set(self.transcript)
                if word in self.trie.word_rarity
            ]
        )
        self.tbq: TranscriptBufferQuery = TranscriptBufferQuery(self.transcript)

    @staticmethod  # multiprocessing needs the function as static method
    def process_index(input_data):
        idx = input_data["idx"]
        rare_words = input_data["rare_words"]
        word_rarity = input_data["word_rarity"]
        common_words = input_data.get("common_words", None)
        transcript = input_data.get("transcript", None)

        print(input_data["i"])

        tb = TranscriptBufferStream(idx=idx)
        tb.add_words(rare_words, word_rarity)

        if common_words and transcript:
            tb.add_ntuples(input_data["common_words"], input_data["transcript"])

        tb.slice()

        return tb

    def search(self, use_multiprocessing=False):
        rare_word_list = [
            *self.transcript_rarity_map["extremely_rare"],
            *self.transcript_rarity_map["very_rare"],
            *self.transcript_rarity_map["rare"],
            # *self.transcript_rarity_map["very_common"],
            # *self.transcript_rarity_map["common"],
        ]

        if len(rare_word_list) == 0:
            rare_word_list = [w for w, _ in self.transcript_rarity_list[:10]]
        # Collect rare and common words
        rare_words = self.trie.get_words_indexes(rare_word_list)
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
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(
                    self.process_index,
                    inputs,
                )
        else:
            results = []
            for i in inputs:
                results.append(self.process_index(i))

        slices = [(curr_slice, tb) for tb in results for curr_slice in tb.slices]
        slices_sorted = sorted(
            slices, key=lambda i: i[1].buffer[i[0]].sum(), reverse=True
        )
        return slices_sorted

    @staticmethod  # transcript is a list of text, transcript_timestamp, is a list of dict {text,duration,start}, given an slice of the text, returns the slice but for the dict
    def text_slice_to_subtitle_slice(s: slice, transcript_timestamp):
        count = 0
        new_start = -1
        for i, subtitle in enumerate(transcript_timestamp):
            for w in subtitle["text"].lower().split(" "):

                count += 1

                if s.start == count:
                    new_start = i

                if s.stop == count:
                    return slice(new_start, i)

        return None

    # from an slice search back in the transcript
    def backsearch_slice(
        self, slice: slice, idx
    ) -> tuple[
        srt.Subtitle, list[slice]  # subtitle of the slice given
    ]:  # id where the slice is from

        subtitles_srt = binary_search_index(
            self.trie.id_to_path[idx], slice.start, slice.stop
        )

        if subtitles_srt is None:
            print(self.trie.id_to_path[idx], slice.start, slice.stop)
            return None

        subtitles = [word for sub in subtitles_srt for word in sub.content.split(" ")]
        self.tbq.add_ntuples(subtitles)
        slices = self.tbq.slice()

        if len(slices) == 0:
            return None

        res = (
            subtitles_srt,
            [
                VideoMatcher.text_slice_to_subtitle_slice(s, self.transcript_timestamp)
                for s in sorted(slices, key=lambda s: sum(self.tbq.buffer[s]))
            ],
        )

        self.tbq.clear_buffer()
        return res



def main():
    all_transcripts = [
        os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], idx)
        for idx in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
    ]
    trie = TrieTranscript(all_transcripts, "./transcripts-list.marisa")
    # https://www.youtube.com/watch?v=MEZ3sKdaRXI
    with open("./video_subs/man.json", "r") as f:
        test_transcript_json = json.load(f)

    vm = VideoMatcher(trie, test_transcript_json)
    print("start")
    res = vm.search(use_multiprocessing=False)

    for slice, tb in res:
        if not sum(tb.buffer[slice]) > 1:
            break

        res2 = vm.backsearch_slice(slice=slice, idx=tb.idx)
        if res2 is None:
            continue

        subtitles, query_slices = res2

        subtitles = [word for sub in subtitles for word in sub.content.split(" ")]
        print(" ".join(subtitles))
        print(
            yt_link_format(
                tb.idx,
                trie.bsearch_transcript_by_index(tb.idx, slice.start).start,
            ),
            yt_link_format(
                tb.idx,
                trie.bsearch_transcript_by_index(tb.idx, slice.stop).start,
            ),
        )
        from datetime import datetime

        seconds = lambda t: (
            datetime.strptime(t, "%H:%M:%S.%f") - datetime(1900, 1, 1)
        ).total_seconds()

        for query_slice in query_slices:
            print(
                "\t",
                yt_link_format_seconds(
                    "MEZ3sKdaRXI",
                    seconds(test_transcript_json[query_slice.start]["start"]),
                ),
            )


if __name__ == "__main__":
    from timeit import timeit

    time = timeit(main, number=1)
    print(time)
