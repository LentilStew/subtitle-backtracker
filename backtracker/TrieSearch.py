from backtracker.TrieTranscript import TrieTranscript, binary_search_index
import os
import srt
import configparser
import numpy as np
import json
from backtracker.TranscriptBuffer import (
    TranscriptBufferQuery,
    TranscriptBufferStream,
)
from typing import Callable, Iterator, Union, Literal

from backtracker.WordInstances import WordInstances

from backtracker.math_helper import *
import multiprocessing as mp
import time
from collections import OrderedDict

config = configparser.ConfigParser()
config.read("config.ini")


def text_slice_to_subtitle_slice(s: slice, transcript_timestamp):
    # transcript is a list of text, transcript_timestamp is a list of dicts {text,duration,start}, given an slice for the text, returns the slice but for the dict
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


class TrieSearch:
    def __init__(self, trie: TrieTranscript, transcript: list[str], filters=[]) -> None:
        self.trie: TrieTranscript = trie
        self.transcript: list[str] = transcript
        self.transcript_rarity_map = self.trie.sort_word_rarity(self.transcript)
        self.transcript_rarity_list = sorted(
            [
                (word, self.trie.get_word_rarity(word))
                for word in set(self.transcript)
                if self.trie.get_word_rarity(word, None) != None
            ],
            key=lambda v: v[1],
            reverse=True,
        )

    @staticmethod  # multiprocessing needs the function as static method
    def process_index(input_data):
        tb = TranscriptBufferStream(idx=input_data["idx"])

        if "rare_words" in input_data and "word_rarity" in input_data:
            tb.add_words(input_data["rare_words"], input_data["word_rarity_map"])

        if "common_words" in input_data and "transcript" in input_data:
            tb.add_ntuples(input_data["common_words"], input_data["transcript"])

        tb.slice()

        return tb

    def search(
        self,
        max_search=-1,
        rare_threshold=float(config["thresholds"]["rare"]),
        common_threshold=float(config["thresholds"]["super_extremely_common"]),
    ) -> Iterator[TranscriptBufferStream]:

        # id -> Word -> WordInstances
        rare_words: dict[str : dict[str:WordInstances]] = {}  # used for adding in index
        common_words: dict[str : dict[str:WordInstances]] = {}  # used for ntuples
        idx_words: OrderedDict[str : dict[str:WordInstances]] = (OrderedDict())  # used for choosing index order

        maps = {"rare": rare_words, "common": common_words, "indexes": idx_words}

        for word, rarity in self.transcript_rarity_list:
            self.trie.get_words_indexes(word, [m for m in maps.values()])

            if rarity < rare_threshold and "rare" in maps:
                maps.pop("rare")

            if rarity < common_threshold and "common" in maps:
                maps.pop("common")

            if len(idx_words) > max_search and "indexes" in maps:
                maps.pop("indexes")
                # indexes are chosen based on the appearance of rare words rather than making a score and sorting because otherwise long streams would be favoured
            if not maps:
                break

        ids = list(idx_words.keys())

        inputs = [
            {
                "idx": idx,
                "rare_words": rare_words[idx],
                "word_rarity_map": self.trie.word_rarity_map,
                "common_words": common_words[idx],
                "transcript": self.transcript,
                "i": i,
            }
            for i, idx in enumerate(ids)
            if idx in rare_words and idx in common_words
        ]

        for i,input in enumerate(inputs):
            if i >= max_search:
                break
            yield self.process_index(input)

        # slices = [(curr_slice, tb) for tb in results for curr_slice in tb.slices]
        # slices_sorted = sorted(
        #    slices, key=lambda i: i[1].buffer[i[0]].sum(), reverse=False
        # )

        # [
        #    print(
        #        yt_link_format(
        #            tb.idx,
        #            self.trie.bsearch_transcript_by_index(
        #                tb.idx, curr_slice.start
        #            ).start,
        #        ),
        #        int(tb.buffer[curr_slice].sum()),
        #        tb.idx,
        #    )
        #    for curr_slice, tb in slices_sorted
        #    if curr_slice and int(tb.buffer[curr_slice].sum()) > 1
        # ]

class VideoTrieSearch(TrieSearch):
    def __init__(
        self, trie: TrieTranscript, srt_dicts_transcript: list[dict], filters=[]
    ):
        transcript = [
            w
            for subtitle in srt_dicts_transcript
            for w in subtitle["text"].lower().split(" ")
        ]
        self.srt_dicts_transcript = srt_dicts_transcript
        super().__init__(trie, transcript)

    # from an slice search back in the transcript
    def backsearch_slice(
        self, slice: slice, idx
    ) -> tuple[
        srt.Subtitle, list[slice]  # subtitle of the slice given
    ]:  # id where the slice is from

        subtitles_srt: list[srt.Subtitle] = self.trie.bsearch_transcript_slice(
            idx, slice
        )

        if subtitles_srt is None:
            print("slice doesn't exist")
            print(idx, slice.start, slice.stop)
            return None

        subtitles = [
            word for sub in subtitles_srt for word in sub.content.lower().split(" ")
        ]

        self.tbq: TranscriptBufferQuery = TranscriptBufferQuery(self.transcript)

        self.tbq.add_ntuples(subtitles)

        if len(self.tbq.slice()) == 0:
            return None

        return self.tbq
