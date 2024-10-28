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

config = configparser.ConfigParser()
config.read("config.ini")
from backtracker.helper import yt_link_format, yt_link_format_seconds
from backtracker.math_helper import *
import multiprocessing as mp
import time


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
        self.transcript_rarity_map = self.trie.get_word_rarity(self.transcript)
        self.transcript_rarity_list = sorted(
            [
                (word, self.trie.word_rarity[word])
                for word in set(self.transcript)
                if word in self.trie.word_rarity
            ]
        )

    @staticmethod  # multiprocessing needs the function as static method
    def process_index(input_data):
        tb = TranscriptBufferStream(idx=input_data["idx"])

        if "rare_words" in input_data and "word_rarity" in input_data:
            tb.add_words(input_data["rare_words"], input_data["word_rarity"])

        if "common_words" in input_data and "transcript" in input_data:
            tb.add_ntuples(input_data["common_words"], input_data["transcript"])

        tb.slice()

        return tb

    def search(
        self, use_multiprocessing=False
    ) -> list[tuple[slice, TranscriptBufferStream]]:
        # TODO add words from the rarity list until n streams

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
            slices, key=lambda i: i[1].buffer[i[0]].sum(), reverse=False
        )

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

        return slices_sorted


class VideoTrieSearch(TrieSearch):
    def __init__(self, trie: TrieTranscript, timestamped_transcript: list[dict], filters=[]):
        transcript = [
            w
            for subtitle in timestamped_transcript
            for w in subtitle["text"].lower().split(" ")
        ]
        super().__init__(trie, transcript)
        self.tbq: TranscriptBufferQuery = TranscriptBufferQuery(self.transcript)

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
                text_slice_to_subtitle_slice(s, self.transcript_timestamp)
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
