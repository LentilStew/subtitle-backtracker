import marisa_trie
from functools import reduce
from backtracker.WordInstances import WordInstances

# https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
import os
import srt
from typing import Callable, Iterator, Union, Literal
import struct
import configparser
import numpy as np
from numba.typed import Dict
from numba import types

config = configparser.ConfigParser()
config.read("config.ini")


def split_numbers(numbers_raw):
    max_numbers = len(numbers_raw) // 2
    numbers = struct.unpack(f"{max_numbers}H", numbers_raw)
    return list(numbers)


def raw_id_get_id(idx_raw):
    return idx_raw[:11]


def raw_id_get_index_list(idx_raw):
    numbers_raw = idx_raw[12:]  # 1 byte is padding
    max_numbers = len(numbers_raw) // 2
    numbers = struct.unpack(f"{max_numbers}H", numbers_raw)
    return list(numbers)


def raw_id_get_index_bytes(idx_raw, dtype):
    return np.frombuffer(idx_raw, dtype=dtype, offset=12)


def raw_id_get_id_bytes(idx_raw, dtype="uint8"):
    return np.frombuffer(idx_raw, dtype=dtype, offset=0, count=11)


def read_subtitle(fp) -> srt.Subtitle:
    fp.readline()  # /n
    index = fp.readline()  # either index or eof
    if index == b"":
        return None

    start, end = fp.readline().split(b" --> ")
    content = fp.readline().decode("utf-8").strip()
    return srt.Subtitle(
        index=int(index),
        start=srt.srt_timestamp_to_timedelta(start.decode("utf-8")),
        end=srt.srt_timestamp_to_timedelta(end.decode("utf-8")),
        content=content,
    )


def _binary_search_index(fp, search, low, high):
    def _find_next_index(fp):
        chars = [b" ", b" "]

        while True:
            chars[0] = chars[1]  # shift
            chars[1] = fp.read(1)
            if chars[0] == b"\n" and chars[1] == b"\n":
                fp.read(1)
                break
            if fp.tell() == 1:
                fp.seek(-1, 1)
                return
            fp.seek(-2, 1)

    def find_next_index(fp):
        chars = [b" ", b" "]

        while True:
            chars[0] = chars[1]  # shift
            chars[1] = fp.read(1)
            if chars[0] == b"\n" and chars[1] == b"\n":
                break

    if high >= low:
        mid = low + (high - low) // 2
        fp.seek(mid)
        find_next_index(fp)

        line = fp.readline()  # either eof or index
        if line == b"":
            return False

        curr_index = int(line)

        if search == curr_index:
            return True

        elif search > curr_index:
            return _binary_search_index(fp, search, mid + 1, high)
        else:
            return _binary_search_index(fp, search, low, mid - 1)
    else:
        return False


def binary_search_index(
    file, index, index_end=0
):  # returns either None, or srt.Subtitle, or [srt.Subtitle 's]
    with open(file, "rb") as fp:
        if (
            index == 1
        ):  # the find_next_index can't find this, it crashesm _find_next_index finds it but is significantly slower
            fp.readline()  # skip index
            start, end = fp.readline().split(b" --> ")
            content = fp.readline().decode("utf-8").strip()
            subtitle = srt.Subtitle(
                index=index,
                start=srt.srt_timestamp_to_timedelta(start.decode("utf-8")),
                end=srt.srt_timestamp_to_timedelta(end.decode("utf-8")),
                content=content,
            )
        else:
            fp.seek(0, os.SEEK_END)
            file_size = fp.tell()
            res = _binary_search_index(fp, index, 0, file_size - 1)

            if res == False:
                print(f"Subtitle not found for index {index}")
                return None
            start, end = fp.readline().split(b" --> ")
            content = fp.readline().decode("utf-8").strip()
            subtitle = srt.Subtitle(
                index=index,
                start=srt.srt_timestamp_to_timedelta(start.decode("utf-8")),
                end=srt.srt_timestamp_to_timedelta(end.decode("utf-8")),
                content=content,
            )

        if index_end != 0 and index_end > index:
            last_subtitle = subtitle
            subtitles = [subtitle]

            while last_subtitle.index <= index_end:
                last_subtitle = read_subtitle(fp)
                if last_subtitle is None:
                    break
                subtitles.append(last_subtitle)

            return subtitles
        else:
            return subtitle


class TrieTranscript(marisa_trie.BytesTrie):
    def __init__(self, transcripts_paths: list[str], trie_path: str) -> None:

        super().__init__()
        self.transcripts_paths: list[str] = transcripts_paths

        self.id_to_path: dict = {
            bytes(os.path.basename(path), "utf-8"): path for path in transcripts_paths
        }

        self.mmap(trie_path)  # also self.load is valid

        self.word_rarity = self.compute_word_rarity()

    def compute_word_rarity(self) -> dict:
        word_rarity = {}

        for word, raw in self.iteritems():
            instances = WordInstances(raw)
            instances.buffer
            word_rarity[word] = word_rarity.get(word, 0)
            word_rarity[word] += instances.count

        for word, val in word_rarity.items():
            word_rarity[word] = 1 / val

        return word_rarity

    # generator, yields subtitle for each mention of the word, in each stream
    # TODO add option to get multiple subtitles, i.e 3 after
    def bsearch_transcripts_by_word(
        self, word
    ) -> Union[Iterator[tuple[srt.Subtitle, str]] | None]:

        raw_results = self.get(word.lower())
        if raw_results is None:
            return

        for raw_result in raw_results:
            wi = WordInstances(raw_result)
            for index in wi:

                path = self.id_to_path.get(wi.idx, None)

                if path is None:
                    print(f"word found in {wi.idx} but transcript file not found")
                    continue
                try:
                    res = (binary_search_index(path, index), wi.idx)
                except Exception as err:
                    print(err, index, word, path, wi.idx)
                    continue

                yield res

    # returns subtitle, from transcript idx, in index
    def bsearch_transcript_by_index(
        self, idx, index, index_end=0
    ) -> Union[srt.Subtitle | None]:
        # returns index from idx
        path = self.id_to_path.get(idx, None)

        if path is None:
            print(f"transcript file not found")
            return None
        try:
            res = binary_search_index(path, index, index_end)
        except Exception as err:
            return None

        return res

    def bsearch_transcript_slice(self, idx, slice: slice) -> Union[srt.Subtitle | None]:
        # returns index from idx
        path = self.id_to_path.get(idx, None)

        if path is None:
            print(f"transcript file not found")
            return None
        try:
            res = binary_search_index(path, slice.start, slice.stop)
        except Exception as err:
            return None

        return res

    def get_word_rarity(self, words: Union[str, list[str]]):
        if isinstance(words, str):
            words = [words]

        words_set = set(words)

        # Define rarity thresholds (lower score means more common)
        rarity_thresholds = {
            "super_extremely_common": 0.000001,  # Most common category
            "extremely_common": 0.00001,  # Most common category
            "very_common": 0.0001,
            "common": 0.001,
            "rare": 0.01,  # Adjusted to capture a broader range of rare words
            "very_rare": 0.1,
            "extremely_rare": 1.0,  # Rarest category
        }

        result = {category: [] for category in rarity_thresholds}

        for word in words_set:
            rarity_score = self.word_rarity.get(word, -1)
            if rarity_score <= rarity_thresholds["super_extremely_common"]:
                result["super_extremely_common"].append(word)
            if rarity_score <= rarity_thresholds["extremely_common"]:
                result["extremely_common"].append(word)
            elif rarity_score <= rarity_thresholds["very_common"]:
                result["very_common"].append(word)
            elif rarity_score <= rarity_thresholds["common"]:
                result["common"].append(word)
            elif rarity_score <= rarity_thresholds["rare"]:
                result["rare"].append(word)
            elif rarity_score <= rarity_thresholds["very_rare"]:
                result["very_rare"].append(word)
            else:
                result["extremely_rare"].append(word)

        return result

    def get_mutual_word_streams(
        self, words: Union[str, list[str]]
    ):  # returns ids to all instances of the streams where all words are mentioned

        if isinstance(words, str):
            words = [words]

        initial = set(raw_id_get_id(raw_idx) for raw_idx in self.get(words[0]))

        for word in words[1:]:
            initial.intersection_update(
                [raw_id_get_id(raw_idx) for raw_idx in self.get(word)]
            )
            print(len(self.get(word)), word)

        return initial

    def get_words_indexes_mutually_inclusive(self, words: Union[str, list[str]]):
        if isinstance(words, str):
            words = [words]
        words_set = set([word.lower() for word in words])

        first_word = words_set.pop()
        res = {
            raw_id_get_id(raw_idx): {
                first_word: raw_id_get_index_bytes(raw_idx, self.dtype)
            }
            for raw_idx in self.get(first_word)
        }
        for word in words_set:

            new_dict = dict()
            raw_ids = self.get(word)
            if raw_ids is None:
                return None

            for raw_idx in raw_ids:

                video_idx = raw_id_get_id(raw_idx)

                idx_dict = res.get(video_idx, None)
                if idx_dict is None:
                    continue
                new_dict[video_idx] = idx_dict

                new_dict[video_idx][word] = raw_id_get_index_bytes(raw_idx, self.dtype)

            if not bool(new_dict):
                return {}

            res = new_dict

    def get_words_indexes(
        self, words: Union[str, list[str]]
    ):  # returns paths to all instances of the word mentioned, and the index in the timestamp
        if isinstance(words, str):
            words = [words]
        words_set = set([word.lower() for word in words])

        res = {}
        for word in words_set:
            raw_results = self.get(word)

            if not raw_results:
                continue

            for raw_result in raw_results:
                instances = WordInstances(raw_result)

                res[instances.idx] = res.get(instances.idx, {})
                res[instances.idx][word] = instances

        return res

    @classmethod
    def create_trie(cls, transcripts_paths, trie_path=None, padded=False):
        words_indexes = []
        for i, f in enumerate(transcripts_paths):
            print(f"({i+1}/{len(transcripts_paths)}) opening {f}")

            with open(f, encoding="utf-8") as fp:
                subs = srt.parse(fp)
                curr_words_indexes = {}
                for sub in subs:
                    for word in sub.content.split(" "):
                        word_lower = word.lower()
                        curr_words_indexes[word_lower] = curr_words_indexes.get(
                            word_lower, []
                        )
                        curr_words_indexes[word_lower].append(sub.index)

                for word, indexes in curr_words_indexes.items():
                    unique_indexes = sorted(set(indexes))

                    wi = WordInstances.pack(
                        idx=os.path.basename(f).encode(),
                        instances=unique_indexes,
                    )

                    words_indexes.append((word, wi.buffer))

        print(f"key value pairs added {len(words_indexes)}")

        trie = marisa_trie.BytesTrie(words_indexes)

        if trie_path is not None:
            print(f"saving to {trie_path}")
            trie.save(trie_path)

        return cls(transcripts_paths, trie_path)