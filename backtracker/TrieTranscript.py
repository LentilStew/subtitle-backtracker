import marisa_trie
from functools import reduce
from backtracker.WordInstances import WordInstances
from typing import IO
from backtracker.VideoData import VideoData
from backtracker.Filter import FilterChain

# https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
import os
import srt
from typing import Callable, Iterator, Union, Literal, Tuple,Optional,List,Dict
import struct
import configparser
import numpy as np
from numba.typed import Dict
from numba import types
import io

config = configparser.ConfigParser()
config.read("config.ini")


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
    fp, index, index_end=0
):  # returns either None, or srt.Subtitle, or [srt.Subtitle 's]
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

        self.word_rarity_map = self.compute_word_rarity()

    def id_to_fp(self, idx) -> Union[io.BufferedReader, None]:
        path = self.id_to_path.get(idx)
        if path:
            return open(path, "rb")
        return None

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
                try:
                    with self.id_to_fp(wi.idx) as fp:
                        res = (binary_search_index(fp, index), wi.idx)

                except TypeError as e:
                    print(f"Error: {e}, \n Probably {wi.idx} not found")
                    continue
                except Exception as err:
                    print(err, index, word, wi.idx)
                    continue

                yield res

    # returns subtitle, from transcript idx, in index
    def bsearch_transcript_by_index(
        self, idx, index, index_end=0
    ) -> Union[srt.Subtitle | None]:

        try:
            with self.id_to_fp(idx=idx) as fp:
                res = binary_search_index(fp, index, index_end)

        except TypeError as e:
            print(f"Error: {e}, \n Probably {idx} not found")
            return None

        except Exception as err:
            return None
        return res

    def bsearch_transcript_slice(self, idx, slice: slice) -> Union[srt.Subtitle | None]:
        try:
            with self.id_to_fp(idx=idx) as fp:

                res = binary_search_index(fp, slice.start, slice.stop)

        except TypeError as e:
            print(f"Error: {e}, \n Probably {idx} not found")
            return None

        except Exception as err:
            return None

        return res

    def get_word_rarity(self, word: str, default=None):
        return self.word_rarity_map.get(word, default)

    def sort_word_rarity(self, words: Union[str, list[str]]):

        if isinstance(words, str):
            words = [words]

        words_set = set(words)

        # Define rarity thresholds (lower score means more common)
        thresholds = {key: float(val) for key, val in config["thresholds"].items()}
        result = {category: [] for category in thresholds}

        for word in words_set:
            rarity_score = self.get_word_rarity(word, -1)
            if rarity_score == -1:
                continue
            elif rarity_score <= thresholds["super_extremely_common"]:
                result["super_extremely_common"].append(word)
            elif rarity_score <= thresholds["extremely_common"]:
                result["extremely_common"].append(word)
            elif rarity_score <= thresholds["very_common"]:
                result["very_common"].append(word)
            elif rarity_score <= thresholds["common"]:
                result["common"].append(word)
            elif rarity_score <= thresholds["rare"]:
                result["rare"].append(word)
            elif rarity_score <= thresholds["very_rare"]:
                result["very_rare"].append(word)
            else:
                result["extremely_rare"].append(word)

        return result

    """
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
    """

    def update_idx_word_map(
        self,
        words: Union[str, list[str]],
        idx_word_map: Optional[Union[Dict, List[Dict]]] = None,
        filter: Optional[FilterChain] = None,
    ):  # idx->word->instances # can append to idx_word_map

        if isinstance(words, str):
            words = [words]

        if idx_word_map is None:
            idx_word_map = {}

        words_set = set([word.lower() for word in words])

        for word in words_set:
            raw_results = self.get(word)

            if not raw_results:
                continue

            for raw_result in raw_results:
                instances = WordInstances(raw_result)

                if filter and not filter.run(instances):
                    continue

                if isinstance(idx_word_map, dict):
                    idx_word_map.setdefault(instances.idx, {})[word] = instances
                    continue

                for map in idx_word_map:
                    map.setdefault(instances.idx, {})[word] = instances

        return idx_word_map

    @classmethod
    def create_trie(cls, transcripts_paths, trie_path=None):
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


class TrieTranscriptVideo(TrieTranscript):
    def __init__(
        self,
        videos: dict[str, VideoData],
        trie_path: str,  # map of idx to IO
    ) -> None:

        super().__init__(
            [v.transcript_path for v in videos.values()], trie_path=trie_path
        )
        self.id_to_video: dict[str:VideoData] = videos

    @classmethod
    def create_trie(
        cls, transcripts_video_data_map: dict[str, VideoData], trie_path=None
    ):
        words_indexes = []

        for i, (idx, video_data) in enumerate(transcripts_video_data_map.items()):
            print(f"({i+1}/{len(transcripts_video_data_map)}) {idx}")
            subs = srt.parse(io.StringIO(video_data.transcript_srt), ignore_errors=True)
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
                    idx=bytearray(idx, "utf-8"),
                    instances=unique_indexes,
                )

                words_indexes.append((word, wi.buffer))

        print(f"key value pairs added {len(words_indexes)}")

        trie = marisa_trie.BytesTrie(words_indexes)

        if trie_path is not None:
            print(f"saving to {trie_path}")
            trie.save(trie_path)

        return cls(transcripts_video_data_map, trie_path)
