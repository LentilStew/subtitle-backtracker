import marisa_trie
from functools import reduce

# https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
import os
import srt
from typing import Callable, Iterator, Union, Literal
import struct
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read("config.ini")


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
        curr_index = int(fp.readline())

        if search == curr_index:
            return True
        elif search > curr_index:
            return _binary_search_index(fp, search, mid + 1, high)
        else:
            return _binary_search_index(fp, search, low, mid - 1)
    else:
        return False


def binary_search_index(file, index):
    with open(file, "rb") as fp:
        if (
            index == 1
        ):  # the find_next_index can't find this, it crashesm _find_next_index finds it but is significantly slower
            fp.readline()  # skip index
            start, end = fp.readline().split(b" --> ")
            content = fp.readline()
        else:
            fp.seek(0, os.SEEK_END)
            file_size = fp.tell()

            res = _binary_search_index(fp, index, 0, file_size - 1)

            if res == False:
                print(f"Subtitle not found for index {index}")

            start, end = fp.readline().split(b" --> ")
            content = fp.readline()

    return srt.Subtitle(
        index=index,
        start=srt.srt_timestamp_to_timedelta(start.decode("utf-8")),
        end=srt.srt_timestamp_to_timedelta(end.decode("utf-8")),
        content=content,
    )


def long_id_get_id(idx_raw):
    return idx_raw[:11]


def long_id_get_index_list(idx_raw):
    numbers_raw = idx_raw[12:]  # 1 byte is padding
    max_numbers = len(numbers_raw) // 2
    numbers = struct.unpack(f"{max_numbers}H", numbers_raw)
    return list(numbers)


def long_id_get_index_bytes(idx_raw):
    return idx_raw[12:]


def split_numbers(numbers_raw):
    max_numbers = len(numbers_raw) // 2
    numbers = struct.unpack(f"{max_numbers}H", numbers_raw)
    return list(numbers)


class TrieTranscript:
    def __init__(
        self,
        transcripts_paths: list[str],
        trie: Union[str | marisa_trie.RecordTrie],
        word_rarity=True,
    ) -> None:
        self.transcripts_paths: list[str] = transcripts_paths

        self.id_to_path: dict = {
            bytes(os.path.basename(path), "utf-8"): path for path in transcripts_paths
        }
        if isinstance(trie, str):
            self.trie: marisa_trie.BytesTrie = self.load_trie(trie)

        if isinstance(trie, marisa_trie.BytesTrie):
            self.trie: marisa_trie.BytesTrie = trie
        if word_rarity:
            self.word_rarity_update()

    def find_sections_stream(self, words_indexes: dict):
        stream_array = np.zeros(np.iinfo(np.dtype("uint16")).max)

        for word, indexes in words_indexes.items():
            indexes = np.frombuffer(indexes, np.dtype("uint16"))
            for index in indexes:
                stream_array[index] += self.word_rarity.get(word, 0)

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

    def load_trie(self, path, mmap=True):
        self.trie = marisa_trie.BytesTrie()
        if (
            mmap
        ):  # Maybe mmap?  https://marisa-trie.readthedocs.io/en/latest/tutorial.html#memory-mapped-i-o
            self.trie.mmap(path)
        else:
            self.trie.load(path)

        return self.trie

    def word_rarity_update(self) -> dict:
        self.word_rarity = {}
        for key, val in self.trie.iteritems():
            self.word_rarity[key] = self.word_rarity.get(key, 0)
            self.word_rarity[key] += len(val) - 12  # -12 because of the id
        for key, val in self.word_rarity.items():
            self.word_rarity[key] = 1 / val

        return self.word_rarity

    # generator, yields full subtitle
    # TODO add option to get multiple subtitles, i.e 3 after
    def bsearch_transcript(
        self,
        word,
        filter=lambda id, *args, **kwargs: False,  # if true it's skipped
        *args,
        **kwargs,  # this function is called before searching for the word, args and kwargs are passed to the function
    ) -> Union[Iterator[tuple[srt.Subtitle, str]] | None]:

        ids = self.trie.get(word.lower())
        if ids is None:
            return

        for value in ids:
            id = long_id_get_id(value)
            indexes = long_id_get_index_list(value)
            for index in indexes:
                if filter(id, *args, **kwargs):
                    continue

                path = self.id_to_path.get(id, None)

                if path is None:
                    print(f"word found in {id} but transcript file not found")
                    continue
                try:
                    res = (binary_search_index(path, index), id)
                except Exception as err:
                    print(err, index, word, path, id)
                    continue

                yield res

    def bsearch_transcript_index(self, idx, index) -> Union[srt.Subtitle | None]:

        path = self.id_to_path.get(idx, None)

        if path is None:
            print(f"transcript file not found")
            return None
        try:
            res = binary_search_index(path, index)
        except Exception as err:
            return None

        return res

    def get_mutual_stream(
        self, words: Union[str, list[str]]
    ):  # returns ids to all instances of the streams where all words are mentioned

        if isinstance(words, str):
            words = [words]

        initial = set(long_id_get_id(raw_idx) for raw_idx in self.trie.get(words[0]))

        for word in words[1:]:
            initial.intersection_update(
                [long_id_get_id(raw_idx) for raw_idx in self.trie.get(word)]
            )
            print(len(self.trie.get(word)), word)

        return initial

    def get_rarity(self, words: Union[str, list[str]]):
        if isinstance(words, str):
            words = [words]

        words_set = set(words)
        
        # Define rarity thresholds (lower score means more common)
        rarity_thresholds = {
            'super_extremely_common': 0.000001,  # Most common category
            'extremely_common': 0.00001,  # Most common category
            'very_common': 0.0001,
            'common': 0.001,
            'rare': 0.01,  # Adjusted to capture a broader range of rare words
            'very_rare': 0.1,
            'extremely_rare': 1.0  # Rarest category
        }

        result = {category: [] for category in rarity_thresholds}

        for word in words_set:
            rarity_score = self.word_rarity.get(word, -1)
            if rarity_score <= rarity_thresholds['super_extremely_common']:
                result['super_extremely_common'].append(word)
            if rarity_score <= rarity_thresholds['extremely_common']:
                result['extremely_common'].append(word)
            elif rarity_score <= rarity_thresholds['very_common']:
                result['very_common'].append(word)
            elif rarity_score <= rarity_thresholds['common']:
                result['common'].append(word)
            elif rarity_score <= rarity_thresholds['rare']:
                result['rare'].append(word)
            elif rarity_score <= rarity_thresholds['very_rare']:
                result['very_rare'].append(word)
            else:
                result['extremely_rare'].append(word)

        return result

    def get_indexes(
        self, words: Union[str, list[str]], mutually_inclusive: bool = False
    ):  # returns paths to all instances of the word mentioned, and the index in the timestamp

        if isinstance(words, str):
            words = [words]
        words_set = set(words)
        if mutually_inclusive:
            ########################################long_id_get_index_list(raw_idx)
            first_word = words_set.pop()
            res = {
                long_id_get_id(raw_idx): {first_word: raw_idx[12:]}
                for raw_idx in self.trie.get(first_word)
            }
            for word in words_set:

                new_dict = dict()
                raw_ids = self.trie.get(word)
                if raw_ids is None:
                    return None

                for raw_idx in raw_ids:

                    video_idx = long_id_get_id(raw_idx)

                    idx_dict = res.get(video_idx, None)
                    if idx_dict is None:
                        continue
                    new_dict[video_idx] = idx_dict

                    new_dict[video_idx][word] = raw_idx[
                        12:
                    ]  # long_id_get_index_list(raw_idx)

                if not bool(new_dict):
                    return {}

                res = new_dict

        else:
            res = {}
            for word in words_set:
                raw_ids = self.trie.get(word)

                if raw_ids is None:
                    continue

                for raw_idx in raw_ids:
                    idx = long_id_get_id(raw_idx)
                    res[idx] = res.get(idx, {})
                    res[idx][word] = long_id_get_index_bytes(raw_idx)

        return res

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
                    numbers_format = f"{len(unique_indexes)}H"
                    string_format = f"11s"
                    packed_data = struct.pack(
                        f"{string_format}{numbers_format}",
                        os.path.basename(f).encode(),
                        *unique_indexes,
                    )
                    words_indexes.append((word, packed_data))

        print(f"key value pairs added {len(words_indexes)}")

        trie = marisa_trie.BytesTrie(words_indexes)
        if trie_path is not None:
            print(f"saving to {trie_path}")
            trie.save(trie_path)
        return cls(transcripts_paths, trie)
