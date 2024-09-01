import marisa_trie

# https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
import os
import srt
from typing import Callable, Iterator, Union, Literal
import struct
import configparser

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


class TrieTranscript:
    def __init__(
        self,
        transcripts_paths: list[str],
        fmt: Literal[
            "11s", "11sL"
        ],  # the first 11 bytes are the name of the file and the id of the yt video, and the last 4 the index of the word appearing
        trie: Union[str | marisa_trie.RecordTrie],
    ) -> None:
        self.transcripts_paths: list[str] = transcripts_paths
        self.fmt: str = fmt

        self.id_to_path: dict = {
            bytes(os.path.basename(path),"utf-8"): path for path in transcripts_paths
        }
        if isinstance(trie, str):
            self.trie: marisa_trie.RecordTrie = self.load_trie(trie)

        if isinstance(trie, marisa_trie.RecordTrie):
            self.trie: marisa_trie.RecordTrie = trie

    def load_trie(self, path, mmap=True):
        self.trie = marisa_trie.RecordTrie(self.fmt)
        if (
            mmap
        ):  # Maybe mmap?  https://marisa-trie.readthedocs.io/en/latest/tutorial.html#memory-mapped-i-o
            self.trie.mmap(path)
        else:
            self.trie.load(path)

        return self.trie

    # generator, yields full subtitle
    # TODO add option to get multiple subtitles, i.e 3 after
    def bsearch_transcript(
        self,
        word,
        filter=lambda id, *args, **kwargs: False,  # if true it's skipped
        *args,
        **kwargs,  # this function is called before searching for the word, args and kwargs are passed to the function
    ) -> Union[Iterator[tuple[srt.Subtitle, str]] | None]:

        if self.fmt != "<15s":
            print("Trie doesn't have indexes")

        ids = self.trie.get(word.lower())

        if ids is None:
            return

        for value in ids:
            id, index = value[0]  # check

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

    def fast_mutually_inclusive_get_indexes(self,words: Union[str, list[str]]):
        
        if isinstance(words, str):
            words = [words]

        res = set(idx[0] for idx in self.trie.get(words[0]))
        for word in words[1:]:
            res.intersection_update(idx[0] for idx in self.trie.get(word))

        return res
    def get_indexes(
        self, words: Union[str, list[str]], mutually_inclusive: bool = True
    ):  # returns paths to all instances of the word mentioned, and the index in the timestamp

        if isinstance(words, str):
            words = [words]

        if mutually_inclusive:
            res = {}
            first = True

            for word in words:
                ids = self.trie.get(word.lower())
                if ids is None:
                    return

                new_res = {}
                for idx, index in ids:
                    if first:
                        new_res[idx] = new_res.get(idx, {"path": self.id_to_path[idx]})
                        new_res[idx][word] = new_res[idx].get(word, [])
                        new_res[idx][word].append(index)
                        continue

                    prev_idx_dict = res.get(idx, None)
                    if prev_idx_dict is None:
                        continue

                    prev_idx_dict[word] = prev_idx_dict.get(word, [])
                    prev_idx_dict[word].append(index)

                    new_res[idx] = prev_idx_dict

                res = new_res
                
                if not bool(res):
                    return None
                
                first = False

            return res
        else:
            res = {}

            for word in words:
                ids = self.trie.get(word.lower())

                for idx_raw in ids:
                    idx, index = idx_raw[0]

                    res[idx] = res.get(idx, {"path": self.id_to_path[idx]})
                    res[idx][word] = res[idx].get(word, [])

                    res[idx][word].append(index)

            return res

    @classmethod
    def create_trie(cls, transcripts_paths, trie_path=None, save_index=True,fmt="11s"):

        key_val_pairs = []
        for i, f in enumerate(transcripts_paths):
            print(f"({i+1}/{len(transcripts_paths)}) opening {f}")

            with open(f, encoding="utf-8") as fp:
                subs = srt.parse(fp)

                for sub in subs:
                    for word in sub.content.split(" "):
                        if save_index:
                            val = (
                                bytes(os.path.basename(f), encoding="utf-8"),
                                sub.index,
                            )
                        else:
                            val = os.path.basename(f)

                        key_val_pairs.append((word.lower(), val))

        print(f"key value pairs added {len(key_val_pairs)}")
        trie = marisa_trie.RecordTrie(fmt, key_val_pairs)

        if trie_path is not None:
            print(f"saving to {trie_path}")
            trie.save(trie_path)

        return trie

    @classmethod
    def create_trie_transcript(cls, transcripts_paths, trie_path=None, save_index=True,fmt="11s"):
        trie = TrieTranscript.create_trie(transcripts_paths, trie_path, save_index,fmt=fmt)
        return cls(trie=trie, transcripts_paths=transcripts_paths, fmt=fmt)
