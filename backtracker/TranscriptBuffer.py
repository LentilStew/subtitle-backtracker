import numpy as np
import matplotlib.pyplot as plt
from backtracker.math_helper import drop_missing_three,find_consecutive_tuples,find_5tuple


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


class TranscriptBufferStream(TranscriptBuffer):
    def __init__(self, idx="", dtype="uint16",size=np.iinfo(np.dtype("uint16")).max):
        self.buffer = np.zeros(size)
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
        for i in range(len(transcript) - 2):
            if (
                transcript[i + 0] not in word_index_map
                or transcript[i + 1] not in word_index_map
                or transcript[i + 2] not in word_index_map
            ):
                continue
            
            drop_missing_three(
                word_index_map.get(transcript[i + 0]).instances,
                word_index_map.get(transcript[i + 1]).instances,
                word_index_map.get(transcript[i + 2]).instances,
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
            self.buffer += buffer * self.ntuple_value
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
