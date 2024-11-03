from backtracker.TrieTranscript import TrieTranscript, TrieTranscriptVideo
from backtracker.TrieSearch import TrieSearchSubtitle, text_slice_to_subtitle_slice
import os
import configparser
import numpy as np
import json
from backtracker.Filter import DateFilter
import backtracker.VideoData as VideoData
from backtracker.TranscriptBuffer import TranscriptBufferStream, TranscriptBufferQuery
from backtracker.helper import (
    yt_link_format,
    yt_link_format_seconds,
)
import srt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

config = configparser.ConfigParser()
config.read("config.ini")
trie_selected = config["data"]["MARISA_TRIE_FAST"]
transcript_limit = 5000


def create_trie():
    TrieTranscriptVideo.create_trie(VideoData.video_data, trie_selected)


if __name__ == "__main__":
    with open("./data/man.json", "r") as f:
        test_transcript_json = json.load(f)
    VideoData.load_video_data()
    t = TrieTranscriptVideo(VideoData.video_data,trie_selected)

    all_transcripts = sorted(
        [
            os.path.join(config["data"]["TRANSCRITPS_FOLDER"], idx)
            for idx in os.listdir(config["data"]["TRANSCRITPS_FOLDER"])
        ]
    )
    t = TrieTranscript(all_transcripts, config["data"]["MARISA_TRIE"])


    TS = TrieSearchSubtitle(t, test_transcript_json)

    import time
    print("Started searching")
    curr1 = time.time()
    queries = TS.define_search(max_search=5)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        # Submit each input as a separate process task
        futures = [pool.submit(TS.search_idx, querie) for querie in queries]

        # Collect results as they complete
        for future in as_completed(futures):
            tb = future.result()
            tb_slice: slice
            for tb_slice in tb.slices:
                if not sum(tb.buffer[tb_slice]) > 1:
                    continue
                tbq: TranscriptBufferQuery
                tbq = TS.backsearch_slice(slice=tb_slice, idx=tb.idx)
                if tbq is None:
                    continue

                subtitles = [
                    word
                    for sub in t.bsearch_transcript_slice(tb.idx, tb_slice)
                    for word in sub.content.lower().split(" ")
                ]

                sub: srt.Subtitle
                print(" ".join(subtitles))
                print(
                    yt_link_format(
                        tb.idx,
                        t.bsearch_transcript_by_index(tb.idx, tb_slice.start).start,
                    ),
                    yt_link_format(
                        tb.idx,
                        t.bsearch_transcript_by_index(tb.idx, tb_slice.stop).start,
                    ),
                )

                seconds = lambda t: (
                    datetime.strptime(t, "%H:%M:%S.%f") - datetime(1900, 1, 1)
                ).total_seconds()
                query_slice: slice
                for query_slice in tbq.slices:
                    correct_slice = text_slice_to_subtitle_slice(
                        query_slice, test_transcript_json
                    )
                    print(
                        "\t",
                        yt_link_format_seconds(
                            "MEZ3sKdaRXI",
                            seconds(test_transcript_json[correct_slice.start]["start"]),
                        ),
                    )

    tb: TranscriptBufferStream

    curr2 = time.time()
    print("Time ", curr2 - curr1)

    # for s,tb in results:
    #    if sum(tb.buffer[s]) < 1:
    #        continue
    #    section = t.bsearch_transcript_slice(tb.idx,s)
    #    print(tb.idx)
    #    for sub in section:
    #        print("\t", sub.content)
