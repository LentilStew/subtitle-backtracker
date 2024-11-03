from flask import Flask, request, jsonify
from flask.wrappers import Request
import os
import configparser
from flask import abort

from backtracker.VideoData import db
from backtracker.TrieTranscript import TrieTranscript
from backtracker.TranscriptBuffer import TranscriptBufferQuery, TranscriptBufferStream
from backtracker.TrieSearch import TrieSearch, TrieSearchSubtitle


config = configparser.ConfigParser()
config.read("config.ini")


def init():
    global trie
    all_transcripts = [
        os.path.join(config["data"]["TRANSCRITPS_FOLDER"], idx)
        for idx in os.listdir(config["data"]["TRANSCRITPS_FOLDER"])
    ]
    trie = TrieTranscript(all_transcripts, "./transcripts-list.marisa")


init()

import json

app = Flask(__name__)
from functools import reduce


@app.route("/quick_search", methods=["POST"])
def search_text_short():
    MAX_RES = 10
    if request.method != "POST":
        return

    request_data: dict = request.json

    words = helper.text_to_words(request_data.get("query"))

    res = trie.update_idx_word_map(map(helper.format_word_trie, words))

    if not isinstance(res, dict):
        abort(404)

    count = 1
    results = []
    for idx, stream in res.items():
        curr_stream = {}
        for word, instances in stream.items():
            curr_stream[word] = instances.tolist()

        sub = trie.bsearch_transcript_by_index(idx, instances[0])
        if sub is None:
            abort(404)
        idx: bytes
        results.append(
            {
                "idx": idx.decode("ascii"),
                "stream": curr_stream,
                "preview": sub.content,
            }
        )

        count += 1
        if count > MAX_RES:
            break

    return results
