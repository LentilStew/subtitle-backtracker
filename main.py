from backtracker.TrieTranscript import TrieTranscript 
from backtracker.VideoMatching import VideoMatcher
import os
import configparser
import numpy as np
import json
import timeit
config = configparser.ConfigParser()
config.read("config.ini")
transcript_limit = 5
trie_selected = config["youtube"]["MARISA_TRIE"]
def create_trie():
    all_transcripts = sorted([
        os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], idx)
        for idx in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
    ])
    TrieTranscript.create_trie(all_transcripts[:transcript_limit],trie_selected)

def load_trie():
    all_transcripts = sorted([
        os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], idx)
        for idx in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
    ])
    return TrieTranscript(all_transcripts[:transcript_limit],trie_selected)

if __name__ == "__main__":
    #t = create_trie()
    t = load_trie()
    with open("./data/man.json", "r") as f:
        test_transcript_json = json.load(f)
    print("loaded trie :)")
    def tmp():
        VideoMatcher(t,test_transcript_json).search()
    res = timeit.timeit(tmp,number=1)
    print(f"Time: {res}")