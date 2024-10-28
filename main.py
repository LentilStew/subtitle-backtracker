from backtracker.TrieTranscript import TrieTranscript
from backtracker.TrieSearch import TrieSearch
import os
import configparser
import numpy as np
import json
from backtracker.Filter import DateFilter

config = configparser.ConfigParser()
config.read("config.ini")
trie_selected = config["youtube"]["MARISA_TRIE_FAST"]
transcript_limit = 50


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
    #create_trie()
    t = load_trie()
    
    with open("./data/man.json", "r") as f:
        test_transcript_json = json.load(f)
    
    DateFilter()
    
    TS = TrieSearch(t,["Hello", "my", "name", "is"],filters=[])

    results = TS.search()
    for s,tb in results:
        section = t.bsearch_transcript_slice(tb.idx,s)
        print(tb.idx)
        for sub in section:
            print("\t", sub.content)