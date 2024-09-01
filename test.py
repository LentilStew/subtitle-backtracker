from TrieTranscript import TrieTranscript, binary_search_index
import configparser
import os
import srt
from typing import Union
from datetime import datetime
from helper import yt_link_format
import timeit

config = configparser.ConfigParser()
config.read("config.ini")
all_transcripts = [
    os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], idx)
    for idx in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
]
import marisa_trie
import struct
def create_trie( transcripts_paths, trie_path=None):

    words_indexes = {}
    for i, f in enumerate(transcripts_paths):
        print(f"({i+1}/{len(transcripts_paths)}) opening {f}")

        with open(f, encoding="utf-8") as fp:
            subs = srt.parse(fp)

            for sub in subs:
                for word in sub.content.split(" "):
                    words_indexes[word] = words_indexes.get(word,(os.path.basename(f),[]))
                    words_indexes[word][1].append(sub.index)
    
    print(f"key value pairs added {len(words_indexes.keys())}")
    key_val_pairs = []
    for word,val in words_indexes.items():
        idx,instances = val
        numbers_format = f"{len(instances)}H" 
        string_format = f"11s"    
        packed_data = struct.pack(f"{string_format}{numbers_format}", idx.encode(),*instances)
        
        key_val_pairs.append(
            (word,packed_data)
        )

    trie = marisa_trie.BytesTrie(key_val_pairs)

    print(f"saving to {trie_path}")
    trie.save(trie_path)

    return trie

#create_trie( all_transcripts, trie_path="./temp2")
def get_indexes_list(trie, words:Union[str,list[str]], mutually_inclusive:bool=True):# returns paths to all instances of the word mentioned, and the index in the timestamp

    if isinstance(words,str):
        words = [words]

    if mutually_inclusive:
        res = set(idx[:11] for idx in trie.get(words[0]))
        for word in words[1:]:
            res.intersection_update(set(idx[:11] for idx in trie.get(word)))
        print(res)
        return res
    
list_trie = marisa_trie.BytesTrie()
list_trie.load("./temp2")
def list_trie_test():
    res = get_indexes_list(trie.trie,["for","the","minions","tomorrow"])
    print(len(res))

trie = TrieTranscript(all_transcripts, "11s", config["youtube"]["MARISA_TRIE"])
def get_indexes(trie, words:Union[str,list[str]], mutually_inclusive:bool=True):# returns paths to all instances of the word mentioned, and the index in the timestamp

    if isinstance(words,str):
        words = [words]

    if mutually_inclusive:
        res = set(idx[0] for idx in trie.get(words[0]))
        for word in words[1:]:
            res.intersection_update(set(idx[0] for idx in trie.get(word)))
        return res
# Function 1: Using get_indexes from trie
def run_get_indexes():
    res = get_indexes(trie.trie,["for","the","minions","tomorrow"])
    print(len(res))

trie_index = TrieTranscript(all_transcripts, "11sL", config["youtube"]["MARISA_TRIE_INDEX"])
# Function 2: Using fast_mutually_inclusive_get_indexes from trie_index
def run_fast_mutually_inclusive_get_indexes():
    res =trie_index.fast_mutually_inclusive_get_indexes(["for","the","minions","tomorrow"])
    
    print(len(res))
    
    return res

# Measure the execution time of each function
print("time_get_indexes = timeit.timeit(run_get_indexes, number=1)")
time_get_indexes = timeit.timeit(run_get_indexes, number=1)
print("time_fast_get_indexes = timeit.timeit(run_fast_mutually_inclusive_get_indexes, number=1)")
time_fast_get_indexes = timeit.timeit(run_fast_mutually_inclusive_get_indexes, number=1)
print("time_fast_get_indexes2 = timeit.timeit(run_fast_mutually_inclusive_get_indexes, number=1)2")
trie_list_time = timeit.timeit(list_trie_test, number=1)
print("trie_list")

print(f"Time taken by get_indexes: {time_get_indexes:.6f} seconds")
print(f"Time taken by fast_mutually_inclusive_get_indexes: {time_fast_get_indexes:.6f} seconds")
print(f"Time taken by trie_list: {trie_list_time:.6f} seconds")



"""
712
75 {b'zs0y2kJs65o', b'f3yU2CeteOw', b'rzhPYdAX-nU', b'Ci8zrC8PTzk', b'XPTn6uEZveY', b'3j3ueKfQ-8g', b'9HcZcMaXCXU', b'rxdzysIP_uY', b'y4KacusBxnU', b'DJNTRewm_JM', b'e4VrN3OW8gQ', b'OTqC8NRktPM', b'zbDMMOtOf7g', b'RxpcdlxDeQA', b'MPdQzjja8S4', b'bDhAAoUM2JI', b'ZCTcpQXQnyg', b'GUKz8EeNGGU', b'8H0SrwCYCUM', b'emke62Upfb0', b'aNRwQhv84wc', b'wNK294yvUXQ', b'DcmkRwfhJKw', b'3wO4BMCOR5M', b'jvtDL_O3xrw', b'SqBa6K3pWGo', b'ZnVzI05c8bM', b'Mc7zdpu7dnc', b's-jYHUQgPbM', b'RatF6TDi5rc', b'RZb0KKGqb0A', b'5VAHj7JPuog', b'f_YGAVlSOYo', b'tw75aPra1no', b'Sdyx-cGCN-4', b'P3DJVLP6c9M', b'WuN-uas5A_I', b'k-0U05AdTPo', b'OK7VJjeHK0o', b'msItKMSBj4U', b'U6ByFIDk-Js', b'XT9cWKADSKo', b'lreQRJu-kYI', b'U97djgkAri4', b'a5StuTuZFNc', b'aCayq6onu7w', b'0t0gfouhYlY', b'tObrTYi2bLA', b'ZfI1B9JCjDU', b'OyEMHXEpmzU', b'o7nphpuTRcU', b'ey9JY9f_gK8', b'uhq-sEnQ5eg', b'hrjrCoyH_d4', b'eZzsc6hnyv0', b'v3PJkjC-CiE', b'8BKMNKnDE6M', b'ZqVYWchz9Es', b'qnxyylHXDlw', b'6ujWAUtidF0', b'r-apcvSS_v4', b'2BmQvpkAJB0', b'vWNbgSCZSyc', b'4ambLcTc4FQ', b'V6oY83jqfCg', b'dEc4WPYIRZY', b'wZLgo_83TNc', b'9Ru-kv5yG7I', b'cFRoIh_tPg8', b'sUPzXLCeCV0', b'AsFZ4bMTAxg', b'fDrs2maEYG4', b'l36CMfZ47mM', b'nf2adJBqCDs', b'fXGnslG0xLA'}
time_fast_get_indexes = timeit.timeit(run_fast_mutually_inclusive_get_indexes, number=1)
103179
75 {b'zs0y2kJs65o', b'f3yU2CeteOw', b'Ci8zrC8PTzk', b'rzhPYdAX-nU', b'XPTn6uEZveY', b'3j3ueKfQ-8g', b'9HcZcMaXCXU', b'rxdzysIP_uY', b'y4KacusBxnU', b'DJNTRewm_JM', b'e4VrN3OW8gQ', b'OTqC8NRktPM', b'zbDMMOtOf7g', b'RxpcdlxDeQA', b'MPdQzjja8S4', b'bDhAAoUM2JI', b'ZCTcpQXQnyg', b'GUKz8EeNGGU', b'8H0SrwCYCUM', b'emke62Upfb0', b'aNRwQhv84wc', b'3wO4BMCOR5M', b'SqBa6K3pWGo', b'DcmkRwfhJKw', b'wNK294yvUXQ', b'jvtDL_O3xrw', b'ZnVzI05c8bM', b'Mc7zdpu7dnc', b's-jYHUQgPbM', b'f_YGAVlSOYo', b'RatF6TDi5rc', b'RZb0KKGqb0A', b'5VAHj7JPuog', b'tw75aPra1no', b'Sdyx-cGCN-4', b'P3DJVLP6c9M', b'WuN-uas5A_I', b'k-0U05AdTPo', b'OK7VJjeHK0o', b'XT9cWKADSKo', b'U6ByFIDk-Js', b'lreQRJu-kYI', b'msItKMSBj4U', b'U97djgkAri4', b'a5StuTuZFNc', b'tObrTYi2bLA', b'aCayq6onu7w', b'0t0gfouhYlY', b'ZfI1B9JCjDU', b'OyEMHXEpmzU', b'ey9JY9f_gK8', b'o7nphpuTRcU', b'hrjrCoyH_d4', b'uhq-sEnQ5eg', b'eZzsc6hnyv0', b'v3PJkjC-CiE', b'8BKMNKnDE6M', b'ZqVYWchz9Es', b'qnxyylHXDlw', b'6ujWAUtidF0', b'r-apcvSS_v4', b'2BmQvpkAJB0', b'vWNbgSCZSyc', b'4ambLcTc4FQ', b'V6oY83jqfCg', b'dEc4WPYIRZY', b'wZLgo_83TNc', b'9Ru-kv5yG7I', b'cFRoIh_tPg8', b'sUPzXLCeCV0', b'AsFZ4bMTAxg', b'fDrs2maEYG4', b'l36CMfZ47mM', b'nf2adJBqCDs', b'fXGnslG0xLA'}
Time taken by get_indexes: 0.001535 seconds
Time taken by fast_mutually_inclusive_get_indexes: 0.267985 seconds
"""








""" Filter by time example
from TrieTranscript import TrieTranscript, binary_search_index
import configparser
import os
import srt
import json
from datetime import datetime
from helper import yt_link_format
config = configparser.ConfigParser()
config.read("config.ini")



with open(config["youtube"]["ALL_VIDEOS_FORMATTED"], "r") as fp:
    video_metadata = json.load(fp)



better_date_format = "%Y-%m-%d %H:%M:%S"
start_date_str = "2024-01-01 00:00:00"
end_date_str = "2025-05-01 00:00:00"
is_between_date = {}


for _,video_data in video_metadata.items():
    better_date_str = video_data["better_date"]
    better_date = datetime.strptime(better_date_str, better_date_format)
    start_date = datetime.strptime(start_date_str, better_date_format)
    end_date = datetime.strptime(end_date_str, better_date_format)
    if start_date <= better_date <= end_date:
        is_between_date[video_data["id"]] = True
    else:
        is_between_date[video_data["id"]] = False

def filter(idx,is_between_date):
    return not is_between_date[idx]

trie = TrieTranscript(all_transcripts, "<15s", config["youtube"]["MARISA_TRIE_INDEX"])

sub: srt.Subtitle
for sub,idx in trie.bsearch_transcript("hot",filter=filter,is_between_date=is_between_date):
    print(yt_link_format (video_metadata[idx]["id"],sub.start))
"""



""" Create transcript
config = configparser.ConfigParser()
config.read("config.ini")

all_transcripts = [
    os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], idx)
    for idx in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
]

TrieTranscript.create_trie(all_transcripts,config["youtube"]["MARISA_TRIE_INDEX"],True)
"""