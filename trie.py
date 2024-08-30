import marisa_trie
import configparser
import os
from datetime import timedelta
import time
import srt
import codecs
import json
config = configparser.ConfigParser()
config.read("config.ini")

def transcript_to_srt(transcripts_dict):
    transcript = []

    for idx, entry in enumerate(transcripts_dict):
        new_subtitle= srt.Subtitle(
            index=idx,
            start=timedelta(seconds=entry['start']),
            end=timedelta(seconds=entry['start']+ entry['duration']),
            content=entry["text"]
        )
        print(entry["text"])
        transcript.append(new_subtitle)
        
    return srt.compose(transcript)


def all_transcript_to_srt():
    for f in os.listdir(config["youtube"]['RAW_VIDEOS_FOLDER']):
        file_path = os.path.join(config["youtube"]["RAW_VIDEOS_FOLDER"],f)
        new_file_path = os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"],f)
        
        try:
            with open( file_path,"r") as fp:
                transcript_list = json.load(fp)
            
            with open(new_file_path,"w") as fp:
                fp.write(transcript_to_srt(transcript_list))
                print(f"writing transcript_to_srt {new_file_path}")  

        except Exception as err:
            print(file_path,err)

def delete_empty():
    for f in os.listdir(config["youtube"]['RAW_VIDEOS_FOLDER']):
        file_path = os.path.join(config["youtube"]["RAW_VIDEOS_FOLDER"],f)
        try:
            if (os.stat(file_path).st_size == 0):
                print(f"removing, {file_path}")
                os.remove(file_path)

        except Exception as err:
            print(err)

#before running this, you should have config["youtube"]['TRANSCRITPS_FOLDER'] with all the transcripts as srts, with the name of the ids
def create_trie(
    output_file=None,
    transcripts_paths=[os.path.join(config["youtube"]['TRANSCRITPS_FOLDER'],file_name) for file_name in os.listdir(config["youtube"]['TRANSCRITPS_FOLDER'])],
    save_index=True

):
    print("Creating Trie ~~!\n")
    #marisa_trie calls struct.pack(fmt,*value) , since I want to store <11s, b"OOAOOAOOAOO", this would be split into 11 different tuples instead of 1
    class ByteConverter:
        def __init__(self, value):
            self.value = value
            self.once = True
        def __iter__(self):
            return self

        def __next__(self):
            if self.once == True:
                self.once = False
                return bytes(self.value,'utf-8')
            
            raise StopIteration

    if save_index:
        fmt = "<15s"
    else:
        fmt = "<11s"

    key_val_pairs = []
    for i,f in enumerate(transcripts_paths):
        print(f"({i+1}/{len(transcripts_paths)}) opening {f}")
        with open(f, encoding='utf-8') as fp:
            subs = srt.parse(fp)
            for sub in subs:
                for word in sub.content.split(" "):
                    if save_index:
                        new_val = (word.lower(),ByteConverter(os.path.basename(f) + format(sub.index, '04X') ))
                        key_val_pairs.append(new_val)
    
                    else:
                        key_val_pairs.append((word.lower(),ByteConverter(os.path.basename(f))))
    print("")
    print(f"key value pairs added {len(key_val_pairs)}")
    
    trie = marisa_trie.RecordTrie(fmt, key_val_pairs)

    if output_file is not None:
        print(f"saving to {output_file}")
        trie.save(output_file)

    return trie

def get_word_instances(word):#old and incomplete
    trie = marisa_trie.RecordTrie("<11s")
    trie.load("transcripts.marisa")
    ids = trie.get(word)

    for idx in ids:
        instance = idx[0].decode("utf-8")

        subs_path = os.path.join(config["youtube"]['TRANSCRITPS_FOLDER'],instance)

        with open(subs_path, encoding='utf-8') as fp:
            subs = srt.parse(fp)
            for sub in subs:
                if word in sub.content.split(" "):
                    pass

#If returns true fp.readline()should return timestamps, and fp.readling() should return content
def _binary_search_index(fp,search,low,high):
    def find_next_index(fp):
        index = 0
        chars = [b' ',b' ']

        while True:
            chars[0] = chars[1]#shift
            chars[1] = fp.read(1)
            if chars[0] == b'\n' and chars[1] == b'\n':
                break

    if high >= low:
        mid = low + (high - low)//2
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

def binary_search_index(file,index):

    with open(file,'rb') as fp:
        fp.seek(0, os.SEEK_END)
        file_size = fp.tell()
        
        res = _binary_search_index(fp,index,0,file_size-1)
        if res == False:
            print(f"Subtitle not found for index {index}")
        start,end = fp.readline().split(b' --> ')
        content = fp.readline()

        return srt.Subtitle(
                index=index,
                start = srt.srt_timestamp_to_timedelta(start.decode("utf-8")),
                end=srt.srt_timestamp_to_timedelta(end.decode("utf-8")),
                content=content
            )
        
    
        #edge case EOF
        #last item
    

#generator returns srt.Subtitle 
def bsearch_get_word_instances(word, trie,transcripts_folder=config["youtube"]['TRANSCRITPS_FOLDER']):
    ids = trie.get(word)
    for idx in ids:
        instance = idx[0].decode("utf-8")
        subs_path = os.path.join(transcripts_folder,instance[:-4])
        index = int(instance[-4:],16)
        yield binary_search_index(subs_path,index)



#create_trie(config["youtube"]["MARISA_TRIE_INDEX"])

trie = marisa_trie.RecordTrie(config["youtube"]["MARISA_TRIE_INDEX_FMT"])
trie.load(config["youtube"]["MARISA_TRIE_INDEX"])


for sub in bsearch_get_word_instances("album",trie):
    print(sub)
"""
import struct
res = marisa_trie.RecordTrie(">11sl",[("album",(*ByteConverter("ASDGASDASDA"),500))])
print(res["album"])


get_word_instances("album")
real	0m10.417s
user	0m10.329s
sys	0m0.064s
bsearch_get_word_instances("album")
real	0m0.184s
user	0m0.136s
sys	0m0.047s


binary_search_index(
    os.path.join(config["youtube"]['TRANSCRITPS_FOLDER'],os.listdir(config["youtube"]['TRANSCRITPS_FOLDER'])[0]),
    500
)







"""