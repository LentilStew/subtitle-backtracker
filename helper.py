# youtube
import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi
import os
import pickle
import json
import configparser
from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
import marisa_trie
import srt

config = configparser.ConfigParser()
config.read("config.ini")


def save_videos_pickle():
    videos = list(
        scrapetube.get_channel(channel_url="https://www.youtube.com/@AtriocVODs")
    )

    with open(config["youtube"]["ALL_VIDEOS_TEST"], "wb") as fp:
        pickle.dump(videos, fp)


def load_save_videos_pickle():
    with open(config["youtube"]["ALL_VIDEOS_TEST"], "rb") as fp:
        videos = pickle.load(fp)
    return videos


def save_video_transcript_pickle(video_id):
    YouTubeTranscriptApi.get_transcript(video_id)  # TODO


def extract_important_info(video_data):
    # Extract relevant fields from the input JSON
    simplified_data = {
        "id": video_data.get("videoId"),
        "title": video_data.get("title", {}).get("runs", [{}])[0].get("text", ""),
        "description": video_data.get("descriptionSnippet", {})
        .get("runs", [{}])[0]
        .get("text", ""),
        "date": video_data.get("publishedTimeText", {}).get("simpleText", ""),
        "view_count": video_data.get("viewCountText", {}).get("simpleText", ""),
        "thumbnail": video_data.get("thumbnail", {}),
    }
    return simplified_data


def better_date(date: str):
    time_deltas = {
        "hour": timedelta(hours=1),
        "day": timedelta(days=1),
        "week": timedelta(weeks=1),
        "month": relativedelta(months=1),
        "year": relativedelta(years=1),
    }

    res = date.split(" ")

    assert len(res) == 3
    n = int(res[0])  # number of times time_type
    curr_time_type = res[1].replace("s", "")  # time type

    if curr_time_type not in time_deltas:
        raise ValueError("Invalid time type provided")

    # Calculate the date difference
    delta = time_deltas[curr_time_type] * n
    computed_date = datetime.now() - delta

    return computed_date.strftime("%Y-%m-%d %H:%M:%S")


# Create a file for each video, with all the data
def format_videos(videos, save_file=config["youtube"]["ALL_VIDEOS_FORMATTED"]):
    short_json_dict = {}

    for video in videos:
        short_json = extract_important_info(video)
        short_json["better_date"] = better_date(short_json["date"])
        short_json_dict[short_json["id"]] = short_json
        
    if save_file != None:
        with open(save_file, "w") as fp:
            json.dump(short_json_dict, fp)

    return short_json_dict
format_videos(load_save_videos_pickle())

def load_formated_videos():
    with open(config["youtube"]["ALL_VIDEOS_FORMATTED"], "r") as fp:
        video_metadata = json.load(fp)
    return video_metadata


def transcript_to_srt(transcripts_dict):
    transcript = []

    for idx, entry in enumerate(transcripts_dict):
        new_subtitle = srt.Subtitle(
            index=idx,
            start=timedelta(seconds=entry["start"]),
            end=timedelta(seconds=entry["start"] + entry["duration"]),
            content=entry["text"],
        )
        print(entry["text"])
        transcript.append(new_subtitle)

    return srt.compose(transcript)


def all_transcript_to_srt():
    for f in os.listdir(config["youtube"]["RAW_VIDEOS_FOLDER"]):
        file_path = os.path.join(config["youtube"]["RAW_VIDEOS_FOLDER"], f)
        new_file_path = os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], f)

        try:
            with open(file_path, "r") as fp:
                transcript_list = json.load(fp)

            with open(new_file_path, "w") as fp:
                fp.write(transcript_to_srt(transcript_list))
                print(f"writing transcript_to_srt {new_file_path}")

        except Exception as err:
            print(file_path, err)


def delete_empty():
    for f in os.listdir(config["youtube"]["RAW_VIDEOS_FOLDER"]):
        file_path = os.path.join(config["youtube"]["RAW_VIDEOS_FOLDER"], f)
        try:
            if os.stat(file_path).st_size == 0:
                print(f"removing, {file_path}")
                os.remove(file_path)

        except Exception as err:
            print(err)


# before running this, you should have config["youtube"]['TRANSCRITPS_FOLDER'] with all the transcripts as srts, with the name of the ids
def create_trie(
    output_file=None,
    transcripts_paths=[
        os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], file_name)
        for file_name in os.listdir(config["youtube"]["TRANSCRITPS_FOLDER"])
    ],
    save_index=True,
):
    print("Creating Trie ~~!\n")

    # marisa_trie calls struct.pack(fmt,*value) , since I want to store <11s, b"OOAOOAOOAOO", this would be split into 11 different tuples instead of 1
    class ByteConverter:
        def __init__(self, value):
            self.value = value
            self.once = True

        def __iter__(self):
            return self

        def __next__(self):
            if self.once == True:
                self.once = False
                return bytes(self.value, "utf-8")

            raise StopIteration

    if save_index:
        fmt = "<15s"
    else:
        fmt = "<11s"

    key_val_pairs = []
    for i, f in enumerate(transcripts_paths):
        print(f"({i+1}/{len(transcripts_paths)}) opening {f}")
        with open(f, encoding="utf-8") as fp:
            subs = srt.parse(fp)
            for sub in subs:
                for word in sub.content.split(" "):
                    if save_index:
                        new_val = (
                            word.lower(),
                            ByteConverter(
                                os.path.basename(f) + format(sub.index, "04X")
                            ),
                        )
                        key_val_pairs.append(new_val)

                    else:
                        key_val_pairs.append(
                            (word.lower(), ByteConverter(os.path.basename(f)))
                        )
    print("")
    print(f"key value pairs added {len(key_val_pairs)}")

    trie = marisa_trie.RecordTrie(fmt, key_val_pairs)

    if output_file is not None:
        print(f"saving to {output_file}")
        trie.save(output_file)

    return trie


def get_word_instances(word):  # old and incomplete
    trie = marisa_trie.RecordTrie("<11s")
    trie.load("transcripts.marisa")
    ids = trie.get(word)

    for idx in ids:
        instance = idx[0].decode("utf-8")

        subs_path = os.path.join(config["youtube"]["TRANSCRITPS_FOLDER"], instance)

        with open(subs_path, encoding="utf-8") as fp:
            subs = srt.parse(fp)
            for sub in subs:
                if word in sub.content.split(" "):
                    pass


# If returns true fp.readline()should return timestamps, and fp.readling() should return content
def _binary_search_index(fp, search, low, high):
    def find_next_index(fp):
        index = 0
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

    # edge case EOF
    # last item


# generator returns srt.Subtitle
def bsearch_get_word_instances(
    word, trie, transcripts_folder=config["youtube"]["TRANSCRITPS_FOLDER"]
):
    ids = trie.get(word)
    for idx in ids:
        instance = idx[0].decode("utf-8")
        subs_path = os.path.join(transcripts_folder, instance[:-4])
        index = int(instance[-4:], 16)
        yield binary_search_index(subs_path, index)


def yt_link_format(idx,timestamp):
    return f"https://youtu.be/{idx}?t={int(timestamp.total_seconds())}" 