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
import re
from pytube import YouTube
from pytube import Playlist

text_to_words = lambda text: [word for word in text.lower().split(" ")]

# formats word for search in trie
format_word_trie = lambda word: word.lower()

config = configparser.ConfigParser()
config.read("config.ini")


def save_playlist_pickle(file, playlist_id):
    videos = list(scrapetube.scrapetube.get_playlist(playlist_id=playlist_id))

    with open(file, "wb") as fp:
        pickle.dump(videos, fp)


def load_playlist_pickle(file):
    with open(file, "rb") as fp:
        videos = pickle.load(fp)
    return videos


def save_videos_pickle():
    videos = list(
        scrapetube.get_channel(channel_url="https://www.youtube.com/@AtriocVODs")
    )

    with open(config["data"]["ALL_VIDEOS_TEST"], "wb") as fp:
        pickle.dump(videos, fp)


def load_save_videos_pickle():
    with open(config["data"]["ALL_VIDEOS_TEST"], "rb") as fp:
        videos = pickle.load(fp)
    return videos


def get_transcript(video_id):
    return YouTubeTranscriptApi.get_transcript(video_id)  # TODO


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
def format_videos(videos, save_file=config["data"]["ALL_VIDEOS_FORMATTED"]):
    short_json_dict = {}

    for video in videos:
        short_json = extract_important_info(video)
        short_json["better_date"] = better_date(short_json["date"])
        short_json_dict[short_json["id"]] = short_json

    if save_file != None:
        with open(save_file, "w") as fp:
            json.dump(short_json_dict, fp)

    return short_json_dict


# format_videos(load_save_videos_pickle())


def load_formated_videos():
    with open(config["data"]["ALL_VIDEOS_FORMATTED"], "r") as fp:
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
    for f in os.listdir(config["data"]["RAW_VIDEOS_FOLDER"]):
        file_path = os.path.join(config["data"]["RAW_VIDEOS_FOLDER"], f)
        new_file_path = os.path.join(config["data"]["TRANSCRITPS_FOLDER"], f)

        try:
            with open(file_path, "r") as fp:
                transcript_list = json.load(fp)

            with open(new_file_path, "w") as fp:
                fp.write(transcript_to_srt(transcript_list))
                print(f"writing transcript_to_srt {new_file_path}")

        except Exception as err:
            print(file_path, err)


def delete_empty():
    for f in os.listdir(config["data"]["RAW_VIDEOS_FOLDER"]):
        file_path = os.path.join(config["data"]["RAW_VIDEOS_FOLDER"], f)
        try:
            if os.stat(file_path).st_size == 0:
                print(f"removing, {file_path}")
                os.remove(file_path)

        except Exception as err:
            print(err)


def get_word_instances(word):  # old and incomplete
    trie = marisa_trie.RecordTrie("<11s")
    trie.load("transcripts.marisa")
    ids = trie.get(word)

    for idx in ids:
        instance = idx[0].decode("utf-8")

        subs_path = os.path.join(config["data"]["TRANSCRITPS_FOLDER"], instance)

        with open(subs_path, encoding="utf-8") as fp:
            subs = srt.parse(fp)
            for sub in subs:
                if word in sub.content.split(" "):
                    pass

def yt_link_format(idx, timestamp):
    if isinstance(idx, bytes):
        idx = idx.decode("utf-8")
    return f"https://youtu.be/{idx}?t={int(timestamp.total_seconds())}"


def yt_link_format_seconds(idx, timestamp):
    if isinstance(idx, bytes):
        idx = idx.decode("utf-8")
    return f"https://youtu.be/{idx}?t={int(timestamp)}"
