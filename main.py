#youtube
import scrapetube
from youtube_transcript_api import YouTubeTranscriptApi
import os.path
import pickle
import json
import configparser
import time
config = configparser.ConfigParser()
config.read("config.ini")



if not config.getboolean("run","DEBUG"):
    videos = list(scrapetube.get_channel(channel_url="https://www.youtube.com/@AtriocVODs"))
else:
    with open(config["youtube"]["CHANNEL_CACHE"],"rb") as fp:
        videos = pickle.load(fp)

#videos_transcripts = [[YouTubeTranscriptApi.get_transcript(video["videoId"]),video] for video in videos]
for video in videos:
    print(video["videoId"])
    if os.path.isfile(f"./videos/{video['videoId']}"):
        continue
    
    with open(f"./videos/{video['videoId']}","w") as fp:
        try:
            res = YouTubeTranscriptApi.get_transcript(video["videoId"])
            if(res):
                json.dump(res,fp)
        except:
            print(res)
            time.sleep(1)
            
#with open("videos",w) as fp:
#    for video in videos:
#        fp.write(video["videoId"], YouTubeTranscriptApi.get_transcript(video["videoId"]))


#if not config.getboolean("run","DEBUG"):
#    videos_transcripts = [[YouTubeTranscriptApi.get_transcript(video["videoId"]),video] for video in videos]
#else:
#    with open(config["youtube"]["VIDEO_CACHE"],"rb") as fp:
#        videos = pickle.load(fp)

