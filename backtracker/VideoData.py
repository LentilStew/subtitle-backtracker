from typing import TypedDict, Optional, List
import json
import configparser
from types import MappingProxyType
config = configparser.ConfigParser()
config.read("./config.ini")
import os


class VideoDict(TypedDict):
    title: str
    type: str
    id: str
    url: str
    thumbnailUrl: str
    viewCount: int
    date: str
    likes: int
    location: Optional[str]
    channelName: str
    channelUrl: str
    channelId: str
    channelDescription: Optional[str]
    channelJoinedDate: str
    channelDescriptionLinks: List[str]
    channelLocation: Optional[str]
    channelAvatarUrl: str
    channelBannerUrl: str
    channelTotalVideos: int
    channelTotalViews: int
    numberOfSubscribers: int
    isChannelVerified: bool
    inputChannelUrl: str
    isAgeRestricted: bool
    duration: str
    commentsCount: int
    text: str
    descriptionLinks: List[str]
    commentsTurnedOff: bool
    comments: Optional[str]
    fromYTUrl: str
    isMonetized: Optional[bool]
    hashtags: List[str]
    formats: List[str]

class VideoData:
    def __init__(self, video_info: VideoDict, transcript_path: str):
        self.video_info: VideoDict = video_info
        self.transcript_path = transcript_path
    
video_data: MappingProxyType[str:VideoData] = None

def load_video_data():
    global video_data
    video_data_dict: dict[str:VideoData] = {}
    with open(config["data"]["RAW_DATA"], "r") as f:
        data: list[dict] = json.load(f)
        for v in data:
            path = os.path.join(
                config["data"]["TRANSCRITPS_FOLDER"],v["id"]
            )

            if os.path.isfile(path):
                video_data_dict[v["id"]] = VideoData(v,path)

    video_data = MappingProxyType(video_data_dict)