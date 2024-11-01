from typing import TypedDict, Optional, List
import json
import configparser
from types import MappingProxyType
config = configparser.ConfigParser()
config.read("./config.ini")



class VideoSubtitle(TypedDict):
    srtUrl: str
    type: str
    language: str


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
    subtitles: List[VideoSubtitle]
    commentsTurnedOff: bool
    comments: Optional[str]
    fromYTUrl: str
    isMonetized: Optional[bool]
    hashtags: List[str]
    formats: List[str]


class VideoData:
    def __init__(self, video_info: VideoDict, transcript_srt: str):
        self.video_info: VideoDict = video_info
        self.transcript_srt = transcript_srt

    @classmethod
    def create(cls, video_info: VideoDict):
        
        if not "subtitles" in video_info:
            return None
         
        for sub in video_info["subtitles"]:
            if "srt" in sub:
                return cls(video_info, sub["srt"])
        return None
    
video_data: MappingProxyType[str:VideoData] = None

def load_video_data():
    global video_data
    video_data_dict: dict[str:VideoData] = {}

    with open(config["videos-data"]["RAW_DATA"], "r") as f:
        data: list[dict] = json.load(f)
        for v in data:
            res = VideoData.create(v)
            if res is not None:
                video_data_dict[v["id"]] = res

    video_data = MappingProxyType(video_data_dict)