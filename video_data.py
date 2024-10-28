from typing import TypedDict, Optional, List
import json
import configparser
from types import MappingProxyType
config = configparser.ConfigParser()
config.read("./config.ini")
import io


class VideoSubtitle(TypedDict):
    srtUrl: str
    type: str
    language: str

class VideoInfo(TypedDict):
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

class Video:
    def __init__(self, video_info: VideoInfo, subtitles_file: io.TextIOWrapper):
        self.video_info: VideoInfo = video_info
        self.subtitles_file: io.TextIOWrapper = subtitles_file

    @classmethod
    def create(cls, video_info: VideoInfo):
        for sub in video_info["subtitles"]:
            if "srt" in sub["type"]:
                return cls(video_info, sub["str"])
        return None

_db: dict[str:Video]
db:MappingProxyType[str:Video]

with open(config["videos-data"]["RAW_DATA"], "r") as f:
    data: list[dict] = json.load(f)
    for v in data:
        res = Video.create(v)
        if res is not None:
          _db[v["id"]] = res

db = MappingProxyType(_db)