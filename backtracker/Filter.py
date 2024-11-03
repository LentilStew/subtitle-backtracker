from datetime import datetime
from backtracker.VideoData import VideoData
from backtracker.WordInstances import WordInstances
from typing import Optional, List, Dict


class Filter:
    """Interface for all filters."""

    def run(self, instances: WordInstances) -> bool:
        raise NotImplementedError("Each filter must implement the run method.")


class FilterChain:
    """Applies a chain of filters to WordInstances."""

    def __init__(self, filters: Optional[List[Filter]] = None):
        self.filters = filters if filters is not None else []

    def add_filter(self, filter: Filter) -> None:
        """Add a filter to the chain."""
        self.filters.append(filter)

    def run(self, instances: "WordInstances") -> bool:
        """Run all filters in the chain; return False if any filter fails."""
        for filter in self.filters:
            if not filter.run(instances):
                return False
        return True


class DateFilter(Filter):
    """Filter to check if the date of a video falls within a specific range."""

    def __init__(
        self, idx_video_map: Dict[str, VideoData], start: datetime, end: datetime
    ):
        self.idx_video_map = idx_video_map
        self.start = start.timestamp()  # Convert to timestamp for comparison
        self.end = end.timestamp()

    def _parse_video_date(self, video_data: VideoData) -> Optional[float]:
        """Helper method to safely parse the date from video data."""
        try:
            return datetime.fromisoformat(
                video_data.video_info["date"].replace("Z", "+00:00")
            ).timestamp()
        except (KeyError, ValueError, TypeError):
            print("Failed parsing date ", video_data.video_info.get("date", "No date"))

            return None

    def run(self, idx: str) -> bool:
        """Run date filter on the given video index."""
        video_data = self.idx_video_map.get(idx)
        if video_data is None:
            return False

        video_date = self._parse_video_date(video_data)
        if video_date is None:
            return False

        return self.start <= video_date <= self.end
