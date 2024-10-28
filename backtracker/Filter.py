from datetime import datetime


class DateFilter:
    def __init__(
        self, idx_date_map: dict[str, datetime], start: datetime, end: datetime
    ):
        self.idx_date_map: dict[str, datetime] = idx_date_map
        self.start: datetime = start
        self.end: datetime = end

    def run(self, idx):
        return self.start < self.idx_date_map[idx] and self.end > self.idx_date_map[idx]
