from datetime import datetime


class TimeStamp:
    @staticmethod
    def now(fmt: str = "%Y%m%d_%H%M%S") -> str:
        return datetime.now().strftime(format=fmt)

    @staticmethod
    def iso() -> str:
        return datetime.now().isoformat()
