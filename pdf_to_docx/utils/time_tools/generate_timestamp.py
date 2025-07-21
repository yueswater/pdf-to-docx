import datetime


def generate_timestamp() -> str:
    current_time = datetime.datetime.now()
    date_str = current_time.strftime("%Y%m%d%H%M%S")
    return date_str
