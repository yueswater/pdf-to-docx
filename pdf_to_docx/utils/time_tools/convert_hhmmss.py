def convert_hhmmss(time: float) -> tuple:
    SECONDS_MINUTE = 60
    SECONDS_HOUR = 60 * SECONDS_MINUTE
    hh, rem_s = divmod(time, SECONDS_HOUR)
    mm, ss = divmod(rem_s, SECONDS_MINUTE)
    return tuple(map(int, (hh, mm, ss)))
