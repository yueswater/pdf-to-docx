import builtins
import os

original_open = builtins.open


def utf8_open(file, mode="r", *args, **kwargs):
    if (
        isinstance(file, (str, os.PathLike))
        and str(file).endswith(".json")
        and "b" not in mode
    ):
        kwargs.setdefault("encoding", "utf-8")
    return original_open(file, mode, *args, **kwargs)


builtins.open = utf8_open
