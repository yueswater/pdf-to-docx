import sys


def progress_bar(message):
    sys.stdout.write(f"\r{message}\n")
    sys.stdout.flush()
