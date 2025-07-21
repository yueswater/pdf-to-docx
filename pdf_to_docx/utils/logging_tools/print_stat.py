import time

from pdf_to_docx.utils.time_tools.convert_hhmmss import convert_hhmmss


def print_stat(start_time: float, total: int, success: int):
    hh, mm, ss = convert_hhmmss(time.time() - start_time)
    print("=" * 50)
    print(f"Time: {hh} hour(s) {mm} minute(s) {ss} second(s)")
    print(f"Total files: {total}")
    print(f"Successfully processed: {success}")
    print(f"Failure rate: {(total - success) / total:.2%}")
