import os
from json_compare import JSONComparator

print(os.getcwd())

latest_json = "./batch/output/json/20250721_141014/RO0015995.json"
prev_json = "./batch/output/json/20250721_134939/RO0015995.json"

comparator = JSONComparator(left_file_path=prev_json, right_file_path=latest_json)
print(comparator.diff_log.log)
