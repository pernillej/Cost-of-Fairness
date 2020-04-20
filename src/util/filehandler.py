from datetime import datetime
from pathlib import Path
import json
import os

# This is relative to the main.py file in each experiment folder. Update when running from different file
RESULT_FOLDER = os.path.abspath("results")


def write_result_to_file(result, file_prefix):
    now = datetime.now()
    timestamp_string = now.strftime("%d-%m-%Y_%H-%M")
    filepath = Path(RESULT_FOLDER + "/" + file_prefix + "_" + timestamp_string + ".txt")
    with open(filepath, 'w') as outfile:
        json.dump(result, outfile)


def read_result_from_file(filename):
    filepath = Path(RESULT_FOLDER + "/" + filename)
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data
