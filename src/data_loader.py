import json
from config import RAW_DATA_PATH

def load_raw_data(path=RAW_DATA_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as data:
        return json.load(data)