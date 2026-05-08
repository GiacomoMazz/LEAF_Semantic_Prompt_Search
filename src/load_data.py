# load_data.py
#
# Purpose:
# Load the raw LEAF PromptKaban dataset.
#
# Behavior:
# Reads dataset.json from the external data folder and stores it in a pandas DataFrame.

import pandas as pd

data = "../data/dataset.json"

df = pd.read_json(data)
