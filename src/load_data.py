# load_data.py
#
# Purpose:
# Load the raw LEAF PromptKaban dataset from the shared data folder.
# The loaded DataFrame is used by the EDA and preprocessing scripts.

import pandas as pd

data = "../data/dataset.json"

df = pd.read_json(data)
