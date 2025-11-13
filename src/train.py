import pickle
import pandas as pd
from fpgrowth_py import fpgrowth
import os, json, time

DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_PATH = "/mnt/model/model.pkl"

df = pd.read_csv(DATASET_PATH)
transactions = df.groupby("pid")["track_name"].apply(list).tolist()

freq, rules = fpgrowth(transactions, minSupRatio=0.01, minConf=0.2)

model = {"rules": rules, "timestamp": time.time()}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(model, f)
