import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv("data.csv")

start_time = datetime(2026, 2, 1, 10, 0, 0)

timestamps = [
    (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
    for i in range(len(df))
]

df.insert(1, "timestamp", timestamps)
df.to_csv("data.csv", index=False)

print("Timestamp added.")
