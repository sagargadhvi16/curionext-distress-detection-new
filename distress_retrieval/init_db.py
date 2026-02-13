import sqlite3
import pandas as pd

# Load CSV
df = pd.read_csv("data.csv")

# Create SQLite DB
conn = sqlite3.connect("distress.db")

df.to_sql(
    "distress_records",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("SQLite database created: distress.db")
