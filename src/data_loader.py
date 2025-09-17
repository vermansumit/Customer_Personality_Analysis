# src/data_loader.py
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "marketing_campaign.csv"

def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path, sep=None, engine="python")  # auto-detect sep
    return df

if __name__ == "__main__":
    df = load_data()
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print(df.head(3).to_dict(orient='records'))