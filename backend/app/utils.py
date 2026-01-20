import pandas as pd

def load_dataset(file_path: str):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format")


def detect_schema(df: pd.DataFrame):
    return [
        {"name": col, "type": str(df[col].dtype)}
        for col in df.columns
    ]
