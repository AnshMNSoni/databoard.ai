import pandas as pd

def build_dataset_summary(df: pd.DataFrame):
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "numeric_columns": [],
        "categorical_columns": []
    }

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe().to_dict()
            summary["numeric_columns"].append({
                "name": col,
                "mean": round(stats.get("mean", 0), 2),
                "min": round(stats.get("min", 0), 2),
                "max": round(stats.get("max", 0), 2),
                "std": round(stats.get("std", 0), 2)
            })
        else:
            top_values = df[col].value_counts().head(1)
            summary["categorical_columns"].append({
                "name": col,
                "unique_values": int(df[col].nunique()),
                "top_value": str(top_values.index[0]) if not top_values.empty else "N/A",
                "top_count": int(top_values.iloc[0]) if not top_values.empty else 0
            })

    return summary
