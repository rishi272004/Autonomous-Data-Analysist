# backend/query_engine/insights.py
import pandas as pd
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def basic_describe(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "summary": df.describe(include='all').to_dict('split')
    }

def detect_time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        sample = df[col].dropna().head(10)
        if len(sample) > 0:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().sum() / len(sample) > 0.7:
                return col
    return ""