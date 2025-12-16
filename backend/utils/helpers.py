# backend/utils/helpers.py
import pandas as pd
import re
from contextlib import contextmanager
import warnings
import json
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def df_head_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(n)

def sanitize_column_name(col: str) -> str:
    col = str(col).strip()
    col = re.sub(r'[^0-9a-zA-Z_]', '_', col)
    col = re.sub(r'_+', '_', col)
    if re.match(r'^[0-9]', col):
        col = f"col_{col}"
    return col.lower()

def sanitize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [sanitize_column_name(c) for c in df.columns]
    return df

@contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        yield

def safe_float_convert(series: pd.Series, col_name: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception as e:
        logger.warning(f"Failed to convert {col_name} to float: {e}")
        return series

def batch_json_normalize(raw_dfs: list) -> pd.DataFrame:
    normalized_dfs = []
    for raw_df in raw_dfs:
        try:
            if not raw_df.empty:
                json_series = raw_df['data_json'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                norm_df = pd.json_normalize(json_series)
                normalized_dfs.append(norm_df)
        except Exception as e:
            logger.warning(f"JSON normalization failed for batch: {e}")
            continue
    return pd.concat(normalized_dfs, ignore_index=True) if normalized_dfs else pd.DataFrame()