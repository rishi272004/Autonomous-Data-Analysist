# backend/data_ingestion/schema_generator.py
from typing import Dict
import pandas as pd
from ..utils.logger import get_logger

logger = get_logger(__name__)

def generate_schema_from_df(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return a simple column->SQL type mapping (MySQL-ish).
    """
    mapping = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            mapping[col] = "BIGINT"
        elif pd.api.types.is_float_dtype(dtype):
            mapping[col] = "DOUBLE"
        elif pd.api.types.is_bool_dtype(dtype):
            mapping[col] = "BOOLEAN"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            mapping[col] = "DATETIME"
        else:
            # default to text with length heuristic
            max_len = int(df[col].astype(str).map(len).max() or 100)
            if max_len < 256:
                mapping[col] = f"VARCHAR({min(1024, max_len*2)})"
            else:
                mapping[col] = "TEXT"
    logger.debug(f"Generated schema mapping: {mapping}")
    return mapping
