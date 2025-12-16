# backend/visualization/dashboard.py
import pandas as pd
from typing import Tuple, Optional
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def recommend_chart(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    time_col = next((col for col in df.columns if 'date' in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col])), None)
    if len(numeric_cols) >= 2 and time_col:
        return ("line", time_col)
    elif categorical_cols and numeric_cols:
        return ("bar", categorical_cols[0])
    elif len(numeric_cols) >= 1:
        return ("line", df.index.name or numeric_cols[0])
    else:
        return ("table", None)

logger.info("Dashboard utils loaded")