import pandas as pd
from sqlalchemy import create_engine, text
import json, os, tempfile
from backend.utils.helpers import sanitize_dataframe_columns
from backend.utils.logger import get_logger
from backend.utils.config import DB_CONFIG

logger = get_logger(__name__)

CHUNK_SIZE = 10000

def create_engine_mysql():
    conn_str = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        conn.execute(text("SET SESSION sql_mode = ''"))
        conn.commit()
    return engine

def read_file(file_path, chunked=False):
    try:
        if file_path.endswith((".xls", ".xlsx")):
            if chunked:
                return pd.read_excel(file_path, engine="openpyxl", chunksize=CHUNK_SIZE)
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_path.endswith(".csv"):
            if chunked:
                return pd.read_csv(file_path, chunksize=CHUNK_SIZE)
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file type")
        if not chunked:
            df = sanitize_dataframe_columns(df)
        return df
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return pd.DataFrame() if not chunked else iter([])

def load_excel_to_mysql(files, table_name: str, chunked=False):
    engine = create_engine_mysql()
    first_chunk = True
    sample_df = pd.DataFrame()
    temp_files = []
    for file in files:
        suffix = os.path.splitext(file.filename)[1] if file.filename else '.csv'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        temp_files.append(tmp_path)
        df_iter = read_file(tmp_path, chunked=chunked)
        if chunked:
            for chunk in df_iter:
                if chunk.empty: continue
                chunk['file_source'] = file.filename
                chunk = sanitize_dataframe_columns(chunk)
                for col in chunk.columns:
                    if chunk[col].dtype == 'object' and chunk[col].apply(lambda x: isinstance(x, (list, dict))).any():
                        chunk[col] = chunk[col].apply(lambda x: json.dumps(x) if x is not None else None)
                chunk.to_sql(table_name, engine, if_exists='append' if not first_chunk else 'replace', index=False)
                first_chunk = False
                if sample_df.empty: sample_df = chunk.head(10)
        else:
            df = df_iter
            if df.empty: continue
            df['file_source'] = file.filename
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                    df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
            df.to_sql(table_name, engine, if_exists='append' if not first_chunk else 'replace', index=False)
            first_chunk = False
            if sample_df.empty: sample_df = df.head(10)
    # Clean up temp files
    for tmp_path in temp_files:
        try:
            os.unlink(tmp_path)
            logger.info(f"Cleaned up temp file: {tmp_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {tmp_path}: {e}")
    # Final columns
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1"))
        columns = list(result.keys())
    logger.info(f"Loaded data into table `{table_name}` successfully")
    engine.dispose()
    return table_name, columns, sample_df

def drop_table(table_name: str):
    engine = create_engine_mysql()
    with engine.connect() as conn:
        try:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()
            logger.info(f"Dropped table `{table_name}`")
        except Exception as e:
            logger.error(f"Failed to drop table `{table_name}`: {e}")
    engine.dispose()
