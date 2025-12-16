# backend/query_engine/executor.py (Fixed Matplotlib backend and warnings)
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Fix: Non-interactive backend for server
import matplotlib.pyplot as plt
import io
from typing import Optional, List, Tuple
from sqlalchemy import create_engine, text
from backend.llm.ollama_client import query_gemini
from backend.utils.logger import get_logger
from backend.utils.config import DB_CONFIG
from backend.visualization.dashboard import recommend_chart
from backend.query_engine.insights import basic_describe
from backend.analysis.forecast import forecast_time_series, forecast_chart
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

logger = get_logger(__name__)

def execute_query_mysql(sql: str) -> pd.DataFrame:
    engine = None
    try:
        conn_str = f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            conn.execute(text("SET SESSION sql_mode = ''"))
            conn.commit()
            df = pd.read_sql(sql, conn)

        logger.info(f"Executed SQL, returned {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"SQL exec failed: {e}. Bad SQL snippet: {sql[:200]}...")
        print(sql)
        error_df = pd.DataFrame({"Error": [f"Query failed: {str(e)}. SQL: {sql[:150]}..."]})
        return error_df
    finally:
        if engine:
            engine.dispose()

def prepare_query(sql: str, uploaded_files: list) -> str:
    if "..." in sql:
        file_names = "', '".join([f.filename for f in uploaded_files])
        sql = sql.replace("...", file_names)
    return sql

def generate_visualization(df: pd.DataFrame, chart_type: str = "auto", x: Optional[str] = None, y: Optional[str] = None) -> io.BytesIO:
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))

    rec_type, rec_col = recommend_chart(df)
    if chart_type == "auto":
        chart_type = rec_type
    if x is None:
        x = rec_col or (df.columns[0] if len(df.columns) > 0 else None)
    
    # Ensure x is a single column name, not a list
    if isinstance(x, list) and len(x) > 0:
        x = x[0]

    try:
        if chart_type == "bar" and x and len(df.columns) > 1:
            y = y or df.select_dtypes(include='number').columns[0]
            ax.bar(df[x].astype(str), df[y], color='skyblue', alpha=0.7)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} by {x}")
        elif chart_type == "line" and x and len(df.columns) > 1:
            y = y or df.select_dtypes(include='number').columns[0]
            ax.plot(df[x].astype(str), df[y], marker='o', color='green', linewidth=2)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_title(f"{y} over {x}")
        else:
            if len(df.select_dtypes(include='number').columns) > 0:
                df.hist(ax=ax, color='lightblue')
            else:
                ax.scatter(range(len(df)), range(len(df)), color='red')
            ax.set_title("Data Distribution")
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        ax.text(0.5, 0.5, "Visualization unavailable", ha='center', va='center', transform=ax.transAxes)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)  # Fix: Explicitly close figure
    return buf

def explain_output(df: pd.DataFrame, user_query: str) -> str:
    if df.empty:
        return "No data available for this query."
    insights = basic_describe(df)
    summary_str = f"Rows: {insights['shape'][0]}, Columns: {insights['shape'][1]}"

    prompt = f"""
You are a senior business analyst. Provide concise, point-wise insights.

QUERY: {user_query}
SUMMARY: {summary_str}
COLUMNS: {list(df.columns)}
SAMPLE (top 5):
{df.head(5).to_string()}

REQUIREMENTS:
1. Each point on a new line with a bullet (•, -, or *).
2. Keep each point brief but descriptive.
3. Include specific numbers where relevant.
4. Focus on insights and actionable recommendations.

FORMAT EXAMPLE:
• Key finding with specific numbers
• Trend or comparison insight
• Brief implication
• Actionable recommendation
"""
    try:
        explanation_text = query_gemini(prompt)
        # Late import to avoid circulars when running outside app context
        from frontend.app import format_explanation_to_html
        return format_explanation_to_html(explanation_text)
    except Exception as e:
        logger.error(f"Explain failed: {e}")
        return f"{summary_str}\n\n• Review the data for patterns matching your query."

def execute_advanced_analysis(df: pd.DataFrame, user_query: str) -> Tuple[pd.DataFrame, List[Tuple[str, io.BytesIO]]]:
    charts = []
    df_final = df.copy()

    time_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if not time_cols or not num_cols:
        return df_final, charts

    time_col = None
    for col in time_cols:
        try:
            df_final[col] = pd.to_datetime(df_final[col], infer_datetime_format=False)  # Fix: Explicit format
            time_col = col
            break
        except Exception:
            continue
    if not time_col:
        return df_final, charts

    value_col = num_cols[0]

    try:
        df_forecast = forecast_time_series(df_final, time_col, value_col, periods=12)
        if not df_forecast.empty:
            buf = forecast_chart(df_forecast, value_col)
            charts.append((f"Forecast: {value_col}", buf))
    except Exception as e:
        logger.warning(f"Forecast failed: {e}")

    return df_final, charts