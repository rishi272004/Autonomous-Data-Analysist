# backend/query_engine/executor.py
import pandas as pd
import matplotlib.pyplot as plt
import io
from typing import Optional, List, Tuple
from sqlalchemy import create_engine, text
from backend.llm.ollama_client import query_gemini
from backend.utils.logger import get_logger
from backend.utils.config import DB_CONFIG
from backend.visualization.dashboard import recommend_chart
from backend.query_engine.insights import basic_describe
from backend.analysis.forecast import forecast_time_series, forecast_chart, forecast_by_group

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
        # Raise to allow callers to suppress UI output for failed sections
        raise
    finally:
        if engine: engine.dispose()

def prepare_query(sql: str, uploaded_files: list) -> str:
    if "..." in sql:
        file_names = "', '".join([f.filename for f in uploaded_files])
        sql = sql.replace("...", file_names)
    return sql

def generate_visualization(df: pd.DataFrame, chart_type: str = "auto", x: Optional[str] = None, y: Optional[str] = None) -> io.BytesIO:
    # Validate input
    if df is None or df.empty or len(df) < 2:
        logger.warning(f"Cannot visualize: df empty or too few rows (rows={len(df) if df is not None else 0})")
        # Return a placeholder image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data for visualization\n(Need at least 2 rows)", 
                ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        return buf
    
    plt.style.use('default')
    
    # Adjust figure size based on number of rows
    fig_width = max(10, min(16, len(df) * 0.8))
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
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
            # pick first numeric column for y to avoid ndarray errors
            y = y or (df.select_dtypes(include='number').columns.tolist()[0] if len(df.select_dtypes(include='number').columns) > 0 else None)
            if y is None:
                raise ValueError("No numeric column available for bar chart")
            
            # Create bar chart with better formatting
            bars = ax.bar(df[x].astype(str), df[y], color='skyblue', alpha=0.7, edgecolor='navy')
            ax.set_xlabel(x, fontsize=11, fontweight='bold')
            ax.set_ylabel(y, fontsize=11, fontweight='bold')
            ax.set_title(f"{y} by {x}", fontsize=13, fontweight='bold', pad=15)
            
            # Rotate x-axis labels if there are many categories
            if len(df) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}', ha='center', va='bottom', fontsize=9)
            
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
        elif chart_type == "line" and x and len(df.columns) > 1:
            y = y or (df.select_dtypes(include='number').columns.tolist()[0] if len(df.select_dtypes(include='number').columns) > 0 else None)
            if y is None:
                raise ValueError("No numeric column available for line chart")
            ax.plot(df[x].astype(str), df[y], marker='o', color='green', linewidth=2, markersize=8)
            ax.set_xlabel(x, fontsize=11, fontweight='bold')
            ax.set_ylabel(y, fontsize=11, fontweight='bold')
            ax.set_title(f"{y} over {x}", fontsize=13, fontweight='bold', pad=15)
            
            # Rotate x-axis labels if there are many categories
            if len(df) > 5:
                plt.xticks(rotation=45, ha='right')
            
            ax.grid(True, alpha=0.3, linestyle='--')
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
    plt.close()
    return buf

def explain_output(df: pd.DataFrame, user_query: str) -> str:
    if df.empty:
        return "No data available for this query. The query may need different column names or the data may not contain the requested information."

    # Basic summary
    n_rows, n_cols = df.shape
    summary_str = f"Rows: ({n_rows:,}), Columns: ({n_cols:,})"

    # Concise, point-wise prompt aligned with report style
    concise_prompt = f"""
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
        text = query_gemini(concise_prompt)
        # Format to HTML bullets/headings similar to report sections
        from frontend.app import format_explanation_to_html
        return format_explanation_to_html(text)
    except Exception as e:
        logger.error(f"Concise explanation failed: {e}")
        return format_explanation_to_html(f"• {summary_str}\n• Review the data for patterns matching your query.")

def execute_advanced_analysis(df: pd.DataFrame, user_query: str, generate_charts: bool = True) -> Tuple[pd.DataFrame, List[Tuple[str, io.BytesIO]]]:
    charts = []
    df_final = df.copy()
    time_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()
    if not time_cols or not num_cols: return df_final, charts
    time_col = None
    for col in time_cols:
        try:
            df_final[col] = pd.to_datetime(df_final[col])
            time_col = col
            break
        except Exception: continue
    if not time_col: return df_final, charts
    value_col = num_cols[0]
    try:
        q = user_query.lower()
        wants_forecast = any(k in q for k in ['forecast', 'predict', 'projection', 'predictive', 'future', 'next', 'estimate'])
        wants_by_product = any(k in q for k in ['product', 'item', 'sku'])
        group_col = None
        if wants_by_product:
            product_like = [c for c in df.columns if any(k in c.lower() for k in ['product', 'item', 'sku', 'name'])]
            if product_like:
                group_col = product_like[0]
        if wants_forecast:
            if group_col:
                df_group_fc = forecast_by_group(df_final, time_col, group_col, value_col, periods=12)
                if not df_group_fc.empty:
                    df_final = df_group_fc
            else:
                df_forecast = forecast_time_series(df_final, time_col, value_col, periods=12)
                if not df_forecast.empty and generate_charts:
                    buf = forecast_chart(df_forecast, value_col)
                    charts.append((f"Forecast: {value_col}", buf))
    except Exception as e:
        logger.warning(f"Forecast failed: {e}")
    return df_final, charts