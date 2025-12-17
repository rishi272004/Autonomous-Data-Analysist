# backend/query_engine/executor.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker
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

def format_number_label(value: float) -> str:
    """
    Format a number for display on charts, converting large numbers to readable format.
    Examples: 1000000 -> "1.0M", 1500 -> "1.5K", 100 -> "100"
    """
    if value == 0:
        return "0"
    
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs_val >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:,.0f}"

def format_axis_label(value, pos):
    """
    Format axis labels to avoid scientific notation.
    Used as matplotlib FuncFormatter.
    """
    return format_number_label(value)

def generate_visualization(df: pd.DataFrame, chart_type: str = "auto", x: Optional[str] = None, y: Optional[str] = None) -> io.BytesIO:
    """
    Generate a professional visualization from a DataFrame.
    Always returns a BytesIO object with PNG data, even if visualization fails.
    """
    fig = None
    try:
        # Validate input
        if df is None or df.empty:
            logger.warning(f"Cannot visualize: df is None or empty")
            return create_error_plot("No data to visualize")
        
        if len(df) < 2:
            logger.warning(f"Cannot visualize: insufficient rows (rows={len(df)})")
            return create_error_plot(f"Need at least 2 rows (have {len(df)})")
        
        # Set professional style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        matplotlib.rcParams['figure.facecolor'] = 'white'
        matplotlib.rcParams['axes.facecolor'] = '#f8f9fa'
        matplotlib.rcParams['font.size'] = 10
        
        # Adjust figure size based on number of rows
        fig_width = max(12, min(18, len(df) * 0.9))
        fig_height = 7
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Get recommendation for chart type
        rec_type, rec_col = recommend_chart(df)
        if chart_type == "auto":
            chart_type = rec_type
        
        if x is None:
            x = rec_col or (df.columns[0] if len(df.columns) > 0 else None)
        
        # Ensure x is a single column name, not a list
        if isinstance(x, list) and len(x) > 0:
            x = x[0]
        
        success = False
        
        # Try bar chart
        if chart_type == "bar" and x:
            try:
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric column available for bar chart")
                
                y = y or numeric_cols[0]
                
                # Check if both columns exist and have data
                if x not in df.columns or y not in df.columns:
                    raise ValueError(f"Columns {x} or {y} not found in dataframe")
                
                df_plot = df[[x, y]].dropna()
                if df_plot.empty:
                    raise ValueError("No valid data after dropping NaNs")
                
                # Create bar chart with professional styling
                bars = ax.bar(df_plot[x].astype(str), df_plot[y], 
                             color='#3498db', alpha=0.85, edgecolor='#2c3e50', linewidth=1.5)
                
                ax.set_xlabel(x, fontsize=12, fontweight='bold', labelpad=10)
                ax.set_ylabel(y, fontsize=12, fontweight='bold', labelpad=10)
                ax.set_title(f"{y} by {x}", fontsize=14, fontweight='bold', pad=20)
                
                # Format y-axis to avoid scientific notation
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_axis_label))
                
                # Rotate x-axis labels if there are many categories
                if len(df_plot) > 5:
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                else:
                    plt.xticks(fontsize=10)
                
                # Add value labels on bars with proper formatting
                for bar in bars:
                    height = bar.get_height()
                    if height != 0:
                        label_text = format_number_label(height)
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                success = True
            except Exception as e:
                logger.warning(f"Bar chart failed: {e}")
                success = False
        
        # Try line chart
        if not success and chart_type == "line" and x:
            try:
                numeric_cols = df.select_dtypes(include='number').columns.tolist()
                if not numeric_cols:
                    raise ValueError("No numeric column available for line chart")
                
                y = y or numeric_cols[0]
                
                if x not in df.columns or y not in df.columns:
                    raise ValueError(f"Columns {x} or {y} not found in dataframe")
                
                df_plot = df[[x, y]].dropna()
                if df_plot.empty:
                    raise ValueError("No valid data after dropping NaNs")
                
                # Create line chart with professional styling
                ax.plot(df_plot[x].astype(str), df_plot[y], marker='o', color='#2ecc71', 
                       linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1.5)
                
                ax.set_xlabel(x, fontsize=12, fontweight='bold', labelpad=10)
                ax.set_ylabel(y, fontsize=12, fontweight='bold', labelpad=10)
                ax.set_title(f"{y} over {x}", fontsize=14, fontweight='bold', pad=20)
                
                # Format y-axis
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_axis_label))
                
                if len(df_plot) > 5:
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                else:
                    plt.xticks(fontsize=10)
                
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                success = True
            except Exception as e:
                logger.warning(f"Line chart failed: {e}")
                success = False
        
        # Fallback: histogram or scatter
        if not success:
            try:
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) > 0:
                    # Use first numeric column for histogram
                    col = numeric_cols[0]
                    df_clean = df[col].dropna()
                    bins = min(30, max(10, len(df_clean) // 5))
                    ax.hist(df_clean, bins=bins, color='#9b59b6', edgecolor='#2c3e50', 
                           alpha=0.8, linewidth=1.5)
                    ax.set_title(f"Distribution of {col}", fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel(col, fontsize=12, fontweight='bold', labelpad=10)
                    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold', labelpad=10)
                    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_axis_label))
                    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_axis_label))
                    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                else:
                    # No numeric columns, show row count
                    categories = list(df.columns)[:10] if len(df.columns) > 0 else ["Data"]
                    counts = [len(df)] * len(categories)
                    ax.bar(range(len(categories)), counts, color='#e74c3c', edgecolor='#2c3e50', 
                          alpha=0.8, linewidth=1.5)
                    ax.set_title(f"Data Overview ({len(df)} rows)", fontsize=14, fontweight='bold', pad=20)
                    ax.set_xlabel("Columns", fontsize=12, fontweight='bold', labelpad=10)
                    ax.set_ylabel("Count", fontsize=12, fontweight='bold', labelpad=10)
                    plt.xticks(range(len(categories)), categories, rotation=45, ha='right', fontsize=10)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                success = True
            except Exception as e:
                logger.warning(f"Fallback visualization failed: {e}")
                success = False
        
        logger.info(f"Chart generation success: {success}")
        plt.tight_layout()
        
    except Exception as e:
        logger.error(f"Unexpected error in visualization: {e}")
        success = False
    
    # Always create a buffer and save, even if visualization failed
    buf = None
    try:
        if fig is None:
            logger.error("Figure not created, returning error plot")
            return create_error_plot("Failed to generate visualization")
        
        buf = io.BytesIO()
        logger.debug("Saving figure to buffer...")
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        buf_size = buf.getbuffer().nbytes
        logger.info(f"✓ Visualization saved successfully. Buffer size: {buf_size} bytes")
        if buf_size == 0:
            logger.error("Buffer is empty after savefig!")
            return create_error_plot("Generated empty visualization")
        return buf
    except Exception as e:
        logger.error(f"Failed to save visualization to buffer: {e}", exc_info=True)
        return create_error_plot("Failed to generate visualization")
    finally:
        # Always close figure to prevent memory leak
        if fig is not None:
            plt.close(fig)

def create_error_plot(message: str) -> io.BytesIO:
    """Create a professional error/message plot."""
    fig = None
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        
        # Create a professional error message display
        ax.text(0.5, 0.6, "📊 Visualization", ha='center', va='center', 
                transform=ax.transAxes, fontsize=18, fontweight='bold', color='#2c3e50')
        
        ax.text(0.5, 0.45, message, ha='center', va='center', 
                transform=ax.transAxes, fontsize=14, color='#555',
                bbox=dict(boxstyle='round,pad=1', facecolor='#e8f4f8', edgecolor='#3498db', linewidth=2))
        
        ax.text(0.5, 0.15, "Your data is being analyzed...", ha='center', va='center', 
                transform=ax.transAxes, fontsize=11, color='#7f8c8d', style='italic')
        
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Save to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        logger.debug(f"Error plot created, buffer size: {buf.getbuffer().nbytes} bytes")
        return buf
    except Exception as e:
        logger.error(f"Failed to create error plot: {e}", exc_info=True)
        # Return a minimal valid image buffer as last resort
        buf = io.BytesIO()
        try:
            fig2, ax2 = plt.subplots(figsize=(1, 1))
            ax2.text(0.5, 0.5, "Error", ha='center', va='center', fontsize=8)
            ax2.axis('off')
            fig2.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close(fig2)
        except:
            pass
        return buf
    finally:
        if fig is not None:
            plt.close(fig)



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
    """
    Perform advanced analysis on the dataframe (forecasting, grouping, etc.)
    Returns the modified dataframe and any generated charts.
    """
    charts = []
    df_final = df.copy()
    
    if df_final is None or df_final.empty or not generate_charts:
        return df_final, charts
    
    try:
        time_cols = df.select_dtypes(include=['datetime', 'object']).columns.tolist()
        num_cols = df.select_dtypes(include='number').columns.tolist()
        
        # Try to find and convert a time column
        time_col = None
        if time_cols:
            for col in time_cols:
                try:
                    df_final[col] = pd.to_datetime(df_final[col])
                    time_col = col
                    break
                except Exception:
                    continue
        
        # If we have both time and numeric columns, try forecasting
        if time_col and num_cols:
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
                        df_group_fc = forecast_by_group(df_final, time_col, group_col, num_cols[0], periods=12)
                        if not df_group_fc.empty:
                            df_final = df_group_fc
                    else:
                        df_forecast = forecast_time_series(df_final, time_col, num_cols[0], periods=12)
                        if not df_forecast.empty:
                            try:
                                buf = forecast_chart(df_forecast, num_cols[0])
                                if buf and buf.getvalue():
                                    charts.append((f"Forecast: {num_cols[0]}", buf))
                            except Exception as fc_err:
                                logger.warning(f"Forecast chart failed: {fc_err}")
            except Exception as fc_error:
                logger.warning(f"Forecast analysis failed: {fc_error}")
        
        # If no charts generated yet, try creating a basic visualization
        if not charts and len(df_final) >= 2:
            try:
                viz_buf = generate_visualization(df_final)
                if viz_buf and viz_buf.getvalue():
                    charts.append(("Summary Visualization", viz_buf))
            except Exception as viz_error:
                logger.warning(f"Basic visualization failed: {viz_error}")
    
    except Exception as e:
        logger.error(f"Advanced analysis error: {e}")
    
    return df_final, charts