# backend/analysis/forecast.py
import pandas as pd
from prophet import Prophet
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backend.utils.logger import get_logger
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)

def forecast_time_series(df: pd.DataFrame, time_col: str, value_col: str, periods: int = 12) -> pd.DataFrame:
    df_forecast = df[[time_col, value_col]].dropna()
    if df_forecast.empty:
        return pd.DataFrame()
    df_forecast.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_result.rename(columns={'yhat': f'{value_col}_pred'}, inplace=True)
    return forecast_result

def forecast_chart(df_forecast: pd.DataFrame, value_col: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_forecast['ds'], df_forecast[value_col + '_pred'], label='Forecast', color='blue')
    ax.fill_between(df_forecast['ds'], df_forecast['yhat_lower'], df_forecast['yhat_upper'], color='lightblue', alpha=0.4)
    ax.set_title(f'Forecast of {value_col}')
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

def forecast_by_group(df: pd.DataFrame, time_col: str, group_col: str, value_col: str, periods: int = 12) -> pd.DataFrame:
    results = []
    if df.empty:
        return pd.DataFrame()
    try:
        df_local = df.copy()
        df_local[time_col] = pd.to_datetime(df_local[time_col])
        # Ensure monthly frequency aggregation
        df_local['month'] = df_local[time_col].dt.to_period('M').dt.to_timestamp()
        for group_value, df_group in df_local.groupby(group_col):
            series = df_group.groupby('month', as_index=False)[value_col].sum()
            if series.empty:
                continue
            series.columns = ['ds', 'y']
            try:
                model = Prophet()
            except Exception:
                # If Prophet not available or fails to initialize
                continue
            try:
                model.fit(series)
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast = model.predict(future)
                forecast_result = forecast[['ds', 'yhat']].tail(periods)
                forecast_result.rename(columns={'yhat': 'forecast_total_monthly_sales'}, inplace=True)
                forecast_result[group_col] = group_value
                results.append(forecast_result[[group_col, 'ds', 'forecast_total_monthly_sales']])
            except Exception:
                continue
        if not results:
            return pd.DataFrame()
        final_df = pd.concat(results, ignore_index=True)
        final_df.rename(columns={'ds': 'month'}, inplace=True)
        # Order columns
        final_df = final_df[[group_col, 'month', 'forecast_total_monthly_sales']]
        return final_df
    except Exception as e:
        logger.warning(f"Grouped forecast failed: {e}")
        return pd.DataFrame()