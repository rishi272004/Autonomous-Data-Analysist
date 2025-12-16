# frontend/components/download_button.py
import streamlit as st
from backend.export.excel_exporter import df_to_csv, df_to_excel

def download_buttons(df, name_prefix="result"):
    csv_path = df_to_csv(df, f"tmp/{name_prefix}.csv")
    xlsx_path = df_to_excel(df, f"tmp/{name_prefix}.xlsx")
    st.download_button("Download CSV", csv_path, file_name=f"{name_prefix}.csv", mime="text/csv")
    st.download_button("Download Excel", xlsx_path, file_name=f"{name_prefix}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
