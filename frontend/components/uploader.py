# frontend/components/uploader.py
import streamlit as st
from backend.data_ingestion.excel_loader import load_excel_to_dfs, sanitize_dataframe
import tempfile

def uploader_component():
    uploaded_file = st.file_uploader("Upload Excel/CSV", type=['xls','xlsx','csv'])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded_file.name.split(".")[-1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        dfs = load_excel_to_dfs(tmp_path)
        dfs = {k: sanitize_dataframe(v) for k,v in dfs.items()}
        return dfs
    return None
