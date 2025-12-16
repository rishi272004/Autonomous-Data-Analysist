# frontend/components/output_display.py
import streamlit as st
import pandas as pd

def show_dataframe(df: pd.DataFrame):
    st.write(df)
