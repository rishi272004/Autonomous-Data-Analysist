# frontend/components/query_box.py
import streamlit as st

def query_box(default=""):
    return st.text_area("Query (natural language)", default, height=120)
