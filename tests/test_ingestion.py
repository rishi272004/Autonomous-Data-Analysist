# tests/test_ingestion.py
def test_imports():
    import backend.data_ingestion.excel_loader as el
    assert hasattr(el, "load_excel_to_dfs")
