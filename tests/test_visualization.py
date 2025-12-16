# tests/test_visualization.py
def test_charts_import():
    from backend.visualization.charts import df_to_table_image
    assert callable(df_to_table_image)
